import math
import logging  
import time     
import random  
import json     
import numpy as np
import requests
import os
import cv2
import torch    # 🚀 Cuda/Device yönetimi için şart
import sys
from PIL import Image                                   
from collections import deque                           
from transformers import AutoImageProcessor, AutoModel   
from ultralytics import YOLO 
from .constants import classes, landing_statuses, motion_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction
from azure.iot.device import IoTHubDeviceClient
from dotenv import load_dotenv

# Kök dizindeki .env dosyasını belleğe yükler
load_dotenv()

class MockBox:
    """ Kodunun alt kısımlarındaki box.id, box.conf yapılarını bozmamak için sahte sınıf """
    def __init__(self, x1, y1, x2, y2, conf, cls_id, tid):
        self.xyxy = torch.tensor([[x1, y1, x2, y2]]) 
        self.conf = [conf]
        self.cls = [torch.tensor(cls_id)]
        self.id = [torch.tensor(tid)]

class AeroThermalTracker:
    def __init__(self, alpha=0.45, max_missed_frames=3, min_hits_to_start=3):
        self.alpha = alpha
        self.max_missed_frames = max_missed_frames
        self.min_hits_to_start = min_hits_to_start
        self.last_box = None  
        self.missed_counter = 0
        self.hit_counter = 0 

    def update(self, prediction):
        if prediction is not None:
            self.hit_counter += 1
            self.missed_counter = 0
            if self.hit_counter < self.min_hits_to_start:
                return None
            current_box = np.array([
                prediction["top_left_x"], prediction["top_left_y"],
                prediction["bottom_right_x"], prediction["bottom_right_y"]
            ])
            if self.last_box is None:
                self.last_box = current_box
            else:
                self.last_box = self.alpha * current_box + (1 - self.alpha) * self.last_box
            prediction["top_left_x"] = int(self.last_box[0])
            prediction["top_left_y"] = int(self.last_box[1])
            prediction["bottom_right_x"] = int(self.last_box[2])
            prediction["bottom_right_y"] = int(self.last_box[3])
            return prediction
        else:
            self.hit_counter = max(0, self.hit_counter - 1)
            if self.last_box is not None and self.missed_counter < self.max_missed_frames and self.hit_counter >= self.min_hits_to_start:
                self.missed_counter += 1
                return {
                    "top_left_x": int(self.last_box[0]), "top_left_y": int(self.last_box[1]),
                    "bottom_right_x": int(self.last_box[2]), "bottom_right_y": int(self.last_box[3]),
                    "confidence": 0.30, "status": "TRACKING_FROM_MEMORY"
                }
            else:
                self.last_box = None
                return None


class BayesianScaleFilter: # pozisyon kestirimi için drift sayısını azaltmakta yardımcı olur.
    """
    PTC-Depth makalesi baz alinarak yazilmiş, 1 Boyutlu Kalman Filtresi (Recursive Bayesian Update).
    İHA'nin optik akiş metrik ölçeğini (scale) bellek tüketmeden, varyans/şüphe hesaplayarak düzenleştirir.
    """
    def __init__(self, initial_scale=1.0, initial_variance=10.0):   # 100 DEN 10 YAPILDI 
        # Scale ve varyans (belirsizlik) baslangic state'leri
        # Ilk kalkista scale'i bilmedigimiz icin varyans kasten cok yuksek (100.0)
        self.prior_scale = initial_scale
        self.prior_variance = initial_variance

    def update(self, measured_scale, measurement_variance=0.05):  # Hata toleransı 0.1'den 0.05'e çekildi (daha agresif güven)
        """
        yeni bir ölçüm geldiğinde ölçeği ve belirsizliği günceller.
        measurement_variance (R): Sensörün, Optik Akişin donanimsal hata payi sabitidir.
        """
        # Kalman Gain (K): Yeni gelen optik akis olcumune mi guvenelim, eski hafizaya mi?
        kalman_gain = self.prior_variance / (self.prior_variance + measurement_variance)  # 100/100+0.1 = 0.99

        # Adım 2: ölçeği güncelle (S_yeni) - Eski tahmin ile yeni ölçümü harmanla
        estimated_scale = self.prior_scale + kalman_gain * (measured_scale - self.prior_scale)

        # Adım 3: Şüpheyi/Belirsizliği Güncelle (P_yeni) - Sistem tecrübe kazandıkça şüphe azalır
        self.prior_variance = (1.0 - kalman_gain) * self.prior_variance

        # Gelecek kare için eski hafızayı güncelle
        self.prior_scale = estimated_scale

        return self.prior_scale


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        
        self.model = YOLO("weights/best.pt") 

        # Azure IoT Hub Bağlantısı
        self.azure_connection_string = os.getenv("AZURE_IOT_CONNECTION_STRING")
        
        try:
            self.azure_client = IoTHubDeviceClient.create_from_connection_string(self.azure_connection_string)  # type: ignore
            self.azure_client.connect()
            logging.info("Azure IoT Hub bağlantısı başarıyla başlatıldı.")
        except Exception as e:
            logging.error(f"Azure IoT Hub başlatılamadı (Sistem offline/yerel modda çalışacak): {e}")
            self.azure_client = None


        
        self.yaw_history = []
        self.vehicle_history = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dino_dir = "./weights/dinov2-base"
        
        # Eğer klasör yoksa veya içi boşsa ilk seferlik internetten indirip diske mühürlüyoruz
        if not os.path.exists(dino_dir) or not os.listdir(dino_dir):
            logging.info("[FIRST RUN] DINOv2 weights not found locally. Downloading from Hugging Face Hub...")
            os.makedirs(dino_dir, exist_ok=True)
            
            # İnternetten geçici olarak çek
            temp_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            temp_model = AutoModel.from_pretrained("facebook/dinov2-base")
            
            # Belirttiğin weights klasörünün altına kalıcı olarak kaydet
            temp_processor.save_pretrained(dino_dir)
            temp_model.save_pretrained(dino_dir)
            logging.info(f"[SUCCESS] DINOv2 weights downloaded and permanently deployment-ready at {dino_dir}")
            
            # Bellek temizliği
            del temp_processor, temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Kod bu satıra geldiğinde model kesinlikle disktedir. İnternet kopsa bile yerelden ayağa kalkar.
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_dir, local_files_only=True)
        self.dino_model = AutoModel.from_pretrained(dino_dir, local_files_only=True).to(self.device).eval()
        self.precomputed_3rd_refs = {}
        self.goraev_trackers = {}

        # ────────────── KAMERA KALİBRASYON MATRİSLERİ ──────────────
        # RGB Kamera 4K Matrisi (3000x4000)
        self.K_rgb_4k = np.array([
            [2792.2, 0.0, 1988.0],
            [0.0, 2795.2, 1562.2],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Termal Kamera Matrisi (512x640)
        self.K_termal = np.array([
            [731.7965, 0.0, 319.2367],
            [0.0, 732.0172, 251.2424],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # RGB Kamera 1080p Matrisi (1080x1920)
        self.K_rgb_1080p = np.array([
            [1389.7, 0.0, 954.007],
            [0.0, 1387.1, 558.896],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Varsayılan başlangıç kamerası
        self.K = self.K_rgb_4k

        # Odometry changes
        self.prev_gray = None   # geçmiş fotoğraf hafızası
        self.prev_points = None # geçmiş nokta hafızası
        self.current_x = 0.0    # anlık X konumu
        self.current_y = 0.0    # anlık Y konumu
        self.current_z = -0.1   # anlık Z konumu (irtifa)
        self.prev_gt_x = 0.0    
        self.prev_gt_y = 0.0
        self.prev_gt_z = 0.0
        self.scale_filter = BayesianScaleFilter(initial_scale=1.0, initial_variance=10.0)
        self.current_yaw = 0.0  # Uçağın pusulası (Radyan cinsinden)
        self.smooth_global_dx = 0.0  # Atalet koruyucu X hızı
        self.smooth_global_dy = 0.0  # Atalet koruyucu Y hızı
        
        self.global_R = np.eye(3, dtype=np.float64)

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1, auth_token=None):
        """ auth_token parametresi fonksiyon imnasına eklendi ve 401 hatası çözüldü kanka """
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]
        
        # 🚀 GÜVENLİK GÜNCELLEMESİ: İşletim sistemi yolları os.path.join ile zırhlandı
        target_path = os.path.join(images_folder, image_name)
        
        if image_name not in images_files:
            headers = {'Authorization': f'Token {auth_token}'} if auth_token else {}
            
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, headers=headers, timeout=60)
                    response.raise_for_status()
                    
                    img_bytes = response.content
                    with open(target_path, 'wb') as img_file:
                        img_file.write(img_bytes)

                    t2 = time.perf_counter()
                    logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {target_path}')
                    return

                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files, auth_token=None, *args, **kwargs):
        """ auth_token burada yakalanıp alt fonksiyona üfleniyor """
        self.download_image(
            evaluation_server_url + "media" + prediction.image_url, 
            images_folder, 
            images_files,
            auth_token=auth_token
        )
        
        image_name = prediction.image_url.split("/")[-1]
        image_path = os.path.join(images_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            logging.error(f"gorsel okunamadi: {image_path}")
            return prediction
        self._bind_camera_matrix(frame)
            
        print(f"\n Okunan Görsel: {image_path} | Boyut: {frame.shape}")

        # 1. GÖREV: Nesne Tespiti (YOLOv8s ile Tasit / Insan / UAP / UAI Tespiti)
        new_prediction = self.detect(prediction, health_status, frame)

        if new_prediction is not None:
            prediction = new_prediction
        else:
            logging.error("DETECT FONKSİYONU HATA YAPTI, ESKİ VERİ İLE DEVAM EDİLİYOR.")
        
        # 2. GÖREV: Pozisyon Kestirimi (Optik Akış / Odometri / Sürüklenme Hesabı)
        self.track_optical_flow(frame, prediction, health_status)
        
        is_valid = self.validate_output(prediction)

        if not is_valid:
            logging.info("Model tespit yapamadı, kare atlandı.")
        
        # ==================================================================
        #  DINOv2
        # ==================================================================
        active_refs = kwargs.get('active_refs', [])
        
        # Fallback: If active_refs is null, default to evaluating all precomputed RAM references
        if not active_refs and self.precomputed_3rd_refs:
            active_refs = [{"url": k} for k in self.precomputed_3rd_refs.keys()]

        if self.precomputed_3rd_refs and active_refs:
            h, w = frame.shape[:2]

            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_inputs = self.dino_processor(
                images=frame_image, return_tensors="pt", do_resize=True, 
                size={"height": 448, "width": 448}, do_center_crop=False
            ).to(self.device)

            with torch.no_grad():
                frame_outputs = self.dino_model(**frame_inputs)
                patch_features = frame_outputs.last_hidden_state[:, 1:, :]
                patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

            raw_candidates = []
            for ref_item in active_refs:
                ref_url = ref_item.get('url')
                if ref_url not in self.precomputed_3rd_refs:
                    continue
                    
                obj_id, ref_feature = self.precomputed_3rd_refs[ref_url]
                similarity = torch.matmul(patch_features, ref_feature.t()).squeeze()
                similarity_map = similarity.reshape(32, 32)

                raw_prediction = self.process_similarity_and_filter_3rd(similarity_map, similarity, w, h)
                prediction_3rd = self.goraev_trackers[ref_url].update(raw_prediction)

                if prediction_3rd:
                    prediction_3rd["ref_name"] = ref_url  
                    prediction_3rd["score"] = raw_prediction["confidence"] if raw_prediction else 0.20
                    raw_candidates.append(prediction_3rd)

            clean_predictions = []
            if raw_candidates:
                raw_candidates = sorted(raw_candidates, key=lambda x: x["score"], reverse=True)
                for cand in raw_candidates:
                    keep = True
                    box_cand = [cand["top_left_x"], cand["top_left_y"], cand["bottom_right_x"], cand["bottom_right_y"]]
                    
                    for accepted in clean_predictions:
                        box_acc_fixed = [accepted["top_left_x"], accepted["top_left_y"], accepted["bottom_right_x"], accepted["bottom_right_y"]]
                        if self.calculate_iou(box_cand, box_acc_fixed) > 0.35:
                            keep = False 
                            break
                    if keep:
                        clean_predictions.append(cand)

            for pred in clean_predictions:
                ref_url = pred["ref_name"]  
                current_frame_url = prediction.frame_url  

                ref_pred_obj = ReferencePrediction(
                    reference_url=ref_url,
                    frame_url=current_frame_url,
                    top_left_x=float(pred["top_left_x"]),
                    top_left_y=float(pred["top_left_y"]),
                    bottom_right_x=float(pred["bottom_right_x"]),
                    bottom_right_y=float(pred["bottom_right_y"])
                )
                prediction.add_reference_prediction(ref_pred_obj)
                print(f"   [Target Locked]: {ref_url} başarıyla eşleşti koordinatlar: {pred['top_left_x']},{pred['top_left_y']}")
                
        return prediction
    
    def _bind_camera_matrix(self, frame):
        """Gelen çerçevenin genişliğine göre doğru intrinsics matrisini kilitler."""
        h, w = frame.shape[:2]
        if w < 1000:
            self.K = self.K_termal
        elif w < 2500:
            self.K = self.K_rgb_1080p
        else:
            self.K = self.K_rgb_4k

    def calculate_iou(self , box1,box2): # Kesişim alanı / Birleşim alanı hesabı
        x1_inter = max(box1[0],box2[0])
        y1_inter = max(box1[1],box2[1])  
        x2_inter = min(box1[2],box2[2])    
        y2_inter = min(box1[3],box2[3])

        inter_w = max(0, x2_inter - x1_inter) 
        inter_h = max(0,y2_inter - y1_inter)
        inter_area = inter_w * inter_h

        if inter_area == 0: 
            return 0.0  

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])  

        union_area = area1 + area2 - inter_area     

        return inter_area / union_area

    def detect(self, prediction, health_status, frame, *args, **kwargs):        # Görev 1: NESNE TESPİTİ (YOLOv8s)
        
        h, w = frame.shape[:2]    
    
        results_list = []
        track_id_counter = 1000  # Yeni nesneler için başlangıç ID'si
        
        yolo_outputs = self.model(frame, verbose=False, device=self.device)
        
        for box in yolo_outputs[0].boxes:
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            
            if conf > 0.20:  # Güvenlik eşiği
                # Extract bounding box coordinates relative to native image resolution
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                matched_id = track_id_counter
                
                # Temporal consistency check against tracked object history
                for tid, v_data in self.vehicle_history.items():
                    dist = math.sqrt((cx - v_data['x'])**2 + (cy - v_data['y'])**2)
                    if dist < 40.0:  # Retain tracking ID if object falls within a 40-pixel proximity radius
                        matched_id = tid
                        break
                
                if matched_id == track_id_counter:
                    track_id_counter += 1
                    
                # Küresel MockBox sınıfına temiz elenmiş veriyi gönderiyoruz
                results_list.append(MockBox(x1, y1, x2, y2, conf, cls_id, matched_id))
                
        class MockResults:
            def __init__(self, boxes):
                self.boxes = boxes

        results = [MockResults(results_list)]

        print(f"YOLOv8 bu karede toplam {len(results[0].boxes)} gerçek nesne yakaladi")

        current_frame_ids = []

        # [Stabilization Shield] - High-resolution motion tracking to handle airframe vibrations
        if not hasattr(self, 'prev_gray_detect'):
            self.prev_gray_detect = None

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera_matrix = None

        if self.prev_gray_detect is not None:
            bg_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            for box in results[0].boxes:
                if box.id is None: 
                    continue
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(bg_mask, (bx1, by1), (bx2, by2), 0, -1)

            p0 = cv2.goodFeaturesToTrack(self.prev_gray_detect, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=bg_mask)
            if p0 is not None and len(p0) > 10:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_detect, current_gray, p0, None)
                if p1 is not None:
                    good_p0 = p0[st == 1]
                    good_p1 = p1[st == 1]
                    if len(good_p0) > 10:
                        camera_matrix, inliers = cv2.estimateAffinePartial2D(good_p0, good_p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)

        self.prev_gray_detect = current_gray.copy()
        
        gt_x = float(prediction.gt_translation_x)
        gt_y = float(prediction.gt_translation_y)
        gt_z = float(prediction.gt_translation_z)
        
        if not hasattr(self, 'prev_gt_x_track'):
            self.prev_gt_x_track = gt_x
            self.prev_gt_y_track = gt_y

        delta_cam_world_x = gt_x - self.prev_gt_x_track
        delta_cam_world_y = gt_y - self.prev_gt_y_track

        # 🎯 ANA TESPİT VE KARAR DÖNGÜSÜ
        for box in results[0].boxes:
            if box.id is None:
                continue
            conf = float(box.conf[0])
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls_id == 0 and conf < 0.40:    
                continue
            elif cls_id == 1 and conf < 0.20:  
                continue
            elif cls_id in [2, 3] and conf < 0.60: 
                continue

            moving_status = motion_statuses["Stationary"] 

            track_id = int(box.id[0].item())
            current_frame_ids.append(track_id)
            
            top_left_x = float(x1)
            top_left_y = float(y1)
            bottom_right_x = float(x2)
            bottom_right_y = float(y2)

            center_x = (top_left_x + bottom_right_x) / 2.0
            center_y = (top_left_y + bottom_right_y) / 2.0

            if cls_id == 0:
                cls = classes["Vehicle"]
                landing_status = landing_statuses["Not_Landing_Zone"] 
                moving_status = motion_statuses["Stationary"]
                
                if track_id in self.vehicle_history:
                    prev_data = self.vehicle_history[track_id]
                    age = prev_data['age'] + 1
                    
                    if camera_matrix is not None:
                        pred_x = camera_matrix[0,0] * prev_data['x'] + camera_matrix[0,1] * prev_data['y'] + camera_matrix[0,2]
                        pred_y = camera_matrix[1,0] * prev_data['x'] + camera_matrix[1,1] * prev_data['y'] + camera_matrix[1,2]
                        net_movement = math.sqrt((center_x - pred_x)**2 + (center_y - pred_y)**2)
                    else:
                        net_movement = ((center_x - prev_data['x'])**2 + (center_y - prev_data['y'])**2) ** 0.5
                    
                    dynamic_threshold = 3.5
                    if net_movement > dynamic_threshold:
                        moving_status = motion_statuses["Moving"]
                        
                    self.vehicle_history[track_id] = {'x': center_x, 'y': center_y, 'age': age}
                else:
                    age = 1
                    self.vehicle_history[track_id] = {'x': center_x, 'y': center_y, 'age': age}

                if age < 3:
                    continue

            elif cls_id == 1:
                cls = classes["Pedestrian"]
                landing_status = landing_statuses["Not_Landing_Zone"] 
                moving_status = motion_statuses["Stationary"]  # Varsayılan olarak sabit kabul ediyoruz
                
                #  Yayalar için de sarsıntı kalkanı ve hareket takibini devreye alıyoruz kanka
                if track_id in self.vehicle_history:
                    prev_data = self.vehicle_history[track_id]
                    age = prev_data['age'] + 1
                    
                    if camera_matrix is not None:
                        pred_x = camera_matrix[0,0] * prev_data['x'] + camera_matrix[0,1] * prev_data['y'] + camera_matrix[0,2]
                        pred_y = camera_matrix[1,0] * prev_data['x'] + camera_matrix[1,1] * prev_data['y'] + camera_matrix[1,2]
                        net_movement = math.sqrt((center_x - pred_x)**2 + (center_y - pred_y)**2)
                    else:
                        net_movement = ((center_x - prev_data['x'])**2 + (center_y - prev_data['y'])**2) ** 0.5
                    
                    # Lowered pedestrian threshold to 1.8 pixels to account for slower displacement speed
                    dynamic_threshold = 1.8
                    if net_movement > dynamic_threshold:
                        moving_status = motion_statuses["Moving"]
                        
                    self.vehicle_history[track_id] = {'x': center_x, 'y': center_y, 'age': age}
                else:
                    age = 1
                    self.vehicle_history[track_id] = {'x': center_x, 'y': center_y, 'age': age}

                # Yanlış alarmları önlemek için en az 2 kare görünmesini bekliyoruz
                if age < 2:
                    continue
                
            elif cls_id in [2, 3]: 
                cls = classes["Landing_Zone_1"] if cls_id == 2 else classes["Landing_Zone_2"]
                landing_status = landing_statuses["Landable"] 
                moving_status = motion_statuses["Stationary"]

                margin = 5.0 
                if top_left_x <= margin or top_left_y <= margin or bottom_right_x >= (w - margin) or bottom_right_y >= (h - margin):
                    landing_status = landing_statuses["Unlandable"]

                uap_coords = [x1, y1, x2, y2]
                for other_box in results[0].boxes:
                    other_cls = int(other_box.cls[0].item())

                    if other_cls in [0, 1]:
                        other_coords = other_box.xyxy[0].tolist()
                        iou_val = self.calculate_iou(uap_coords, other_coords)        

                        if iou_val > 0.1:   
                            landing_status = landing_statuses["Unlandable"]
                            break
            else:
                continue 


            d_obj = DetectedObject(cls,
                                   landing_status,
                                   moving_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            
            prediction.add_detected_object(d_obj)

        keys_to_remove = [k for k in self.vehicle_history.keys() if k not in current_frame_ids]
        for k in keys_to_remove:
            del self.vehicle_history[k]
            
        self.prev_gt_x_track = gt_x
        self.prev_gt_y_track = gt_y

        return prediction
    
    def track_optical_flow(self, frame, prediction, health_status):
        h, w = frame.shape[:2]
        
        if w < 1000:
            self.K = self.K_termal
            logging.info(f"📡 [OTOMATİK KONTROL]: Termal Kamera Modu Aktif ({w}x{h})")
        elif w < 2500:
            self.K = self.K_rgb_1080p
            logging.info(f"📡 [OTOMATİK KONTROL]: RGB 1080p Kamera Modu Aktif ({w}x{h})")
        else:
            self.K = self.K_rgb_4k
            logging.info(f"📡 [OTOMATİK KONTROL]: RGB 4K Kamera Modu Aktif ({w}x{h})")

        current_gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        current_gray = clahe.apply(current_gray_raw)

        if self.prev_gray is not None and self.prev_gray.shape != current_gray.shape:
            logging.warning(f"OPTİK AKIŞ Çözünürlük değişimi algilandi. Hafiza güvenli sekilde sifirlaniyor.")
            self.prev_gray = None
            self.prev_points = None

        good_new = []
        cx, cy = self.K[0, 2], self.K[1, 2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        center_array = np.array([cx, cy], dtype=np.float32).reshape(1, 1, 2)

        flow_mask = np.ones(current_gray.shape, dtype=np.uint8) * 255
        if hasattr(prediction, 'detected_objects') and prediction.detected_objects:
            for obj in prediction.detected_objects:
                obj_cls = obj.cls[0] if isinstance(obj.cls, (list, tuple)) else obj.cls
                if int(obj_cls) == classes["Vehicle"]:
                    tx = max(0, int(obj.top_left_x))
                    ty = max(0, int(obj.top_left_y))
                    bx = min(w, int(obj.bottom_right_x))
                    by = min(h, int(obj.bottom_right_y))
                    cv2.rectangle(flow_mask, (tx, ty), (bx, by), 0, -1)


        if health_status == '0':
            if not hasattr(self, '_gps_handoff_done'):
                if hasattr(self, 'prev_gt_x'):
                    self.current_x = self.prev_gt_x
                    self.current_y = self.prev_gt_y
                    self.current_z = self.prev_gt_z
                    self.cruise_speed_dx = self.smooth_global_dx
                    self.cruise_speed_dy = self.smooth_global_dy
                self._gps_handoff_done = True

            tracking_success = False

            if self.prev_gray is not None and self.prev_points is not None:
                
                if w < 1000:
                    next_points, status, error = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, current_gray, self.prev_points, None, winSize=(45, 45), maxLevel=4
                    )
                    back_points, back_status, back_error = cv2.calcOpticalFlowPyrLK(
                        current_gray, self.prev_gray, next_points, None, winSize=(45, 45), maxLevel=4
                    )
                    
                    diff = np.linalg.norm((self.prev_points - back_points).reshape(-1, 2), axis=1)
                    fb_valid = (diff < 1.5) & (status.ravel() == 1) & (back_status.ravel() == 1)
                    
                    good_new = next_points[fb_valid]
                    good_old = self.prev_points[fb_valid]

                    keep_indices = []
                    good_new = good_new.reshape(-1, 2)
                    
                    for idx, pt in enumerate(good_new):
                        nx, ny = int(pt[0]), int(pt[1])
                        if 0 <= nx < w and 0 <= ny < h and flow_mask[ny, nx] == 255:
                            keep_indices.append(idx)

                    tracking_success = False
                    if len(keep_indices) >= 6:
                        good_new = good_new[keep_indices]
                        good_old = good_old[keep_indices]

                        good_old_centered = good_old - center_array
                        good_new_centered = good_new - center_array

                        transform_matrix, inliers = cv2.estimateAffinePartial2D(
                            good_old_centered, good_new_centered, method=cv2.RANSAC, ransacReprojThreshold=2.5
                        )
                        
                        if transform_matrix is not None:
                            total_points = len(good_new)
                            inlier_count = int(np.sum(inliers.ravel() == 1)) if inliers is not None else 0
                            inlier_ratio = inlier_count / total_points if total_points > 0 else 0.0

                            if inlier_ratio >= 0.35:
                                tracking_success = True
                                a = transform_matrix[0, 0]
                                c = transform_matrix[1, 0]
                                delta_yaw = math.atan2(c, a)
                                
                                if abs(delta_yaw) < 0.005: 
                                    delta_yaw = 0.0
                                    
                                self.current_yaw += delta_yaw

                                z_safe = max(abs(float(self.current_z)), 2.0)
                                dx_pixel = transform_matrix[0, 2]
                                dy_pixel = transform_matrix[1, 2]

                                forward_m = (dy_pixel / fy) * z_safe * self.scale_filter.prior_scale
                                right_m = -(dx_pixel / fx) * z_safe * self.scale_filter.prior_scale

                                raw_global_dx = forward_m * math.cos(self.current_yaw) - right_m * math.sin(self.current_yaw)
                                raw_global_dy = forward_m * math.sin(self.current_yaw) + right_m * math.cos(self.current_yaw)

                                if hasattr(self, 'smooth_global_dx'):
                                    max_allowable_accel = 0.60  
                                    accel_x = raw_global_dx - self.smooth_global_dx
                                    if abs(accel_x) > max_allowable_accel:
                                        raw_global_dx = self.smooth_global_dx + (math.copysign(max_allowable_accel, accel_x))
                                        
                                    accel_y = raw_global_dy - self.smooth_global_dy
                                    if abs(accel_y) > max_allowable_accel:
                                        raw_global_dy = self.smooth_global_dy + (math.copysign(max_allowable_accel, accel_y))

                                max_physical_speed = 6.0  
                                raw_global_dx = max(min(raw_global_dx, max_physical_speed), -max_physical_speed)
                                raw_global_dy = max(min(raw_global_dy, max_physical_speed), -max_physical_speed)

                                alpha_blend = 0.20 if abs(delta_yaw) > 0.01 else 0.45
                                
                                self.smooth_global_dx = ((1.0 - alpha_blend) * self.smooth_global_dx) + (alpha_blend * raw_global_dx)
                                self.smooth_global_dy = ((1.0 - alpha_blend) * self.smooth_global_dy) + (alpha_blend * raw_global_dy)
                                
                                self.current_x += self.smooth_global_dx
                                self.current_y += self.smooth_global_dy
                                good_new = good_new[inliers.ravel() == 1] if inliers is not None else good_new

                #  RGB MOTOR (w >= 1000)
                else:
                    if hasattr(self, 'last_dx_pixel') and hasattr(self, 'last_dy_pixel'):
                        motion_mag = math.sqrt(self.last_dx_pixel**2 + self.last_dy_pixel**2)
                        win_edge = int(21 + min(motion_mag * 0.5, 10))
                        if win_edge % 2 == 0: win_edge += 1
                    else:
                        win_edge = 21

                    next_points, status, error = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, current_gray, self.prev_points, None, winSize=(win_edge, win_edge), maxLevel=3
                    )
                    good_new = next_points[status == 1]
                    good_old = self.prev_points[status == 1]

                    keep_indices = []
                    for idx, pt in enumerate(good_new):
                        nx, ny = int(pt[0]), int(pt[1])
                        if 0 <= nx < w and 0 <= ny < h and flow_mask[ny, nx] == 255:
                            keep_indices.append(idx)

                    if len(keep_indices) >= 6:
                        good_new = good_new[keep_indices]
                        good_old = good_old[keep_indices]

                        good_old_centered = good_old - center_array
                        good_new_centered = good_new - center_array

                        transform_matrix, inliers = cv2.estimateAffinePartial2D(
                            good_old_centered, good_new_centered, method=cv2.RANSAC, ransacReprojThreshold=3.0
                        )
                        
                        if transform_matrix is not None:
                            total_points = len(good_new)
                            inlier_count = int(np.sum(inliers.ravel() == 1)) if inliers is not None else 0
                            inlier_ratio = inlier_count / total_points if total_points > 0 else 0.0

                            if inlier_ratio >= 0.40:
                                tracking_success = True
                                
                                a = transform_matrix[0, 0]
                                c = transform_matrix[1, 0]
                                delta_yaw = math.atan2(c, a)
                                max_yaw_per_frame = 0.07  
                                delta_yaw = max(min(delta_yaw, max_yaw_per_frame), -max_yaw_per_frame)
                                self.current_yaw += delta_yaw

                                if len(good_new) > 1:
                                    dists_old = np.linalg.norm(good_old[:-1] - good_old[1:], axis=1)
                                    dists_new = np.linalg.norm(good_new[:-1] - good_new[1:], axis=1)
                                    valid_dists = (dists_old > 0.1)
                                    scale_change = np.median(dists_new[valid_dists] / dists_old[valid_dists]) if np.any(valid_dists) else 1.0
                                else:
                                    scale_change = 1.0

                                if 0.8 < scale_change < 1.2:
                                    # 🎯 Sıfır kilidini kıran mutlak çapa
                                    absolute_z = 30.0 + self.current_z
                                    target_absolute_z = absolute_z / scale_change
                                    delta_z = target_absolute_z - absolute_z
                                    
                                    #  HIZA DUYARLI ADAPTİF FİLTRE KALKANI
                                    scale_deviation = abs(1.0 - scale_change)
                                    
                                    if scale_deviation > 0.02:
                                        # İHA harbi hızlı aşağı/yukarı gidiyor! Kapıları aç, hızlı tepki ver.
                                        damping_factor = 0.35
                                        max_z_change = 0.80
                                    else:
                                        # İHA stabil uçuyor, gelen oynamalar sadece gürültü. Sıkı sönümle.
                                        damping_factor = 0.05
                                        max_z_change = 0.10
                                    
                                    # Formülü dinamik parametrelerle besliyoruz
                                    delta_z = delta_z * damping_factor
                                    delta_z_clamped = max(min(delta_z, max_z_change), -max_z_change)
                                    
                                    self.current_z += delta_z_clamped
                                    self.current_z = max(min(self.current_z, 50.0), -50.0)
                                
                                # Gerçek dünya çarpanı pürüzsüzleşti
                                z_safe = max(abs(30.0 + float(self.current_z)), 2.0)
                                dx_pixel = transform_matrix[0, 2]
                                dy_pixel = transform_matrix[1, 2]

                                self.last_dx_pixel = dx_pixel
                                self.last_dy_pixel = dy_pixel

                                forward_m = (dy_pixel / fy) * z_safe * self.scale_filter.prior_scale
                                right_m = -(dx_pixel / fx) * z_safe * self.scale_filter.prior_scale
                                right_m *= 0.05  

                                raw_global_dx = forward_m * math.cos(self.current_yaw) - right_m * math.sin(self.current_yaw)
                                raw_global_dy = forward_m * math.sin(self.current_yaw) + right_m * math.cos(self.current_yaw)

                                if abs(delta_yaw) > 0.005:
                                    raw_global_dx *= 0.15

                                max_physical_speed = 1.5
                                raw_global_dx = max(min(raw_global_dx, max_physical_speed), -max_physical_speed)
                                raw_global_dy = max(min(raw_global_dy, max_physical_speed), -max_physical_speed)

                                if abs(delta_yaw) < 0.002:
                                    alpha_blend = 0.75
                                    self.smooth_global_dx *= 0.30
                                    self.smooth_global_dy *= 0.30
                                else:
                                    alpha_blend = 0.35

                                self.smooth_global_dx = ((1.0 - alpha_blend) * self.smooth_global_dx) + (alpha_blend * raw_global_dx)
                                self.smooth_global_dy = ((1.0 - alpha_blend) * self.smooth_global_dy) + (alpha_blend * raw_global_dy)
                                
                                self.current_x += self.smooth_global_dx
                                self.current_y += self.smooth_global_dy
                                good_new = good_new[inliers.ravel() == 1] if inliers is not None else good_new

            if not tracking_success:
                self.current_x += self.smooth_global_dx
                self.current_y += self.smooth_global_dy

            self.prev_gray = current_gray.copy()
            self.prev_points = good_new.reshape(-1, 1, 2) if (len(good_new) > 0 and tracking_success) else None

            if self.prev_points is None or len(self.prev_points) < 50:
                self.prev_points = cv2.goodFeaturesToTrack(
                    self.prev_gray, mask=flow_mask, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7
                )

            if self.prev_points is not None:
                for i in self.prev_points:
                    x_pt, y_pt = i.ravel()
                    cv2.circle(frame, (int(x_pt), int(y_pt)), 5, (0, 255, 0), -1) 

            trans_obj = DetectedTranslation(self.current_x, self.current_y, self.current_z)
            prediction.add_translation_object(trans_obj)
            return frame


        else:
            gt_x = float(prediction.gt_translation_x)
            gt_y = float(prediction.gt_translation_y)
            gt_z = float(prediction.gt_translation_z)

            if not hasattr(self, 'prev_gt_x'):
                self.prev_gt_x = gt_x
                self.prev_gt_y = gt_y
                self.prev_gt_z = gt_z

            delta_gt_x = gt_x - self.prev_gt_x
            delta_gt_y = gt_y - self.prev_gt_y
            gercek_uzunluk = math.sqrt(delta_gt_x**2 + delta_gt_y**2)

            if self.prev_gray is not None and self.prev_points is not None:
                
                if w < 1000:
                    prev_blur = cv2.GaussianBlur(self.prev_gray, (3, 3), 0)
                    current_blur = cv2.GaussianBlur(current_gray, (3, 3), 0)
                    
                    next_points, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_blur, current_blur, self.prev_points, None, winSize=(31, 31), maxLevel=3
                    )
                    back_points, back_status, back_error = cv2.calcOpticalFlowPyrLK(
                        current_blur, prev_blur, next_points, None, winSize=(31, 31), maxLevel=3
                    )
                    
                    diff = np.linalg.norm((self.prev_points - back_points).reshape(-1, 2), axis=1)
                    fb_valid = (diff < 1.0) & (status.ravel() == 1) & (back_status.ravel() == 1)
                    
                    good_new = next_points[fb_valid]
                    good_old = self.prev_points[fb_valid]
                else:
                    next_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_points, None)
                    good_new = next_points[status == 1]
                    good_old = self.prev_points[status == 1]

                if len(good_new) >= 4:
                    good_old_centered = good_old - center_array
                    good_new_centered = good_new - center_array
                    total_before = len(good_new)
                    
                    if w < 1000:
                        transform_matrix, inliers = cv2.estimateAffinePartial2D(
                            good_old_centered, good_new_centered, method=cv2.RANSAC, ransacReprojThreshold=2.0
                        )
                        if transform_matrix is not None and inliers is not None:
                            good_new = good_new[inliers.ravel() == 1]
                            good_old = good_old[inliers.ravel() == 1]
                    else:
                        transform_matrix, inliers = cv2.estimateAffinePartial2D(good_old_centered, good_new_centered)
                    
                    inlier_count = int(np.sum(inliers.ravel() == 1)) if (transform_matrix is not None and inliers is not None) else 0
                    inlier_ratio = inlier_count / total_before if total_before > 0 else 0.0
                    min_ratio = 0.55 if w < 1000 else 0.45

                    if transform_matrix is not None and len(good_new) >= 4:
                        if inlier_ratio >= min_ratio:
                            dx_pixel = transform_matrix[0, 2]
                            dy_pixel = transform_matrix[1, 2]
                            
                            z_safe = max(abs(float(gt_z)), 2.0)
                            pinhole_dist = math.sqrt(((dy_pixel/fy)*z_safe)**2 + ((-dx_pixel/fx)*z_safe)**2)

                            if pinhole_dist > 0.001 and gercek_uzunluk > 0.001:
                                anlik_olcek = gercek_uzunluk / pinhole_dist
                                lo, hi = (0.2, 5.0) if w < 1000 else (0.3, 3.0)
                                if lo < anlik_olcek < hi:
                                    self.scale_filter.update(anlik_olcek)
                        else:
                            logging.warning(f"⚠️ Kalibrasyon atlandi: inlier_ratio={inlier_ratio:.2f} < {min_ratio}")
                                
            if gercek_uzunluk > 0.15:
                instant_yaw = math.atan2(delta_gt_y, delta_gt_x)
                self.yaw_history.append(instant_yaw)
                
                if len(self.yaw_history) > 15:
                    self.yaw_history.pop(0)
                    
                sin_sum = sum(math.sin(y) for y in self.yaw_history)
                cos_sum = sum(math.cos(y) for y in self.yaw_history)
                self.current_yaw = math.atan2(sin_sum, cos_sum)

            self.smooth_global_dx = delta_gt_x
            self.smooth_global_dy = delta_gt_y
            self.current_x = gt_x
            self.current_y = gt_y
            self.current_z = gt_z  
            self.prev_gt_x = gt_x
            self.prev_gt_y = gt_y
            self.prev_gt_z = gt_z

        self.prev_gray = current_gray.copy()
        if len(good_new) > 0:
            self.prev_points = good_new.reshape(-1, 1, 2)
        else:
            self.prev_points = None

        if self.prev_points is None or len(self.prev_points) < 50:
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=flow_mask, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7
            )

        if self.prev_points is not None:
            for i in self.prev_points:
                x_pt, y_pt = i.ravel()
                cv2.circle(frame, (int(x_pt), int(y_pt)), 5, (0, 255, 0), -1) 

        # 🎯 TELEMETRİ VE NESNE AKTARIMI DÖNGÜLERİN DIŞINDA
        # 🚀 PERFORMANS OPTİMİZASYONU: Satır arası 'import json' kaldırıldı, tepedeki modül kullanılıyor.
        telemetry_data = {
            "x": float(self.current_x),
            "y": float(self.current_y),
            "z": float(self.current_z),
            "yaw": float(self.current_yaw),
            "timestamp": time.time()
        }
        
        # Asynchronously transmit telemetry packet to Azure cloud gateway if connection is alive
        if self.azure_client is not None:
            try:
                self.azure_client.send_message(json.dumps(telemetry_data))
            except Exception as e:
                logging.error(f"Azure IoT Hub telemetri gönderim hatası kanka: {e}")

        trans_obj = DetectedTranslation(self.current_x, self.current_y, self.current_z)
        prediction.add_translation_object(trans_obj)

        return frame
    
    def validate_output(self, prediction):
        if len(prediction.detected_objects) == 0:
            logging.warning("SİSTEM UYARISI: Model hiçbir nesne tespit edemedi! (Sıfır çıktı)")
            return False
        
        for obj in prediction.detected_objects:
            if obj.top_left_x < 0 or obj.top_left_y < 0:
                logging.error(f"VERİ HATASI: Negatif koordinat bulundu! ID: {obj.cls}")
                return False
                
        return True

    def load_3rd_gorev_from_ram(self, ref_data, server):
        if not ref_data:
            logging.warning("⚠️ Sunucudan boş referans nesne listesi geldi!")
            return
            
        for item in ref_data:
            obj_id = item.get("id") or item.get("obj_id") or item.get("order")
            img_url = item.get("image_url") or item.get("url")
            
            if obj_id is None or not img_url:
                continue
                
            # Patch: Handle missing media directory prefix validation for edge asset URLs
            if img_url.startswith('/') and hasattr(server, 'base_url'):
                base = server.base_url.rstrip('/')
                if not img_url.startswith('/media/'):
                    img_url = base + "/media" + img_url
                else:
                    img_url = base + img_url
            
            try:
                headers = {'Authorization': f'Token {server.auth_token}'}
                img_resp = requests.get(img_url, headers=headers, timeout=10)
                if img_resp.status_code == 200 and img_resp.content:
                    image_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
                    img_matrix = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    img_rgb = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    inputs = self.dino_processor(images=img_pil, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.dino_model(**inputs)
                        feature = outputs.last_hidden_state[:, 0, :] 
                        feature = feature / feature.norm(dim=-1, keepdim=True)
                    
                    ref_url = item.get('url')
                    self.precomputed_3rd_refs[ref_url] = (obj_id, feature)
                    # 🎯 min_hits_to_start=1 yapılarak hızlı akışta ilk eşleşmede kilitlemeyi tetikliyoruz
                    self.goraev_trackers[ref_url] = AeroThermalTracker(alpha=0.45, max_missed_frames=3, min_hits_to_start=1)
                    logging.info(f"✅ RAM'e mühürlendi -> ID: {obj_id}")
                else:
                    logging.error(f"❌ Referans nesne {obj_id} indirilemedi! Sunucu Kodu: {img_resp.status_code} | URL: {img_url}")
            except Exception as e:
                logging.error(f"⚠️ Referans nesne {obj_id} indirilirken ağ hatası oluştu: {e}")

    def process_similarity_and_filter_3rd(self, similarity_map, similarity, orig_w, orig_h):
        max_sim = similarity_map.max().item()
        min_sim = similarity_map.min().item()
        
        confidence = (max_sim - min_sim) / (1.0 - min_sim + 1e-6)
        if confidence < 0.15 or max_sim < 0.23: 
            return None
            
        thresh = max_sim * 0.93
        mask = similarity_map >= thresh
        y_indices, x_indices = torch.where(mask)
        active_patch_count = len(x_indices)
        
        if active_patch_count > 16:  
            _, top_indices = torch.topk(similarity, k=16)
            y_indices = torch.div(top_indices, 32, rounding_mode='floor')
            x_indices = top_indices % 32
            active_patch_count = 16
        elif active_patch_count < 4:  
            return None
        elif active_patch_count < 6:  
            _, top_indices = torch.topk(similarity, k=8)
            y_indices = torch.div(top_indices, 32, rounding_mode='floor')
            x_indices = top_indices % 32
            active_patch_count = 8

        y_indices = y_indices.cpu().numpy()
        x_indices = x_indices.cpu().numpy()

        ymin_patch, ymax_patch = y_indices.min(), y_indices.max()
        xmin_patch, xmax_patch = x_indices.min(), x_indices.max()

        patch_w = xmax_patch - xmin_patch + 1
        patch_h = ymax_patch - ymin_patch + 1
        patch_area = patch_w * patch_h

        density = active_patch_count / patch_area
        if density < 0.40 and max_sim < 0.32:
            return None

        if patch_w > 14 or patch_h > 14:
            return None

        x1 = int(xmin_patch * 14 * (orig_w / 448))
        y1 = int(ymin_patch * 14 * (orig_h / 448))
        x2 = int((xmax_patch + 1) * 14 * (orig_w / 448))
        y2 = int((ymax_patch + 1) * 14 * (orig_h / 448))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        return {
            "top_left_x": x1, "top_left_y": y1,
            "bottom_right_x": x2, "bottom_right_y": y2,
            "confidence": round(confidence, 4)
        }