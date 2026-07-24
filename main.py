import logging
import os
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel

load_dotenv()

USERNAME = os.getenv("USERNAME", "EmreAltindag")
PASSWORD = os.getenv("PASSWORD", "staj_sifren_buraya")
EVALUATION_SERVER_URL = os.getenv("EVALUATION_SERVER_URL", "http://127.0.0.1:5000/")
MIN_FRAME_INTERVAL = 0.25

def configure_logger(username):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + username + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run():
    print("Aero Intelligence Pipeline Engine Started...")
    configure_logger(USERNAME)

    detection_model = ObjectDetectionModel(EVALUATION_SERVER_URL)
    server = ConnectionHandler(EVALUATION_SERVER_URL, username=USERNAME, password=PASSWORD)

    progress = server.get_progress()
    if progress is None or not progress['session_name']:
        print("Could not reach the evaluation server or no active session.")
        return

    session_name = progress['session_name']
    total_frames = progress['total_frames']
    start_index = progress['frame_index']

    server.video_name = session_name + "/"
    server.create_img_folder(server.video_name)
    images_folder = os.path.join(server.img_save_path, server.video_name)

    all_references = server.get_reference_objects(force_download=True) or []
    if all_references:
        try:
            detection_model.load_3rd_gorev_from_ram(all_references, server)
            print(f"[LOG] Loaded {len(all_references)} reference assets directly into RAM.")
        except Exception as e:
            logging.error(f"Critical error allocating reference assets to RAM: {e}")

    with tqdm(total=total_frames, initial=start_index, desc="Processing Frames") as pbar:
        while True:
            frame_start = time.monotonic()
            frame = server.get_current_frame()
            if frame is None:
                break

            translation = server.get_current_translation()
            if translation is None:
                health_status, gt_x, gt_y, gt_z = None, None, None, None
            else:
                health_status = translation['health_status']
                gt_x, gt_y, gt_z = translation['translation_x'], translation['translation_y'], translation['translation_z']

            images_files = os.listdir(images_folder)
            active_refs = [
                r for r in all_references
                if isinstance(r, dict) and r.get('frame_start_image_url') and r.get('frame_end_image_url')
                and r['frame_start_image_url'] <= frame['image_url'] <= r['frame_end_image_url']
            ]

            predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'], gt_x, gt_y, gt_z)
            predictions = detection_model.process(
                predictions, EVALUATION_SERVER_URL, health_status,
                images_folder, images_files, active_refs=active_refs, auth_token=server.auth_token
            )

            server.send_prediction(predictions)
            pbar.update(1)

            # Eş zamanlama aralığı koruması
            elapsed = time.monotonic() - frame_start
            if elapsed < MIN_FRAME_INTERVAL:
                time.sleep(MIN_FRAME_INTERVAL - elapsed)

if __name__ == '__main__':
    run()