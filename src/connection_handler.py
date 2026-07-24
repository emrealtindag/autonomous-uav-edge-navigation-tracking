import json
import logging
import requests
import time
import os
import cv2
import numpy as np

# 🚀 UNUSED DECOUPLE IMPORT CLEANED
class ConnectionHandler:
    """
    Manages robust REST API communications, authentication states, 
    and session progress caching with the edge evaluation backend.
    """
    def __init__(self, base_url, username=None, password=None):
        self.base_url = base_url
        self.auth_token = None
        self.classes = None
        self.video_name = ''
        self.img_save_path = './_images/'

        self.url_login = self.base_url + "auth/"
        self.url_frames = self.base_url + "frames/"
        self.url_translations = self.base_url + "translation/"
        self.url_prediction = self.base_url + "prediction/"
        self.url_session = self.base_url + "session/"
        self.url_reference = self.base_url + "reference/"
        self.url_progress = self.base_url + "progress/"

        if username and password:
            self.login(username, password)

    def login(self, username, password):
        payload = {'username': username, 'password': password}
        files = []
        try:
            response = requests.post(self.url_login, data=payload, files=files, timeout=10)
            response_json = json.loads(response.text)
            if response.status_code == 200:
                self.auth_token = response_json['token']
                logging.info("Network Auth Token Generated Successfully.")
            else:
                logging.error("Authentication Request Rejected: {}".format(response.text))
        except requests.exceptions.RequestException as e:
            logging.error(f"Login pipeline critical failure: {e}")

    def create_img_folder(self, path):
        post_path = os.path.join(self.img_save_path, path)
        os.makedirs(post_path, exist_ok=True)

    def get_listdir(self):
        save_path = os.path.join(self.img_save_path, self.video_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        return os.listdir(save_path), save_path

    def get_progress(self, retries=3, initial_wait_time=0.5):
        headers = {'Authorization': 'Token {}'.format(self.auth_token)}
        wait_time = initial_wait_time

        for attempt in range(retries):
            try:
                response = requests.get(self.url_progress, headers=headers, timeout=30)
                if response.status_code == 200:
                    progress = json.loads(response.text)
                    logging.info("Stream Synchronization: frame {frame_index}/{total_frames} "
                                 "(session={session_name})".format(**progress))
                    return progress
                else:
                    logging.error("Failed to fetch stream progress: {}".format(response.text))
            except requests.exceptions.RequestException as e:
                logging.error(f"Progress synchronization request error: {e}")

            logging.info(f"Retrying connection in {wait_time}s...")
            time.sleep(wait_time)
            wait_time *= 2

        return None

    def get_current_frame(self, retries=5, initial_wait_time=0.1):
        headers = {'Authorization': 'Token {}'.format(self.auth_token)}
        wait_time = initial_wait_time

        for attempt in range(retries):
            try:
                response = requests.get(self.url_frames, headers=headers, timeout=60)
                if response.status_code == 200:
                    frames = json.loads(response.text)
                    if not frames:
                        logging.info("Session sequence completely processed.")
                        return None
                    frame = frames[0]
                    logging.info("Active processing image target: {}".format(frame.get('image_url')))
                    
                    if frame.get('video_name') and not self.video_name:
                        self.video_name = frame['video_name'] + '/'
                    return frame
                else:
                    logging.error("Frame acquisition failed: {}".format(response.text))
            except requests.exceptions.RequestException as e:
                logging.error(f"Frame request endpoint network error: {e}")

            time.sleep(wait_time)
            wait_time *= 2

        return None

    def get_current_translation(self, retries=5, initial_wait_time=0.1):
        headers = {'Authorization': 'Token {}'.format(self.auth_token)}
        wait_time = initial_wait_time

        for attempt in range(retries):
            try:
                response = requests.get(self.url_translations, headers=headers, timeout=60)
                if response.status_code == 200:
                    translations = json.loads(response.text)
                    if not translations:
                        return None
                    translation = translations[0]
                    return translation
                else:
                    logging.error("Translation data fetch failed: {}".format(response.text))
            except requests.exceptions.RequestException as e:
                logging.error(f"Translation request endpoint network error: {e}")

            time.sleep(wait_time)
            wait_time *= 2

        return None

    def send_prediction(self, prediction, retries=5, initial_wait_time=0.1):
        payload = json.dumps(prediction.create_payload(self.base_url))
        files = []
        headers = {
            'Authorization': 'Token {}'.format(self.auth_token),
            'Content-Type': 'application/json',
        }
        wait_time = initial_wait_time

        for attempt in range(retries):
            try:
                response = requests.post(self.url_prediction, headers=headers, data=payload, files=files, timeout=60)
                if response.status_code == 201:
                    logging.info("Inference metadata payload uploaded successfully.")
                    return response
                elif response.status_code == 406:
                    logging.error("Payload rejected - Double submission caught.")
                    return response
                else:
                    logging.error("Payload submission error: {}".format(response.text))
                    try:
                        detail = json.loads(response.text).get("detail", "")
                    except ValueError:
                        detail = ""
                    if (response.status_code == 403 or "exceeded" in detail.lower()):
                        logging.warning("API gateway rate limits triggered.")
                        return response
            except requests.exceptions.RequestException as e:
                logging.error(f"Prediction transmission failed: {e}")

            time.sleep(wait_time)
            wait_time *= 2

        return None

    def save_references_to_file(self, references):
        try:
            if not self.video_name:
                return
            refs_path = os.path.join(self.img_save_path, self.video_name, "references.json")
            os.makedirs(os.path.dirname(refs_path), exist_ok=True)
            with open(refs_path, 'w') as f:
                json.dump(references, f)
            logging.info(f"Reference assets profile successfully cached locally at {refs_path}")
        except Exception as e:
            logging.warning(f"Local storage cache failure: {e}")
            
    def load_references_from_file(self, session_name):
        base_path = os.path.join(self.img_save_path, session_name, "references.json")
        dirs = os.listdir(self.img_save_path) if os.path.exists(self.img_save_path) else []
        if session_name in dirs and os.path.exists(base_path):
            with open(base_path, 'r') as f:
                refs = json.load(f)
            return refs
        return None

    def get_reference_objects(self, force_download=False, retries=5, initial_wait_time=0.1):
        headers = {'Authorization': 'Token {}'.format(self.auth_token)}
        wait_time = initial_wait_time

        for attempt in range(retries):
            try:
                logging.info(f"Requesting evaluation reference items profile (Attempt {attempt + 1}/{retries})...")
                response = requests.get(self.url_reference, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    ref_data = json.loads(response.text)
                    return ref_data
                else:
                    logging.error("Failed to load reference metadata: {}".format(response.text))
            except requests.exceptions.RequestException as e:
                logging.error(f"Reference profiles download pipeline error: {e}")

            time.sleep(wait_time)
            wait_time *= 2

        return []