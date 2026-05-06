"""
STSMIRS — Face Recognizer Engine (PyTorch Edition)
Integrates the `facenet-pytorch` library to identify enrolled team members.
Provides real-name labels for TrackedPerson objects instead of generic "T-001".
"""

import os
import json
import sys
import time
import pickle
import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognizer:
    """Matches tracked persons against enrolled face database periodically."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        self.face_cfg = config.get("face_recognition", {})
        
        # Check standard STSMIRS folders
        self.enrollment_dir = "faces"
        if not os.path.exists(self.enrollment_dir):
            if os.path.exists("project_videos/faces"):
                self.enrollment_dir = "project_videos/faces"
            elif os.path.exists("../../project_videos/faces"):
                self.enrollment_dir = "../../project_videos/faces"
                
        self.check_interval_sec = self.face_cfg.get("check_interval_sec", 0.5)
        self.max_attempts_per_person = self.face_cfg.get("max_attempts_per_person", 100)
        self.tolerance = self.face_cfg.get("tolerance", 0.8) # Standard tolerance
        
        # Load AI
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[FaceRecognizer] Loading Facenet-PyTorch on {self.device.upper()}...")
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Database
        self.known_encodings = []
        self.known_names = []
        
        # Cache to avoid re-running recognition every frame
        self._identity_cache = {}
        
        self._load_database()

    def _load_database(self):
        """Pre-compute face encodings from enrollment images using Cache."""
        print(f"[FaceRecognizer] Loading enrolled faces from {self.enrollment_dir}...")
        
        if not os.path.exists(self.enrollment_dir):
            print(f"[FaceRecognizer] ⚠ Enrollment directory not found: {self.enrollment_dir}")
            return
            
        cache_path = os.path.join(self.enrollment_dir, "faces_cache.pkl")
        if os.path.exists(cache_path):
            print("[FaceRecognizer] Loading faces from cache (Instant!)...")
            with open(cache_path, "rb") as f:
                self.known_encodings, self.known_names = pickle.load(f)
            if isinstance(self.known_encodings, list):
                self.known_encodings = np.array(self.known_encodings)
            print(f"[FaceRecognizer] Total encodings loaded: {len(self.known_names)}")
            return
            
        print(f"[FaceRecognizer] Processing images (This may take a few minutes)...")
        for person_name in os.listdir(self.enrollment_dir):
            person_path = os.path.join(self.enrollment_dir, person_name)
            if not os.path.isdir(person_path): continue

            for file in os.listdir(person_path):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")): continue

                img_path = os.path.join(person_path, file)
                try:
                    img = cv2.imread(img_path)
                    if img is None: continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = self.mtcnn(img_rgb)
                    if faces is not None:
                        emb = self.resnet(faces.unsqueeze(0).to(self.device)).detach().cpu().numpy()
                        self.known_encodings.append(emb[0])
                        self.known_names.append(person_name)
                except Exception:
                    pass
                    
        # Save cache
        with open(cache_path, "wb") as f:
            pickle.dump((self.known_encodings, self.known_names), f)

        if self.known_encodings:
            self.known_encodings = np.array(self.known_encodings)
        else:
            self.known_encodings = np.zeros((0, 512))

        print(f"[FaceRecognizer] Total encodings generated: {len(self.known_names)}")

    def identify_persons(self, frame, persons):
        """Check track crops against face database."""
        if len(self.known_encodings) == 0:
            return
            
        now = time.time()
        for person in persons:
            tid = person.track_id
            if tid not in self._identity_cache:
                self._identity_cache[tid] = {"name": None, "last_check": 0.0, "attempts": 0}
                
            cache = self._identity_cache[tid]
            if cache["name"] is not None:
                person.track_label = cache["name"]
                continue
                
            if cache["attempts"] >= self.max_attempts_per_person:
                continue
                
            if now - cache["last_check"] < self.check_interval_sec:
                continue
                
            cache["last_check"] = now
            cache["attempts"] += 1
            
            x1, y1, x2, y2 = person.bbox
            h, w = frame.shape[:2]
            pad = 40
            x1p, y1p, x2p, y2p = max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)
            
            if (x2p - x1p) < 30 or (y2p - y1p) < 30:
                continue
                
            try:
                crop = frame[y1p:y2p, x1p:x2p]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                faces = self.mtcnn(crop_rgb)
                if faces is not None:
                    emb = self.resnet(faces.unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]
                    dists = np.linalg.norm(self.known_encodings - emb, axis=1)
                    best_idx = np.argmin(dists)
                    
                    if dists[best_idx] < self.tolerance:
                        name = self.known_names[best_idx]
                        cache["name"] = name
                        person.track_label = name
                        print(f"🔥 [Face AI] IDENTIFIED: {name}!")
            except Exception:
                pass

    def apply_cached_names(self, persons):
        for person in persons:
            cache = self._identity_cache.get(person.track_id)
            if cache and cache["name"]:
                person.track_label = cache["name"]

    def reset_cache(self):
        self._identity_cache.clear()
