"""
STSMIRS — Face Recognizer Engine
Integrates the `face_recognition` library to identify enrolled team members.
Provides real-name labels for TrackedPerson objects instead of generic "T-001".
"""

import os
import json
import sys
import time
import re
from pathlib import Path
import cv2
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if not os.environ.get("YOLO_CONFIG_DIR"):
    os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

# Optional import — degrades gracefully if library not installed
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("[FaceRecognizer] ⚠ 'face_recognition' library not installed. Identity matching disabled.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class FaceRecognizer:
    """Matches tracked persons against enrolled face database periodically."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        self.face_cfg = config.get("face_recognition", {})
        self.enrollment_dir = self.face_cfg.get("enrollment_dir", "project_videos/faces")
        self.model = self.face_cfg.get("model", "hog") # 'hog' (CPU) or 'cnn' (GPU)
        self.tolerance = self.face_cfg.get("tolerance", 0.6)
        self.team_members = self.face_cfg.get("team_members", {})
        self.video_sample_interval_sec = self.face_cfg.get("video_sample_interval_sec", 1.0)
        self.max_video_samples_per_person = self.face_cfg.get("max_video_samples_per_person", 5)
        self.max_image_samples_per_person = int(self.face_cfg.get("max_image_samples_per_person", 24))
        self.opencv_match_threshold = self.face_cfg.get("opencv_match_threshold", 0.55)
        self.check_interval_sec = self.face_cfg.get("check_interval_sec", 0.5)
        self.max_attempts_per_person = self.face_cfg.get("max_attempts_per_person", 100)
        self.opencv_enabled = bool(self.face_cfg.get("opencv_enabled", False) or not FACE_REC_AVAILABLE)
        self.yolo_cls_enabled = self.face_cfg.get("yolo_cls_enabled", True)
        self.yolo_cls_model_path = self.face_cfg.get("yolo_cls_model_path", "models/yolo_cls_faces_best.pt")
        self.yolo_cls_confidence_threshold = self.face_cfg.get("yolo_cls_confidence_threshold", 0.70)
        self.yolo_cls_required_votes = int(self.face_cfg.get("yolo_cls_required_votes", 3))
        self.yolo_cls_model = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # In-memory database
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.known_templates = []
        
        # Cache to avoid re-running recognition every frame
        # track_id -> {"name": str, "last_check": float, "attempts": int}
        self._identity_cache = {}
        
        self._load_database()
        self._load_yolo_classifier()

    def _load_yolo_classifier(self):
        """Load the optional YOLO classifier used to label tracked person crops."""
        if not self.yolo_cls_enabled:
            return
        if not YOLO_AVAILABLE:
            print("[FaceRecognizer] YOLO identity classifier unavailable: ultralytics not installed.")
            return
        if not os.path.exists(self.yolo_cls_model_path):
            print(f"[FaceRecognizer] YOLO identity model not found: {self.yolo_cls_model_path}")
            return

        try:
            self.yolo_cls_model = YOLO(self.yolo_cls_model_path)
            print(f"[FaceRecognizer] Loaded YOLO identity classifier: {self.yolo_cls_model_path}")
        except Exception as e:
            print(f"[FaceRecognizer] Could not load YOLO identity classifier: {e}")

    def _load_database(self):
        """Pre-compute face encodings from enrollment images."""
        print(f"[FaceRecognizer] Loading enrolled faces from {self.enrollment_dir}...")
        
        if not os.path.exists(self.enrollment_dir):
            print(f"[FaceRecognizer] ⚠ Enrollment directory not found.")
            return
            
        # Iterate through team members config to match with folders
        for key, info in self.team_members.items():
            member_name = info["name"]
            member_id = info["id"]
            
            # Directory should be named something like person1, person2, etc. (matching key)
            # Or by name. Let's look for matching files or subfolders.
            member_dir = os.path.join(self.enrollment_dir, key)
            
            if os.path.isdir(member_dir):
                count = 0
                for img_name in os.listdir(member_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(member_dir, img_name)
                        
                        try:
                            # Load and encode
                            image = face_recognition.load_image_file(img_path)
                            boxes = face_recognition.face_locations(image, model=self.model)
                            
                            if len(boxes) > 0:
                                encoding = face_recognition.face_encodings(image, boxes)[0]
                                self.known_encodings.append(encoding)
                                self.known_names.append(member_name)
                                self.known_ids.append(member_id)
                                count += 1
                        except Exception as e:
                            print(f"[FaceRecognizer] Error processing {img_path}: {e}")
                            
                print(f"[FaceRecognizer]   ✓ Loaded {count} encodings for {member_name}")

        if FACE_REC_AVAILABLE:
            total = len(self.known_encodings)
            backend = "face_recognition encodings"
        else:
            total = len(self.known_templates)
            backend = "OpenCV templates"
        print(f"[FaceRecognizer] Total {backend} in database: {total}")

    def _add_encoding_from_rgb(self, image_rgb, member_name, member_id, source_label):
        """Detect and store the first face encoding from one RGB image."""
        if not FACE_REC_AVAILABLE:
            template = self._opencv_template_from_rgb(image_rgb)
            if template is None:
                return False

            self.known_templates.append(template)
            self.known_names.append(member_name)
            self.known_ids.append(member_id)
            return True

        try:
            boxes = face_recognition.face_locations(image_rgb, model=self.model)
            if len(boxes) == 0:
                return False

            encoding = face_recognition.face_encodings(image_rgb, boxes)[0]
            self.known_encodings.append(encoding)
            self.known_names.append(member_name)
            self.known_ids.append(member_id)
            return True
        except Exception as e:
            print(f"[FaceRecognizer] Error processing {source_label}: {e}")
            return False

    def _load_image_file(self, img_path, member_name, member_id):
        if FACE_REC_AVAILABLE:
            image = face_recognition.load_image_file(img_path)
        else:
            bgr = cv2.imread(img_path)
            if bgr is None:
                return False
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._add_encoding_from_rgb(image, member_name, member_id, img_path)

    def _opencv_template_from_rgb(self, image_rgb):
        """Return a normalized grayscale face template using OpenCV Haar detection."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )
        if len(faces) == 0:
            h, w = gray.shape[:2]
            if h < 64 or w < 64:
                return None
            crop_size = int(min(h, w) * 0.65)
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            face = gray[y1:y2, x1:x2]
            if face.size == 0:
                return None
            face = cv2.resize(face, (96, 96))
            face = cv2.equalizeHist(face)
            return face

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (96, 96))
        face = cv2.equalizeHist(face)
        return face

    def _opencv_best_match(self, image_rgb):
        template = self._opencv_template_from_rgb(image_rgb)
        if template is None or len(self.known_templates) == 0:
            return None, 0.0

        best_idx = None
        best_score = -1.0
        for idx, known in enumerate(self.known_templates):
            score = cv2.matchTemplate(template, known, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_idx = idx
                best_score = float(score)

        if best_idx is None or best_score < self.opencv_match_threshold:
            return None, best_score

        return self.known_names[best_idx], best_score

    def _load_video_file(self, video_path, member_name, member_id):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FaceRecognizer] Warning: could not open enrollment video: {video_path}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_step = max(1, int(fps * self.video_sample_interval_sec))

        count = 0
        frame_idx = 0
        while count < self.max_video_samples_per_person:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self._add_encoding_from_rgb(rgb, member_name, member_id, f"{video_path} frame {frame_idx}"):
                    count += 1

            frame_idx += 1

        cap.release()
        return count

    def _member_from_filename(self, file_path):
        stem = Path(file_path).stem.strip()
        clean_name = stem
        clean_name = re.sub(r"_frame_\d+$", "", clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r"\s+frame\s+\d+$", "", clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r"[_\-\s]+frames?$", "", clean_name, flags=re.IGNORECASE)
        for suffix in (" face", "_face", "-face"):
            if clean_name.lower().endswith(suffix):
                clean_name = clean_name[: -len(suffix)].strip()

        display_name = clean_name.replace("_", " ").replace("-", " ").title()
        stable_id = f"T-{len(set(self.known_ids)) + 1:03d}"
        return display_name or stem, stable_id

    @staticmethod
    def _is_usable_member_name(member_name):
        clean = (member_name or "").strip()
        return bool(clean) and not clean.isdigit()

    def _iter_root_samples_grouped(self, image_exts, video_exts):
        """Group loose root-level samples so frame dumps become one person, not many."""
        grouped = {}
        for sample_name in os.listdir(self.enrollment_dir):
            sample_path = os.path.join(self.enrollment_dir, sample_name)
            if not os.path.isfile(sample_path):
                continue

            lower_name = sample_name.lower()
            if not (lower_name.endswith(image_exts) or lower_name.endswith(video_exts)):
                continue

            member_name, member_id = self._member_from_filename(sample_path)
            if not self._is_usable_member_name(member_name):
                continue
            key = (member_name, member_id)
            grouped.setdefault(key, []).append(sample_path)

        return grouped

    def _iter_named_member_dirs(self):
        """Yield enrollment folders, including ad hoc folders named after a person."""
        used_paths = set()

        for key, info in self.team_members.items():
            member_dir = os.path.join(self.enrollment_dir, key)
            if not os.path.isdir(member_dir):
                continue
            used_paths.add(os.path.normcase(os.path.abspath(member_dir)))
            yield member_dir, info["name"], info["id"]

        for entry in sorted(os.listdir(self.enrollment_dir)):
            member_dir = os.path.join(self.enrollment_dir, entry)
            if not os.path.isdir(member_dir):
                continue
            norm_path = os.path.normcase(os.path.abspath(member_dir))
            if norm_path in used_paths:
                continue

            member_name, member_id = self._member_from_filename(member_dir)
            if not self._is_usable_member_name(member_name):
                continue
            yield member_dir, member_name, member_id

    def _sample_image_paths(self, paths):
        """Evenly sample image paths so face startup stays fast and balanced."""
        if self.max_image_samples_per_person <= 0 or len(paths) <= self.max_image_samples_per_person:
            return list(paths)

        indices = np.linspace(0, len(paths) - 1, num=self.max_image_samples_per_person, dtype=int)
        return [paths[idx] for idx in indices]

    def _load_database(self):
        """Pre-compute face encodings from enrollment images and videos."""
        print(f"[FaceRecognizer] Loading enrolled faces from {self.enrollment_dir}...")

        if not os.path.exists(self.enrollment_dir):
            print("[FaceRecognizer] Enrollment directory not found.")
            return

        image_exts = ('.jpg', '.jpeg', '.png')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')

        # Supports both configured member folders and ad hoc folders named after a person.
        for member_dir, member_name, member_id in self._iter_named_member_dirs():
            count = 0
            sample_names = sorted(os.listdir(member_dir))
            image_names = [name for name in sample_names if name.lower().endswith(image_exts)]
            video_names = [name for name in sample_names if name.lower().endswith(video_exts)]

            image_names = [os.path.basename(path) for path in self._sample_image_paths(
                [os.path.join(member_dir, name) for name in image_names]
            )]

            for sample_name in image_names + video_names:
                sample_path = os.path.join(member_dir, sample_name)
                if not os.path.isfile(sample_path):
                    continue

                lower_name = sample_name.lower()
                if lower_name.endswith(image_exts):
                    if self._load_image_file(sample_path, member_name, member_id):
                        count += 1
                elif lower_name.endswith(video_exts):
                    count += self._load_video_file(sample_path, member_name, member_id)

            print(f"[FaceRecognizer]   Loaded {count} encodings for {member_name}")

        # Supports loose files such as gjk face.mp4 and jeevan_frame_00001.jpg.
        for (member_name, member_id), sample_paths in self._iter_root_samples_grouped(image_exts, video_exts).items():
            count = 0
            preview_names = []
            image_paths = sorted([path for path in sample_paths if path.lower().endswith(image_exts)])
            video_paths = sorted([path for path in sample_paths if path.lower().endswith(video_exts)])
            ordered_paths = self._sample_image_paths(image_paths) + video_paths
            for sample_path in ordered_paths:
                lower_name = sample_path.lower()
                preview_names.append(os.path.basename(sample_path))
                if lower_name.endswith(image_exts):
                    if self._load_image_file(sample_path, member_name, member_id):
                        count += 1
                elif lower_name.endswith(video_exts):
                    count += self._load_video_file(sample_path, member_name, member_id)

            sample_preview = preview_names[0] if len(preview_names) == 1 else f"{preview_names[0]} +{len(preview_names)-1}"
            print(f"[FaceRecognizer]   Loaded {count} encodings for {member_name} from {sample_preview}")

        if FACE_REC_AVAILABLE:
            total = len(self.known_encodings)
            backend = "face_recognition encodings"
        else:
            total = len(self.known_templates)
            backend = "OpenCV templates"
        print(f"[FaceRecognizer] Total {backend} in database: {total}")

    def identify_persons(self, frame, persons):
        """
        Check track crops against face database.
        
        Args:
            frame: Full BGR image (OpenCV format)
            persons: list of TrackedPerson objects (modified in place)
        """
        if FACE_REC_AVAILABLE:
            has_database = len(self.known_encodings) > 0
        else:
            has_database = self.opencv_enabled and len(self.known_templates) > 0

        has_yolo_classifier = self.yolo_cls_model is not None
        if not has_database and not has_yolo_classifier:
            return
            
        now = time.time()
        rgb_frame = None  # Lazy conversion to RGB (only if needed)
        
        for person in persons:
            tid = person.track_id
            
            # 1. Check cache first
            if tid not in self._identity_cache:
                self._identity_cache[tid] = {
                    "name": None,
                    "confidence": 0.0,
                    "source": None,
                    "candidate": None,
                    "candidate_votes": 0,
                    "last_check": 0.0,
                    "attempts": 0
                }
                
            cache = self._identity_cache[tid]
            
            # If already identified, just update the label and continue
            if cache["name"] is not None:
                person.track_label = cache["name"]
                person.identity_confidence = cache.get("confidence", 0.0)
                person.identity_source = cache.get("source")
                continue
                
            # If failed too many times, skip (save CPU)
            if cache["attempts"] >= self.max_attempts_per_person:
                continue
                
            # 2. Need physical check. Throttle rate.
            if now - cache["last_check"] < self.check_interval_sec:
                continue
                
            # Time to check!
            cache["last_check"] = now
            cache["attempts"] += 1
            
            # Crop bounding box out of frame (add slight padding)
            x1, y1, x2, y2 = person.bbox
            
            # Ensure bounds are within frame
            h, w = frame.shape[:2]
            pad = 35
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(w, x2 + pad)
            if not FACE_REC_AVAILABLE:
                face_region_bottom = y1 + int((y2 - y1) * 0.80)
                y2_p = min(h, face_region_bottom + pad)
            else:
                y2_p = min(h, y2 + pad)
            
            # Need a reasonably sized crop to even see a face
            if (x2_p - x1_p) < 40 or (y2_p - y1_p) < 40:
                continue
                
            crop = frame[y1_p:y2_p, x1_p:x2_p]
            
            # Lazy RBG conversion of crop (face_recognition uses RGB)
            crop_rgb = crop[:, :, ::-1] 
            
            if FACE_REC_AVAILABLE:
                # Detect face in the crop
                face_locations = face_recognition.face_locations(crop_rgb, model=self.model)
                
                if len(face_locations) == 0:
                    continue  # No face visible
                    
                # Get encoding (assume first face is the person)
                face_encodings = face_recognition.face_encodings(crop_rgb, face_locations)
                
                if len(face_encodings) == 0:
                    continue
                    
                encoding_to_check = face_encodings[0]
                
                # Compare to database
                matches = face_recognition.compare_faces(self.known_encodings, encoding_to_check, tolerance=self.tolerance)
                
                if True in matches:
                    # Find best match based on distance
                    face_distances = face_recognition.face_distance(self.known_encodings, encoding_to_check)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        identified_name = self.known_names[best_match_index]
                        
                        # Store in cache
                        cache["name"] = identified_name
                        cache["confidence"] = 1.0 - float(face_distances[best_match_index])
                        cache["source"] = "face_recognition"
                        
                        # Update person object immediately
                        person.track_label = identified_name
                        person.identity_confidence = cache["confidence"]
                        person.identity_source = cache["source"]
                        print(f"[FaceRecognizer] Match Found! Track {tid} -> {identified_name}")
            elif self.opencv_enabled:
                identified_name, score = self._opencv_best_match(crop_rgb)
                if identified_name:
                    cache["name"] = identified_name
                    cache["confidence"] = score
                    cache["source"] = "opencv"
                    person.track_label = identified_name
                    person.identity_confidence = score
                    person.identity_source = "opencv"
                    print(f"[FaceRecognizer] OpenCV match! Track {tid} -> {identified_name} ({score:.2f})")

            if cache["name"] is None and has_yolo_classifier:
                identified_name, score = self._classify_person_crop(crop)
                if identified_name:
                    if cache.get("candidate") == identified_name:
                        cache["candidate_votes"] = cache.get("candidate_votes", 0) + 1
                    else:
                        cache["candidate"] = identified_name
                        cache["candidate_votes"] = 1

                    if cache["candidate_votes"] >= self.yolo_cls_required_votes:
                        cache["name"] = identified_name
                        cache["confidence"] = score
                        cache["source"] = "yolo_cls"
                        person.track_label = identified_name
                        person.identity_confidence = score
                        person.identity_source = "yolo_cls"
                        print(
                            f"[FaceRecognizer] YOLO identity confirmed! "
                            f"Track {tid} -> {identified_name} ({score:.2f})"
                        )

    def _classify_person_crop(self, crop):
        """Classify a tracked crop with the trained YOLO image classifier."""
        if self.yolo_cls_model is None or crop is None or crop.size == 0:
            return None, 0.0

        try:
            results = self.yolo_cls_model.predict(
                source=crop,
                imgsz=224,
                device="cpu",
                verbose=False,
            )
        except Exception as e:
            print(f"[FaceRecognizer] YOLO identity inference failed: {e}")
            return None, 0.0

        if not results or results[0].probs is None:
            return None, 0.0

        probs = results[0].probs
        class_idx = int(probs.top1)
        confidence = float(probs.top1conf)
        if confidence < self.yolo_cls_confidence_threshold:
            return None, confidence

        name = results[0].names.get(class_idx, str(class_idx))
        return name.replace("_", " ").title(), confidence

    def reset_cache(self):
        """Clear identity cache (useful if tracking IDs get recycled incorrectly)."""
        self._identity_cache.clear()


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/face_recognizer.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=" * 50)
    print("  STSMIRS — Face Recognizer Engine Test")
    print("=" * 50)
    
    fr = FaceRecognizer("config.json")
    print(f"Is library available? {FACE_REC_AVAILABLE}")
    
    class MockPerson:
        def __init__(self, id, bbox):
            self.track_id = id
            self.track_label = f"T-{id:03d}"
            self.bbox = bbox
            
    # Mock frame and person (won't actually match unless realistic faces are provided)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    p1 = MockPerson(1, [100, 100, 200, 200])
    
    print("\nRunning test identification (blank frame)...")
    fr.identify_persons(blank, [p1])
    print(f"Resulting label: {p1.track_label}")
    
    print("\n[Done] Face Recognizer test complete.")
