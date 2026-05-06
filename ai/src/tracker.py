"""
STSMIRS — ByteTrack Person Tracker
Assigns stable tracking IDs across frames using ultralytics ByteTrack.
Combines detection + tracking in one pass for efficiency.

Usage (standalone test):
    python src/tracker.py
    python src/tracker.py --source 0   (laptop webcam)
"""

import cv2
import json
import os
import time
import numpy as np

if not os.environ.get("YOLO_CONFIG_DIR"):
    os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

from ultralytics import YOLO


class TrackedPerson:
    """Represents a tracked person in a single frame."""

    def __init__(self, track_id, bbox, confidence, class_id=0):
        self.track_id = int(track_id)
        self.track_label = f"T-{self.track_id:03d}"
        self.bbox = [int(b) for b in bbox]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)

        x1, y1, x2, y2 = self.bbox
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self.foot_point = (int((x1 + x2) / 2), int(y2))
        self.width = x2 - x1
        self.height = y2 - y1
        self.area = self.width * self.height

        # These get set by other modules later
        self.zone_id = None
        self.event_type = None
        self.action_confidence = 0.0
        self.action_probabilities = {}
        self.health_score = 100
        self.crime_score = 0
        self.is_alert = False

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "track_id": self.track_id,
            "track_label": self.track_label,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "center": self.center,
            "foot_point": self.foot_point,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "zone_id": self.zone_id,
            "event_type": self.event_type,
            "action_confidence": self.action_confidence,
            "action_probabilities": self.action_probabilities,
            "health_score": self.health_score,
            "crime_score": self.crime_score,
            "is_alert": self.is_alert,
        }


class PersonTracker:
    """YOLOv8 + ByteTrack person tracker."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        det_cfg = config["detector"]
        trk_cfg = config["tracker"]

        self.confidence_threshold = det_cfg.get("confidence_threshold", 0.5)
        self.iou_threshold = det_cfg.get("iou_threshold", 0.45)
        self.classes = det_cfg.get("classes", [0])
        self.imgsz = det_cfg.get("imgsz", 640)

        # ByteTrack parameters
        self.tracker_type = trk_cfg.get("tracker_type", "bytetrack")
        self.track_high_thresh = trk_cfg.get("track_high_thresh", 0.5)
        self.track_low_thresh = trk_cfg.get("track_low_thresh", 0.1)
        self.new_track_thresh = trk_cfg.get("new_track_thresh", 0.6)
        self.track_buffer = trk_cfg.get("track_buffer", 30)
        self.match_thresh = trk_cfg.get("match_thresh", 0.8)

        # Device selection
        device = det_cfg.get("device", "auto")
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load YOLO model
        model_name = det_cfg.get("model", "yolov8n.pt")
        print(f"[Tracker] Loading model: {model_name} on {self.device}")
        self.model = YOLO(model_name)
        print(f"[Tracker] ✓ Model loaded. Tracker: {self.tracker_type}")

        # Track history for velocity/trajectory (needed later for LSTM features)
        self._track_history = {}  # track_id -> list of (cx, cy, timestamp)
        self._max_history = 60    # Keep last 60 positions

    def update(self, frame):
        """
        Run detection + tracking on a frame.

        Args:
            frame: BGR numpy array (OpenCV format)

        Returns:
            list of TrackedPerson objects for this frame
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            tracker=f"{self.tracker_type}.yaml",
            persist=True,       # Keep tracking state across frames
            verbose=False
        )

        tracked_persons = []
        now = time.time()

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    person = TrackedPerson(
                        track_id=track_ids[i],
                        bbox=boxes[i],
                        confidence=confs[i]
                    )
                    tracked_persons.append(person)

                    # Update track history
                    tid = person.track_id
                    if tid not in self._track_history:
                        self._track_history[tid] = []
                    self._track_history[tid].append(
                        (person.center[0], person.center[1], now)
                    )
                    # Trim history
                    if len(self._track_history[tid]) > self._max_history:
                        self._track_history[tid] = \
                            self._track_history[tid][-self._max_history:]

        return tracked_persons

    def get_velocity(self, track_id):
        """
        Get velocity (dx, dy) for a tracked person.
        Returns (0, 0) if insufficient history.
        """
        history = self._track_history.get(track_id, [])
        if len(history) < 2:
            return (0.0, 0.0)

        # Use last 5 frames for smoothed velocity
        recent = history[-5:]
        x1, y1, t1 = recent[0]
        x2, y2, t2 = recent[-1]
        dt = t2 - t1
        if dt <= 0:
            return (0.0, 0.0)
        return ((x2 - x1) / dt, (y2 - y1) / dt)

    def get_trajectory(self, track_id, n_points=30):
        """Get last N trajectory points for a tracked person."""
        history = self._track_history.get(track_id, [])
        return [(x, y) for x, y, _ in history[-n_points:]]

    def reset(self):
        """Reset tracking state."""
        self._track_history.clear()
        # Re-initialize model to reset tracker state
        self.model = YOLO(self.model.model_name if hasattr(self.model, 'model_name')
                          else "yolov8n.pt")


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/tracker.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="STSMIRS Person Tracker Test")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None,
                        help="Video source: URL, file path, or '0' for webcam")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        exit(1)

    # Initialize tracker
    tracker = PersonTracker(args.config)

    # Determine video source
    if args.source:
        source = 0 if args.source == "0" else args.source
    else:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        source = cfg["stream"]["url"]

    print("=" * 50)
    print("  STSMIRS — Person Tracker Test")
    print(f"  Source: {source}")
    print("  Press 'q' to quit")
    print("=" * 50)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARN] Could not open {source}, trying fallback webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No video source available.")
            exit(1)

    fps_time = time.time()
    fps_count = 0
    current_fps = 0.0

    # Color palette for different track IDs
    COLORS = [
        (0, 255, 0), (255, 127, 0), (0, 127, 255), (255, 0, 127),
        (127, 255, 0), (0, 255, 255), (255, 0, 255), (127, 0, 255),
        (255, 255, 0), (0, 255, 127), (255, 127, 127), (127, 255, 255),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.5)
            continue

        # Track
        persons = tracker.update(frame)
        annotated = frame.copy()

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            color = COLORS[person.track_id % len(COLORS)]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw tracking ID label
            label = f"{person.track_label} ({person.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw foot point
            cv2.circle(annotated, person.foot_point, 5, (0, 0, 255), -1)

            # Draw trajectory trail
            trajectory = tracker.get_trajectory(person.track_id, n_points=20)
            for j in range(1, len(trajectory)):
                thickness = int(1 + (j / len(trajectory)) * 3)
                cv2.line(annotated, trajectory[j - 1], trajectory[j],
                         color, thickness)

        # FPS counter
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()

        # HUD
        cv2.putText(annotated,
                     f"FPS: {current_fps:.1f} | Tracked: {len(persons)}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, "STSMIRS Tracker Test | 'q' to quit",
                     (10, annotated.shape[0] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("STSMIRS Tracker Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Tracker test complete.")
