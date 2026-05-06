"""
YOLO image classifier for live fighting confirmation.

The trained model has two internal classes: fighting and walking. We hide the
walking label from the UI and expose it as "Normal" so the live system remains
focused on public-safety actions.
"""

import json
import os
import math

if not os.environ.get("YOLO_CONFIG_DIR"):
    os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ActionFrameClassifier:
    """Classifies the current frame as Fighting or Normal."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        cfg = config.get("action_frame_classifier", {})
        self.enabled = cfg.get("enabled", False)
        self.model_path = cfg.get("model_path", "models/action_fighting_vs_walking_best.pt")
        self.positive_class = cfg.get("positive_class", "fighting").lower()
        self.positive_display_label = cfg.get("positive_display_label", "Fighting")
        self.negative_display_label = cfg.get("negative_display_label", "Normal")
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.85))
        self.run_every_n_frames = max(1, int(cfg.get("run_every_n_frames", 3)))
        self.min_persons_for_fighting = max(1, int(cfg.get("min_persons_for_fighting", 2)))

        self.model = None
        self._frame_counter = 0
        self._last_result = self._empty_result()

        if self.enabled:
            self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("[ActionFrameClassifier] Ultralytics not installed; disabled.")
            self.enabled = False
            return
        if not os.path.exists(self.model_path):
            print(f"[ActionFrameClassifier] Model not found: {self.model_path}")
            self.enabled = False
            return

        self.model = YOLO(self.model_path)
        print(f"[ActionFrameClassifier] Loaded model: {self.model_path}")

    def _empty_result(self):
        return {
            "label": self.negative_display_label,
            "confidence": 0.0,
            "raw_label": None,
            "track_ids": [],
        }

    def update(self, frame, persons):
        """
        Run periodic frame classification.

        Returns a dict with label/confidence/raw_label. The label is either
        "Fighting" or "Normal"; "walking" is intentionally hidden.
        """
        if not self.enabled or self.model is None:
            return self._last_result

        self._frame_counter += 1
        if self._frame_counter % self.run_every_n_frames != 0:
            return self._last_result

        if len(persons) < self.min_persons_for_fighting:
            self._last_result = self._empty_result()
            return self._last_result

        best_result = self._empty_result()
        for first, second in self._candidate_pairs(persons):
            crop = self._pair_crop(frame, first, second)
            if crop is None or crop.size == 0:
                continue
            try:
                result = self.model.predict(source=crop, imgsz=224, device="cpu", verbose=False)[0]
            except Exception as exc:
                print(f"[ActionFrameClassifier] Inference failed: {exc}")
                return self._last_result

            if result.probs is None:
                continue

            class_idx = int(result.probs.top1)
            raw_label = result.names[class_idx].lower()
            confidence = float(result.probs.top1conf)
            is_fighting = (
                raw_label == self.positive_class
                and confidence >= self.confidence_threshold
            )
            if is_fighting and confidence > best_result["confidence"]:
                best_result = {
                    "label": self.positive_display_label,
                    "confidence": confidence,
                    "raw_label": raw_label,
                    "track_ids": [first.track_id, second.track_id],
                }

        self._last_result = best_result
        return self._last_result

    def apply_to_persons(self, persons, frame_result):
        """Apply the global fight result only to the most plausible interacting pair."""
        if frame_result.get("label") != self.positive_display_label:
            return []

        detections = []
        confidence = float(frame_result.get("confidence", 0.0))
        target_ids = set(frame_result.get("track_ids", []))
        targets = [person for person in persons if person.track_id in target_ids]
        if not targets:
            targets = self._select_interacting_persons(persons)
        for person in targets:
            person.event_type = self.positive_display_label
            person.action_confidence = confidence
            person.action_probabilities = {
                self.positive_display_label: confidence,
                self.negative_display_label: max(0.0, 1.0 - confidence),
            }
            detections.append({
                "person": person,
                "event_type": self.positive_display_label,
                "confidence": confidence,
                "source": "action_frame_classifier",
            })
        return detections

    def _candidate_pairs(self, persons):
        pairs = []
        for idx, first in enumerate(persons):
            fx, fy = first.center
            for second in persons[idx + 1:]:
                sx, sy = second.center
                distance = math.hypot(fx - sx, fy - sy)
                pairs.append((distance, first, second))
        pairs.sort(key=lambda item: item[0])
        return [(first, second) for _, first, second in pairs[:3]]

    def _pair_crop(self, frame, first, second):
        h, w = frame.shape[:2]
        x1 = max(0, min(first.bbox[0], second.bbox[0]))
        y1 = max(0, min(first.bbox[1], second.bbox[1]))
        x2 = min(w, max(first.bbox[2], second.bbox[2]))
        y2 = min(h, max(first.bbox[3], second.bbox[3]))
        pad_x = int((x2 - x1) * 0.18)
        pad_y = int((y2 - y1) * 0.18)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _select_interacting_persons(self, persons):
        """Pick the closest pair so a scene-level fight label does not tag everyone."""
        if len(persons) <= 2:
            return list(persons)

        best_pair = None
        best_distance = None
        for idx, first in enumerate(persons):
            fx, fy = first.center
            for second in persons[idx + 1:]:
                sx, sy = second.center
                distance = ((fx - sx) ** 2 + (fy - sy) ** 2) ** 0.5
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (first, second)

        return list(best_pair) if best_pair else list(persons[:2])
