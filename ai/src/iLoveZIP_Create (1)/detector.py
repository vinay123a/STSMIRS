"""
STSMIRS — YOLOv8 Person Detector
Detects persons in frames using YOLOv8. Returns bounding boxes + confidence.

Usage (standalone test):
    python src/detector.py
    python src/detector.py --source 0   (laptop webcam)
"""

import cv2
import json
import os
import time
import numpy as np

if not os.environ.get("YOLO_CONFIG_DIR"):
    os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")

from ultralytics import YOLO


class PersonDetector:
    """YOLOv8-based person detector."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        det_cfg = config["detector"]
        self.confidence_threshold = det_cfg.get("confidence_threshold", 0.5)
        self.iou_threshold = det_cfg.get("iou_threshold", 0.45)
        self.classes = det_cfg.get("classes", [0])  # 0 = person in COCO
        self.imgsz = det_cfg.get("imgsz", 640)

        # Determine device
        device = det_cfg.get("device", "auto")
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load YOLOv8 model (auto-downloads if not found)
        model_name = det_cfg.get("model", "yolov8n.pt")
        print(f"[Detector] Loading model: {model_name} on {self.device}")
        self.model = YOLO(model_name)
        print(f"[Detector] ✓ Model loaded.")

    def detect(self, frame):
        """
        Run person detection on a single frame.

        Args:
            frame: BGR numpy array (OpenCV format)

        Returns:
            list of dicts, each with:
                - bbox: [x1, y1, x2, y2] (pixel coords, int)
                - confidence: float (0-1)
                - center: (cx, cy) center point
                - foot_point: (cx, y2) bottom-center (for zone checks)
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()       # [N, 4]
                confs = result.boxes.conf.cpu().numpy()        # [N]

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(confs[i]),
                        "center": (cx, cy),
                        "foot_point": (cx, int(y2))  # Bottom-center for zone checking
                    })

        return detections

    def detect_and_draw(self, frame, color=(0, 255, 0), thickness=2):
        """
        Detect + draw bounding boxes on frame (for testing).
        Returns (annotated_frame, detections).
        """
        detections = self.detect(frame)
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"Person {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 6),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw foot point
            cv2.circle(annotated, det["foot_point"], 4, (0, 0, 255), -1)

        return annotated, detections


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/detector.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import sys
    import os

    # Ensure we can import from project root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="STSMIRS Person Detector Test")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None,
                        help="Video source: URL, file path, or '0' for webcam")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        exit(1)

    # Initialize detector
    detector = PersonDetector(args.config)

    # Determine video source
    if args.source:
        source = 0 if args.source == "0" else args.source
    else:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        source = cfg["stream"]["url"]
        # Try fallback if URL doesn't work
        fallback = cfg["stream"].get("fallback_url", "0")

    print("=" * 50)
    print("  STSMIRS — Person Detector Test")
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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.5)
            continue

        # Detect
        annotated, detections = detector.detect_and_draw(frame)

        # FPS counter
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()

        # Draw FPS + person count
        cv2.putText(annotated, f"FPS: {current_fps:.1f} | Persons: {len(detections)}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, "STSMIRS Detector Test | 'q' to quit",
                     (10, annotated.shape[0] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("STSMIRS Detector Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Detector test complete.")
