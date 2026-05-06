"""
QUICK TEST — Skeleton Fall Detection
Run this directly to see if the skeleton model is actually detecting falls.
Usage: python src/test_fall_detection.py --source "http://172.28.49.230:4747/video"
"""

import cv2
import sys
import os
import numpy as np
import torch
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0")
args = parser.parse_args()

source = 0 if args.source == "0" else args.source

print("=" * 50)
print("  FALL DETECTION — Quick Test")
print(f"  Source: {source}")
print("  Press 'q' to quit")
print("=" * 50)

# ─── 1. Load Skeleton Pose Model ───────────────────────────────
print("\n[1] Loading YOLOv8-Pose model...")
from ultralytics import YOLO
pose_model = YOLO("models/yolov8n-pose.pt")
print("    OK!")

# ─── 2. Load Skeleton LSTM Classifier ─────────────────────────
print("[2] Loading Skeleton LSTM classifier...")
from src.skeleton_action_detector import SkeletonLSTMClassifier
import json

with open("config.json") as f:
    cfg = json.load(f)

skel_cfg = cfg.get("skeleton_action", {})
classes = skel_cfg.get("class_names", ["Walking", "Running", "Loitering", "Fall", "Lying_Still", "Fighting", "Panic"])
input_dim = skel_cfg.get("input_dim", 51)
hidden_dim = skel_cfg.get("hidden_size", 256)
num_layers = skel_cfg.get("num_layers", 2)
num_classes = skel_cfg.get("num_classes", len(classes))

model = SkeletonLSTMClassifier(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_classes=num_classes
)
model_path = "models/skeleton_lstm.pth"
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    print(f"    Loaded weights from {model_path}")
else:
    print(f"    WARNING: {model_path} not found! Using random weights.")
model.eval()
print("    OK!")

# ─── 3. Open Camera ───────────────────────────────────────────
print(f"\n[3] Opening camera: {source}")
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("    ERROR: Cannot open camera!")
    sys.exit(1)
print("    OK!")

# ─── 4. Inference Loop ────────────────────────────────────────
skeleton_buffer = deque(maxlen=30)  # Keep last 30 frames
MIN_FRAMES = 5  # Predict after 5 frames

print("\n[4] Running... stand in front of camera and do a FALL!")
print("    You will see what the AI thinks every second.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame read failed")
        continue

    # Run pose estimation
    results = pose_model(frame, verbose=False, conf=0.3)
    
    display = frame.copy()
    label_text = "No person detected"
    color = (100, 100, 100)

    if results and results[0].keypoints is not None:
        kpts_data = results[0].keypoints.data.cpu().numpy()
        
        if len(kpts_data) > 0:
            # Use first detected person's skeleton
            kpts = kpts_data[0]  # shape (17, 3)
            
            # Draw skeleton keypoints on screen
            h, w = frame.shape[:2]
            for kp in kpts:
                x, y, conf = kp
                if conf > 0.3:
                    cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Normalize keypoints
            normalized = kpts.copy()
            normalized[:, 0] = 2.0 * (kpts[:, 0] / w) - 1.0
            normalized[:, 1] = 2.0 * (kpts[:, 1] / h) - 1.0
            flattened = normalized.flatten()  # (51,)
            skeleton_buffer.append(flattened)

            frames_collected = len(skeleton_buffer)

            if frames_collected >= MIN_FRAMES:
                # Run LSTM inference
                sequence = np.array(list(skeleton_buffer))  # (N, 51)
                x = torch.FloatTensor(sequence).unsqueeze(0)  # (1, N, 51)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).numpy()[0]

                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx]
                confidence = float(probs[pred_idx])

                # Build prob string
                prob_str = " | ".join([f"{c}:{probs[i]:.2f}" for i, c in enumerate(classes)])
                print(f"[AI] {pred_class} ({confidence:.2f}) | Frames:{frames_collected} | {prob_str}")

                if pred_class in ["Fall", "Lying_Still"]:
                    label_text = f"FALL DETECTED! ({confidence:.0%})"
                    color = (0, 0, 255)
                elif pred_class == "Fighting":
                    label_text = f"FIGHTING! ({confidence:.0%})"
                    color = (0, 165, 255)
                else:
                    label_text = f"{pred_class} ({confidence:.0%})"
                    color = (0, 255, 0)
            else:
                label_text = f"Collecting frames... {frames_collected}/{MIN_FRAMES}"
                color = (255, 255, 0)
        else:
            skeleton_buffer.clear()
    else:
        skeleton_buffer.clear()

    # Draw label on screen
    cv2.rectangle(display, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(display, label_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Fall Detection Test", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[Done] Test complete.")
