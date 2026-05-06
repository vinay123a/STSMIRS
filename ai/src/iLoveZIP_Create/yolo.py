import cv2
import os
import logging
from ultralytics import YOLO
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading

# 🔥 Disable YOLO logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

print("===========================================")
print("  STSMIRS — Fast YOLO + Facenet (PyTorch)  ")
print("===========================================")

# ---------------- INITIALIZE AI ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Face AI on {device.upper()}...")
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

model = YOLO("../../models/yolov8n.pt") if not os.path.exists("yolov8n.pt") else YOLO("yolov8n.pt")

# ---------------- LOAD FACE DATA ----------------
import pickle

known_encodings = []
known_names = []

faces_path = "faces"
if not os.path.exists(faces_path):
    if os.path.exists("project_videos/faces"):
        faces_path = "project_videos/faces"
    elif os.path.exists("../../project_videos/faces"):
        faces_path = "../../project_videos/faces"

cache_path = os.path.join(faces_path if faces_path else ".", "faces_cache.pkl")

if faces_path and os.path.exists(cache_path):
    print("Loading faces from cache (Instant!)...")
    with open(cache_path, "rb") as f:
        known_encodings, known_names = pickle.load(f)
elif not os.path.exists(faces_path):
    print(f"⚠ WARNING: Faces folder '{faces_path}' not found!")
else:
    print(f"Processing images (This may take a few minutes for 1400 images)...")
    for person_name in os.listdir(faces_path):
        person_path = os.path.join(faces_path, person_name)
        if not os.path.isdir(person_path): continue

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")): continue

            img_path = os.path.join(person_path, file)
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = mtcnn(img_rgb)
                if faces is not None:
                    emb = resnet(faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()
                    known_encodings.append(emb[0])
                    known_names.append(person_name)
            except Exception:
                pass
                
    # Save cache
    with open(cache_path, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

print("Loaded:", set(known_names))

# ---------------- CAMERA INIT ----------------
cap = cv2.VideoCapture("http://172.28.49.230:4747/video")

frame_count = 0
yolo_boxes = []
detected_faces = [] # List of (x1, y1, x2, y2, name)
is_facenet_running = False

def run_facenet(frame_copy):
    global detected_faces, is_facenet_running
    try:
        rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        
        # 1. Detect faces
        boxes, _ = mtcnn.detect(rgb)
        if boxes is None:
            detected_faces = []
            return
            
        # 2. Extract crops and embed
        faces_tensor = mtcnn(rgb)
        if faces_tensor is None:
            detected_faces = []
            return
            
        embeddings = resnet(faces_tensor.to(device)).detach().cpu().numpy()
        
        new_faces = []
        for i, box in enumerate(boxes):
            name = "Unknown"
            if len(known_encodings) > 0:
                # Calculate Euclidean distances
                dists = np.linalg.norm(known_encodings - embeddings[i], axis=1)
                best_idx = np.argmin(dists)
                # threshold is roughly 0.8 for vggface2
                if dists[best_idx] < 0.8:
                    name = known_names[best_idx]
            
            x1, y1, x2, y2 = map(int, box)
            new_faces.append((x1, y1, x2, y2, name))
            
        detected_faces = new_faces
    except Exception as e:
        pass
    finally:
        is_facenet_running = False

print("\nStarting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1

    # ---------------- YOLO (EVERY 5 FRAMES) ----------------
    if frame_count % 5 == 0:
        results = model(frame, verbose=False)
        yolo_boxes = []
        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_boxes.append((x1, y1, x2, y2))

    # ---------------- FACENET (BACKGROUND THREAD) ----------------
    if not is_facenet_running and frame_count % 15 == 0:
        is_facenet_running = True
        threading.Thread(target=run_facenet, args=(frame.copy(),), daemon=True).start()

    # 🔵 DRAW FACE + NAME
    for (x1, y1, x2, y2, name) in detected_faces:
        color = (255, 0, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 🟢 DRAW YOLO PERSON
    for (x1, y1, x2, y2) in yolo_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face + YOLO (Fixed)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()