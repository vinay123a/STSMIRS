import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import os
import logging

# 🔥 Disable YOLO logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

print("===========================================")
print("  STSMIRS — Fast YOLO + Face_Recognition   ")
print("===========================================")

# ---------------- LOAD FACE DATA ----------------
known_encodings = []
known_names = []

# Support 'faces', or relative paths to STSMIRS root
faces_path = "faces"
if not os.path.exists(faces_path):
    if os.path.exists("project_videos/faces"):
        faces_path = "project_videos/faces"
    elif os.path.exists("../../project_videos/faces"):
        faces_path = "../../project_videos/faces"

if not os.path.exists(faces_path):
    print(f"⚠ WARNING: Faces folder '{faces_path}' not found!")
else:
    for person_name in os.listdir(faces_path):
        person_path = os.path.join(faces_path, person_name)

        if not os.path.isdir(person_path):
            continue

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, file)
            try:
                img = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(img)

                if enc:
                    known_encodings.append(enc[0])
                    known_names.append(person_name)
            except Exception:
                pass

print("Loaded:", set(known_names))

# ---------------- INIT ----------------
model = YOLO("../../models/yolov8n.pt") if not os.path.exists("yolov8n.pt") else YOLO("yolov8n.pt")
cap = cv2.VideoCapture("http://172.28.49.230:4747/video")

frame_count = 0
face_locations = []
face_names = []
yolo_boxes = []

print("\nStarting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ---------------- FACE RECOGNITION (EVERY 10 FRAMES) ----------------
    if frame_count % 10 == 0:
        # Convert raw frame directly (no resizing) to guarantee strict memory layout for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Force a completely clean memory block of type uint8
        rgb = np.array(rgb, dtype=np.uint8, copy=True)

        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)

        face_names = []

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(distances) > 0:
                best = distances.argmin()

                if matches[best]:
                    name = known_names[best]

            face_names.append(name)

    # ---------------- YOLO (EVERY 10 FRAMES) ----------------
    if frame_count % 10 == 0:
        results = model(frame, verbose=False)

        yolo_boxes = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_boxes.append((x1, y1, x2, y2))

    # 🔵 DRAW FACE + NAME
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # 🟢 DRAW YOLO PERSON
    for (x1, y1, x2, y2) in yolo_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    pass # ("Face + YOLO (Fixed)", frame)

    if ord('q') if frame_count > 15 else 255 & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()