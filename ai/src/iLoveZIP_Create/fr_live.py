import cv2
import face_recognition
import os

print("===========================================")
print("  STSMIRS — Fast Face Recognition          ")
print("===========================================")

# ---------------- LOAD FACES ----------------
known_encodings = []
known_names = []

# Support 'faces', or relative paths to STSMIRS root
faces_path = "faces"
if not os.path.exists(faces_path):
    if os.path.exists("project_videos/faces"):
        faces_path = "project_videos/faces"
    elif os.path.exists("../../project_videos/faces"):
        faces_path = "../../project_videos/faces"

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
            encodings = face_recognition.face_encodings(img)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

        except:
            pass

print("Loaded:", set(known_names))

# ---------------- CAMERA ----------------
# Using DroidCam instead of default webcam
cap = cv2.VideoCapture("http://172.28.49.230:4747/video")

frame_count = 0

# Store last results
face_locations = []
face_names = []

print("\nStarting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    import numpy as np
    
    # Convert raw frame directly (no resizing) to guarantee strict memory layout for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Force a completely clean memory block of type uint8
    rgb = np.array(rgb, dtype=np.uint8, copy=True)

    # 🔥 ONLY RUN HEAVY CODE EVERY 10 FRAMES
    if frame_count % 10 == 0:
        face_locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, face_locations)

        face_names = []

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match = face_distances.argmin()

                if matches[best_match]:
                    name = known_names[best_match]

            face_names.append(name)

    # 🔥 DRAW EVERY FRAME (FAST)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back
        top *= 2.5
        right *= 2.5
        bottom *= 2.5
        left *= 2.5

        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

        cv2.putText(frame, name, (int(left), int(top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition (Smooth)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()