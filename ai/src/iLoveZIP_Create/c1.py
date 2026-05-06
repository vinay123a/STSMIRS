import cv2
import numpy as np
import tensorflow as tf
import os

# ---------------- LOAD ACTION MODEL ----------------
print("Loading Action Model...")
model = tf.keras.models.load_model("src/iLoveZIP_Create/action_model.h5")
actions = ["fall", "fighting"]  # change if needed

SEQUENCE_LENGTH = 30
sequence = []

# ---------------- CAMERA ----------------
# If using DroidCam via WiFi (Browser IP Cam), use the URL from the app:
# CAM_SOURCE = "http://192.168.1.xxx:4747/video" 
#
# If using DroidCam PC Client (Virtual Webcam), try 1 or 2:
# CAM_SOURCE = 1
#
# For laptop's default webcam, use 0:
CAM_SOURCE = "http://10.240.32.136:4747/video"

cap = cv2.VideoCapture(CAM_SOURCE)

frame_count = 0
last_action = "..."

print("Starting Camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ---------------- ACTION RECOGNITION ----------------
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0

    sequence.append(img)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) == SEQUENCE_LENGTH and frame_count % 10 == 0:
        input_data = np.expand_dims(sequence, axis=0)
        pred = model.predict(input_data, verbose=0)

        confidence = np.max(pred)

        if confidence > 0.7:
            last_action = actions[np.argmax(pred)]
        else:
            last_action = "uncertain"

    # ---------------- DRAW ACTION ----------------
    cv2.putText(frame, f"Action: {last_action}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,0,255),
                3)

    cv2.imshow("Action Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()