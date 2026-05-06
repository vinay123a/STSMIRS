import cv2
import os

dataset_path = "dataset"
output_path = "dataset_frames"

os.makedirs(output_path, exist_ok=True)

for action in os.listdir(dataset_path):
    action_path = os.path.join(dataset_path, action)
    save_action_path = os.path.join(output_path, action)

    os.makedirs(save_action_path, exist_ok=True)

    for video in os.listdir(action_path):
        video_path = os.path.join(action_path, video)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        video_name = os.path.splitext(video)[0]
        save_folder = os.path.join(save_action_path, video_name)
        os.makedirs(save_folder, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(save_folder, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()

print("Frames extraction completed.")