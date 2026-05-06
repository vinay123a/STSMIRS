import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

DATA_PATH = "dataset_frames"
SEQUENCE_LENGTH = 30

actions = sorted(os.listdir(DATA_PATH))
print(actions)
label_map = {label: num for num, label in enumerate(actions)}

sequences = []
labels = []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    for video in os.listdir(action_path):
        video_path = os.path.join(action_path, video)

        frames = []
        frame_files = sorted(os.listdir(video_path))

        for frame_file in frame_files[:SEQUENCE_LENGTH]:
            img_path = os.path.join(video_path, frame_file)
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            frame = cv2.resize(frame, (64, 64))
            frame = frame / 255.0
            frames.append(frame)

        if len(frames) == SEQUENCE_LENGTH:
            sequences.append(frames)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Shuffle data
X, y = shuffle(X, y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Data preparation done!")
print("X shape:", X.shape)
print("y shape:", y.shape)