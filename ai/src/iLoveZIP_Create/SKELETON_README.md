"""
STSMIRS — Skeleton-Based Face Recognition & Action Detection System
Comprehensive guide and API reference
"""

# Skeleton-Based Face Recognition & Action Detection

## Overview

This system extends STSMIRS with skeleton-based (pose estimation) face recognition and action detection capabilities. Using YOLOv8 Pose, the system extracts human body keypoints and uses them to:

1. **Enhance Action Detection** - Recognize actions (Walking, Running, Falling, Fighting, Panic) based on body pose sequences
2. **Improve Face Recognition** - Add skeleton context to boost face identification accuracy
3. **Provide Visual Context** - Display skeleton overlays for debugging and analysis

## Architecture

```
Camera/Video Input
    ↓
[Detection & Tracking] (YOLOv8 Object Detection)
    ↓
[Skeleton Extraction] (YOLOv8 Pose)
    ├── Normalized Keypoints (17 COCO joints)
    ├── Joint Angles
    ├── Limb Lengths
    └── Body Distances
    ↓
[Enhanced Face Recognition]
    ├── Traditional Face Recognition
    ├── + Skeleton Context
    └── → Person Identity
    ↓
[Action Detection]
    ├── Traditional LSTM (frame features)
    ├── + Skeleton LSTM (pose sequences)
    └── → Action Label (Walking, Running, Fall, etc.)
    ↓
[Event Triggering & Display]
```

## Modules

### 1. Skeleton Extractor (`src/skeleton_extractor.py`)

Extracts human pose keypoints from video frames using YOLOv8 Pose.

**Key Classes:**
- `SkeletonExtractor`: Main skeleton extraction engine

**Key Methods:**
- `extract_skeleton(frame)` - Extract keypoints from single frame
- `normalize_keypoints(keypoints, h, w)` - Normalize to [-1, 1] range
- `extract_skeleton_features(keypoints)` - Compute angles, distances, limb lengths
- `draw_skeleton(frame, keypoints)` - Visualize on frame

**COCO Keypoints (17 points):**
```
 0: nose           1: left_eye      2: right_eye     3: left_ear      4: right_ear
 5: left_shoulder  6: right_shoulder 7: left_elbow   8: right_elbow   9: left_wrist
10: right_wrist   11: left_hip      12: right_hip    13: left_knee    14: right_knee
15: left_ankle    16: right_ankle
```

### 2. Skeleton Action Detector (`src/skeleton_action_detector.py`)

LSTM-based action classifier using skeleton sequences.

**Key Classes:**
- `SkeletonLSTMClassifier` - PyTorch LSTM model
- `SkeletonActionDetector` - Inference engine

**Architecture:**
```
Input: (batch, seq_len=30, input_dim=51)
  ↓
FC: 51 → 128
  ↓
LSTM: 128 → 256 (2 layers)
  ↓
FC: 256 → 7 (action classes)
  ↓
Output: Action logits
```

**Supported Actions:**
- Walking
- Running
- Loitering
- Fall
- Lying_Still
- Fighting
- Panic

### 3. Skeleton-Aware Face Recognizer (`src/skeleton_face_recognizer.py`)

Combines face recognition with skeleton context for improved accuracy.

**Key Classes:**
- `SkeletonAwareFaceRecognizer`: Fusion engine

**Features:**
- Weighted fusion of face and skeleton confidence
- Pose signature matching for person profiles
- Skeleton-based gait recognition
- Multi-person skeleton tracking

### 4. Training Pipeline (`src/train_skeleton_lstm.py`)

Train skeleton LSTM model on action datasets.

**Dataset Format:**
- Input: Action videos organized by class
- Processing: Extract skeleton sequences from videos
- Output: Trained LSTM model

**Usage:**
```bash
python src/train_skeleton_lstm.py --config config.json
```

**Configuration (config.json):**
```json
{
  "training": {
    "data_dir": "data/features",
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "val_split": 0.2
  },
  "skeleton_action": {
    "model_path": "models/skeleton_lstm.pth",
    "sequence_length": 30,
    "hidden_size": 256,
    "num_layers": 2,
    "num_classes": 7,
    "input_dim": 51,
    "dropout": 0.3
  }
}
```

### 5. Feature Extraction Tool (`tools/extract_skeleton_features.py`)

Extract skeleton features from video datasets for training.

**Modes:**
- `--mode videos` - Extract from video files
- `--mode features` - Convert existing feature files

**Usage:**
```bash
python tools/extract_skeleton_features.py \
  --mode videos \
  --source dataset \
  --output data/skeleton_features \
  --classes Fall Fighting Loitering Running Walking Panic
```

### 6. Evaluation Tools (`tools/skeleton_eval.py`)

Evaluate and visualize skeleton-based models.

**Features:**
- Single video evaluation
- Dataset-level evaluation
- Confusion matrix plots
- Performance metrics by class
- Annotated video generation

**Usage - Evaluate video:**
```bash
python tools/skeleton_eval.py \
  --mode evaluate \
  --video test_video.mp4 \
  --output results.json
```

**Usage - Create annotated video:**
```bash
python tools/skeleton_eval.py \
  --mode visualize \
  --video test_video.mp4 \
  --output annotated_video.mp4
```

## Integration with Main Pipeline

The skeleton system is integrated into `src/main.py`:

1. **Initialization Phase:**
   - SkeletonExtractor loaded with YOLOv8 Pose model
   - SkeletonActionDetector initialized with trained LSTM
   - SkeletonAwareFaceRecognizer created with face recognizer

2. **Processing Loop:**
   ```python
   # Step 1: Extract skeleton
   skeleton_result = skeleton_extractor.extract_skeleton(frame)
   
   # Step 2: Update skeleton buffers and extract features
   for person in persons:
       person['skeleton_keypoints'] = normalized_kpts
       person['skeleton_features'] = skeleton_extractor.extract_skeleton_features(kpts)
       skeleton_action_detector.update_skeleton(person_id, normalized_kpts)
   
   # Step 3: Get skeleton-based predictions
   skeleton_pred = skeleton_action_detector.predict(person_id)
   
   # Step 4: Fuse with traditional action detection
   # Combine skeleton_pred['action'] with action_detector results
   ```

3. **Output:**
   - Each person object contains: `skeleton_keypoints`, `skeleton_features`, `skeleton_action`, `skeleton_confidence`
   - Used for event triggering and display

## Configuration Reference

### skeleton (Pose Extraction)
```json
{
  "skeleton": {
    "model_path": "models/yolov8n-pose.pt",
    "confidence_threshold": 0.5,
    "device": "auto",
    "imgsz": 640,
    "use_angles": true,
    "use_distances": true,
    "use_velocities": true,
    "keypoint_confidence_threshold": 0.3,
    "max_frames_per_video": 300,
    "frame_skip": 1
  }
}
```

### skeleton_action (Action Detection)
```json
{
  "skeleton_action": {
    "model_path": "models/skeleton_lstm.pth",
    "sequence_length": 30,
    "hidden_size": 256,
    "num_layers": 2,
    "num_classes": 7,
    "input_dim": 51,
    "dropout": 0.3,
    "device": "auto",
    "postprocess": {
      "confidence_threshold": 0.5,
      "use_smoothing": true,
      "smoothing_alpha": 0.4
    }
  }
}
```

### skeleton_face_recognition (Enhanced Face ID)
```json
{
  "skeleton_face_recognition": {
    "use_skeleton_context": true,
    "skeleton_confidence_weight": 0.3,
    "pose_similarity_threshold": 0.7
  }
}
```

## Feature Details

### Skeleton Features Extracted

1. **Limb Lengths** (6 values):
   - Left arm (shoulder → wrist)
   - Right arm (shoulder → wrist)
   - Left leg (hip → ankle)
   - Right leg (hip → ankle)
   - Torso height (left & right)

2. **Joint Angles** (8 values):
   - Left elbow angle
   - Right elbow angle
   - Left knee angle
   - Right knee angle
   - Left hip angle
   - Right hip angle
   - Shoulder width
   - Hip width

3. **Body Distances** (10 values):
   - Nose to hips (left & right)
   - Shoulder width
   - Hip width
   - Diagonal measurements
   - Extremity distances

### Data Flow

```
Raw Frame (H, W, 3)
    ↓
YOLOv8 Pose Inference
    ↓
Keypoints (17, 3) [x, y, confidence]
    ↓
Normalize to [-1, 1]
    ↓
Extract Features (angles, distances, limb lengths)
    ↓
Buffer Sequence (30 frames)
    ↓
LSTM Forward Pass
    ↓
Action Logits (7,)
    ↓
Softmax → Probabilities
    ↓
Threshold & Smoothing
    ↓
Final Action Prediction
```

## Performance Considerations

### Computational Costs:
- **Skeleton Extraction:** ~30-50ms per frame (GPU) / ~100-150ms (CPU)
- **LSTM Inference:** ~5-10ms per frame
- **Total Overhead:** ~40-60ms per frame (GPU)

### Memory Requirements:
- **Skeleton Buffers:** ~30 frames × 51 dims × 4 bytes × num_persons
- **Model Weights:** ~20MB (LSTM) + ~100MB (Pose model)

### Optimization Tips:
1. Use `frame_skip=2` to process every 2nd frame
2. Reduce `imgsz=480` for faster inference
3. Use smaller pose model: `yolov8s-pose.pt` or `yolov8n-pose.pt`
4. Enable GPU acceleration with proper CUDA setup

## Troubleshooting

### No skeleton keypoints detected:
- Check `keypoint_confidence_threshold` (try lowering from 0.3 to 0.1)
- Verify model path exists
- Ensure good lighting and clear person visibility

### Action detection accuracy low:
- Increase `sequence_length` to 40-60 frames
- Check skeleton extraction quality
- Verify training data quality
- Tune class thresholds in `skeleton_action.postprocess`

### Face recognition not matching with skeleton context:
- Adjust `skeleton_confidence_weight` (currently 0.3)
- Check `pose_similarity_threshold`
- Ensure face enrollment samples are diverse

## Example Usage

### Run Live Pipeline with Skeleton:
```bash
python src/main.py --config config.json --source 0
```

### Train Skeleton Action Model:
```bash
# 1. Extract features from videos
python tools/extract_skeleton_features.py \
  --mode videos \
  --source dataset \
  --output data/skeleton_features

# 2. Train model
python src/train_skeleton_lstm.py --config config.json
```

### Evaluate Model:
```bash
python tools/skeleton_eval.py \
  --mode evaluate \
  --dataset dataset \
  --output evaluation_results.json
```

### Create Visualization:
```bash
python tools/skeleton_eval.py \
  --mode visualize \
  --video test_video.mp4 \
  --output annotated.mp4
```

## Dependencies

Core dependencies (see requirements.txt):
- `torch>=2.0.0` - Deep learning framework
- `ultralytics>=8.1.0` - YOLOv8 models (Pose & Detection)
- `opencv-python>=4.8.0` - Computer vision
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn`, `matplotlib`, `seaborn` - Analysis & visualization

## Future Enhancements

1. **Multi-person Action Recognition** - Detect group actions (coordinated fighting)
2. **Temporal Models** - Use Temporal Convolutional Networks (TCN)
3. **Attention Mechanisms** - Focus on relevant joints
4. **Transfer Learning** - Pre-trained pose models from other datasets
5. **3D Skeleton** - Add Z-dimension from depth sensors
6. **Real-time Optimization** - ONNX export for faster inference
7. **Ensemble Methods** - Combine skeleton + frame-based + face features

## API Examples

### Extract Skeleton from Frame:
```python
from src.skeleton_extractor import SkeletonExtractor
import cv2

extractor = SkeletonExtractor('config.json')
frame = cv2.imread('image.jpg')

result = extractor.extract_skeleton(frame)
keypoints = result['keypoints']  # (N_persons, 17, 3)

if len(keypoints) > 0:
    person_skeleton = keypoints[0]  # First person
    features = extractor.extract_skeleton_features(person_skeleton)
```

### Predict Action from Skeleton Sequence:
```python
from src.skeleton_action_detector import SkeletonActionDetector

detector = SkeletonActionDetector('config.json')

# Feed skeleton frames (already buffered)
for i in range(30):
    normalized_kpts = extractor.normalize_keypoints(kpts, h, w)
    detector.update_skeleton(track_id=0, normalized_keypoints=normalized_kpts)

# Predict
result = detector.predict(track_id=0)
print(f"Action: {result['action']}, Confidence: {result['confidence']:.3f}")
```

### Enhanced Face Recognition with Skeleton:
```python
from src.skeleton_face_recognizer import SkeletonAwareFaceRecognizer

enhanced_recognizer = SkeletonAwareFaceRecognizer(
    face_recognizer, skeleton_extractor, 'config.json'
)

result = enhanced_recognizer.recognize_person_enhanced(frame, bbox)
print(f"Face Conf: {result.get('face_confidence')}")
print(f"Skeleton Conf: {result.get('skeleton_confidence')}")
print(f"Fused Conf: {result.get('confidence')}")
```

---

**Last Updated:** May 2026
**Version:** 1.0.0
