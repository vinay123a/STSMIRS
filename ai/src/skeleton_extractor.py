"""
STSMIRS — Skeleton Pose Extractor
Extracts human pose keypoints using YOLOv8 Pose model.
Provides skeleton-based features for action detection and face recognition enhancement.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[SkeletonExtractor] WARNING: ultralytics not installed. Skeleton extraction disabled.")


class SkeletonExtractor:
    """
    Extracts human pose skeleton from video frames using YOLOv8 Pose.
    Provides normalized keypoints, skeleton angles, and movement features.
    """

    # COCO keypoint names (17 points)
    KEYPOINT_NAMES = [
        "nose",           # 0
        "left_eye",       # 1
        "right_eye",      # 2
        "left_ear",       # 3
        "right_ear",      # 4
        "left_shoulder",  # 5
        "right_shoulder", # 6
        "left_elbow",     # 7
        "right_elbow",    # 8
        "left_wrist",     # 9
        "right_wrist",    # 10
        "left_hip",       # 11
        "right_hip",      # 12
        "left_knee",      # 13
        "right_knee",     # 14
        "left_ankle",     # 15
        "right_ankle",    # 16
    ]

    # COCO skeleton connections (17 keypoints)
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12),                # Torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    ]

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.skeleton_cfg = config.get("skeleton", {})
        self.model_path = self.skeleton_cfg.get("model_path", "models/yolov8n-pose.pt")
        self.confidence_threshold = self.skeleton_cfg.get("confidence_threshold", 0.5)
        self.device = self.skeleton_cfg.get("device", "auto")
        self.imgsz = self.skeleton_cfg.get("imgsz", 640)
        
        # Feature extraction settings
        self.use_angles = self.skeleton_cfg.get("use_angles", True)
        self.use_distances = self.skeleton_cfg.get("use_distances", True)
        self.use_velocities = self.skeleton_cfg.get("use_velocities", True)
        
        # Keypoint confidence threshold
        self.keypoint_confidence_threshold = self.skeleton_cfg.get("keypoint_confidence_threshold", 0.3)
        
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 Pose model."""
        if not YOLO_AVAILABLE:
            print("[SkeletonExtractor] ERROR: ultralytics not installed.")
            return

        if not os.path.exists(self.model_path):
            print(f"[SkeletonExtractor] Model not found at {self.model_path}. Downloading...")
            # YOLOv8 will auto-download if model name is provided without full path
            try:
                self.model = YOLO("yolov8n-pose.pt")
                print("[SkeletonExtractor] Downloaded yolov8n-pose model")
            except Exception as e:
                print(f"[SkeletonExtractor] ERROR loading model: {e}")
                self.model = None
        else:
            try:
                self.model = YOLO(self.model_path)
                print(f"[SkeletonExtractor] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[SkeletonExtractor] ERROR loading model: {e}")
                self.model = None

    def extract_skeleton(self, frame):
        """
        Extract skeleton keypoints from a single frame.
        
        Returns:
            dict: {
                'detections': list of detection objects,
                'keypoints': ndarray of shape (N, 17, 3) [x, y, confidence],
                'frame_height': int,
                'frame_width': int
            }
        """
        if self.model is None:
            return {
                'detections': [],
                'keypoints': np.array([]),
                'frame_height': frame.shape[0],
                'frame_width': frame.shape[1]
            }

        try:
            # Run inference. Some ultralytics versions expect a list or RGB frames;
            # try the straightforward call first, then fallback to list-wrapped or RGB.
            try:
                results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            except Exception as e_direct:
                # Prepare debug info
                info = f"Direct inference failed (type={type(frame)}, shape={getattr(frame, 'shape', None)}, dtype={getattr(frame, 'dtype', None)}): {e_direct}"
                print(f"[SkeletonExtractor] {info}")
                # Try RGB conversion
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    frame_rgb = frame
                try:
                    results = self.model([frame_rgb], verbose=False, conf=self.confidence_threshold)
                    print("[SkeletonExtractor] Inference succeeded with list-wrapped/RGB frame fallback")
                except Exception as e_list:
                    raise RuntimeError(f"Model inference failed: {e_direct}; fallback error: {e_list}")

            keypoints_list = []
            detections = results[0] if isinstance(results, (list, tuple)) else results

            if hasattr(detections, 'keypoints') and detections.keypoints is not None:
                # keypoints.data shape: (num_detections, 17, 3) for COCO
                keypoints_data = detections.keypoints.data.cpu().numpy()
                keypoints_list = keypoints_data

            return {
                'detections': detections,
                'keypoints': np.array(keypoints_list) if len(keypoints_list) > 0 else np.array([]),
                'frame_height': frame.shape[0],
                'frame_width': frame.shape[1]
            }

        except Exception as e:
            print(f"[SkeletonExtractor] Error extracting skeleton: {e}")
            return {
                'detections': [],
                'keypoints': np.array([]),
                'frame_height': frame.shape[0],
                'frame_width': frame.shape[1]
            }

    def normalize_keypoints(self, keypoints, frame_height, frame_width):
        """
        Normalize keypoints to [-1, 1] range relative to frame dimensions.
        
        Args:
            keypoints: ndarray of shape (17, 3) [x, y, confidence]
            frame_height: int
            frame_width: int
            
        Returns:
            ndarray of shape (17, 3) with normalized coordinates
        """
        normalized = keypoints.copy()
        
        # Normalize x and y to [-1, 1]
        normalized[:, 0] = 2.0 * (keypoints[:, 0] / frame_width) - 1.0
        normalized[:, 1] = 2.0 * (keypoints[:, 1] / frame_height) - 1.0
        # Keep confidence as is
        
        return normalized

    def extract_skeleton_features(self, keypoints):
        """
        Extract features from skeleton keypoints.
        
        Args:
            keypoints: ndarray of shape (17, 3) [x, y, confidence]
            
        Returns:
            dict with various skeleton-based features
        """
        features = {}
        
        # Filter valid keypoints (confidence > threshold)
        valid_mask = keypoints[:, 2] > self.keypoint_confidence_threshold
        valid_keypoints = keypoints[valid_mask]
        
        if len(valid_keypoints) == 0:
            # Return zero features if no valid keypoints
            return {
                'valid_keypoint_count': 0,
                'skeleton_area': 0,
                'skeleton_center': np.array([0, 0]),
                'limb_lengths': np.zeros(6),
                'angles': np.zeros(8) if self.use_angles else None,
                'distances': np.zeros(10) if self.use_distances else None,
            }
        
        features['valid_keypoint_count'] = len(valid_keypoints)
        
        # Skeleton bounding box
        x_min, x_max = valid_keypoints[:, 0].min(), valid_keypoints[:, 0].max()
        y_min, y_max = valid_keypoints[:, 1].min(), valid_keypoints[:, 1].max()
        features['skeleton_area'] = (x_max - x_min) * (y_max - y_min)
        features['skeleton_center'] = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        
        # Limb lengths (6 main limbs)
        limb_lengths = self._compute_limb_lengths(keypoints)
        features['limb_lengths'] = limb_lengths
        
        if self.use_angles:
            angles = self._compute_angles(keypoints)
            features['angles'] = angles
        
        if self.use_distances:
            distances = self._compute_key_distances(keypoints)
            features['distances'] = distances
        
        return features

    def _compute_limb_lengths(self, keypoints):
        """
        Compute lengths of main limbs.
        
        Returns:
            array of 6 limb lengths [left_arm, right_arm, left_leg, right_leg, torso_height, width]
        """
        def distance(kp1_idx, kp2_idx):
            kp1 = keypoints[kp1_idx, :2]
            kp2 = keypoints[kp2_idx, :2]
            return np.linalg.norm(kp2 - kp1)

        limbs = [
            distance(5, 9),    # Left arm: shoulder to wrist
            distance(6, 10),   # Right arm: shoulder to wrist
            distance(11, 15),  # Left leg: hip to ankle
            distance(12, 16),  # Right leg: hip to ankle
            distance(5, 11),   # Left torso: shoulder to hip
            distance(6, 12),   # Right torso: shoulder to hip
        ]
        
        return np.array(limbs)

    def _compute_angles(self, keypoints):
        """
        Compute angles at major joints.
        
        Returns:
            array of joint angles in radians
        """
        def angle_between(kp1_idx, kp2_idx, kp3_idx):
            """Compute angle at kp2 formed by kp1-kp2-kp3"""
            v1 = keypoints[kp1_idx, :2] - keypoints[kp2_idx, :2]
            v2 = keypoints[kp3_idx, :2] - keypoints[kp2_idx, :2]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle)

        angles = [
            angle_between(5, 7, 9),    # Left elbow
            angle_between(6, 8, 10),   # Right elbow
            angle_between(11, 13, 15), # Left knee
            angle_between(12, 14, 16), # Right knee
            angle_between(5, 11, 13),  # Left hip
            angle_between(6, 12, 14),  # Right hip
            angle_between(11, 5, 6),   # Left shoulder
            angle_between(12, 6, 5),   # Right shoulder
        ]
        
        return np.array(angles)

    def _compute_key_distances(self, keypoints):
        """
        Compute distances between key body parts.
        
        Returns:
            array of pairwise distances
        """
        def distance(kp1_idx, kp2_idx):
            kp1 = keypoints[kp1_idx, :2]
            kp2 = keypoints[kp2_idx, :2]
            return np.linalg.norm(kp2 - kp1)

        distances = [
            distance(0, 11),   # Nose to left hip
            distance(0, 12),   # Nose to right hip
            distance(5, 6),    # Left shoulder to right shoulder
            distance(11, 12),  # Left hip to right hip
            distance(5, 12),   # Left shoulder to right hip (diagonal)
            distance(6, 11),   # Right shoulder to left hip (diagonal)
            distance(1, 3),    # Left eye to left ear
            distance(2, 4),    # Right eye to right ear
            distance(9, 10),   # Left wrist to right wrist
            distance(15, 16),  # Left ankle to right ankle
        ]
        
        return np.array(distances)

    def draw_skeleton(self, frame, keypoints, skeleton_colors=None):
        """
        Draw skeleton keypoints and connections on frame.
        
        Args:
            frame: ndarray (H, W, 3) BGR image
            keypoints: ndarray of shape (17, 3) [x, y, confidence]
            skeleton_colors: optional dict mapping joint names to BGR colors
            
        Returns:
            annotated frame
        """
        frame = frame.copy()
        
        if skeleton_colors is None:
            skeleton_colors = {}
        
        default_kp_color = (0, 255, 0)  # Green
        default_line_color = (255, 0, 0)  # Blue
        
        # Draw connections
        for start_idx, end_idx in self.SKELETON_CONNECTIONS:
            if (keypoints[start_idx, 2] > self.keypoint_confidence_threshold and
                keypoints[end_idx, 2] > self.keypoint_confidence_threshold):
                
                pt1 = tuple(map(int, keypoints[start_idx, :2]))
                pt2 = tuple(map(int, keypoints[end_idx, :2]))
                cv2.line(frame, pt1, pt2, default_line_color, 2)
        
        # Draw keypoints
        for i, (kp_name, kp) in enumerate(zip(self.KEYPOINT_NAMES, keypoints)):
            if kp[2] > self.keypoint_confidence_threshold:
                pt = tuple(map(int, kp[:2]))
                color = skeleton_colors.get(kp_name, default_kp_color)
                cv2.circle(frame, pt, 4, color, -1)
                cv2.putText(frame, f"{i}", (pt[0] + 5, pt[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame

    def get_skeleton_info(self, keypoints):
        """
        Get human-readable skeleton information.
        
        Returns:
            dict with skeleton statistics
        """
        valid_mask = keypoints[:, 2] > self.keypoint_confidence_threshold
        valid_count = valid_mask.sum()
        
        info = {
            'total_keypoints': len(keypoints),
            'valid_keypoints': int(valid_count),
            'confidence': float(keypoints[:, 2].mean()),
            'keypoint_names': self.KEYPOINT_NAMES
        }
        
        return info
