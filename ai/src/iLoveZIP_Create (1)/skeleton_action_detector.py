"""
STSMIRS — Skeleton-Based Action Detector
Uses skeleton pose keypoints and deep learning for improved action recognition.
Complements the existing LSTM-based action detector with skeleton features.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class SkeletonLSTMClassifier(nn.Module):
    """
    LSTM-based classifier for skeleton action recognition.
    Input: sequence of normalized skeleton keypoints
    Output: Action probability distribution
    """

    def __init__(self, input_dim=51, hidden_dim=256, num_layers=2, num_classes=7, dropout=0.3):
        """
        Args:
            input_dim: Input feature dimension (17 keypoints * 3 coordinates = 51)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of action classes
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Skeleton feature preprocessing
        self.fc_in = nn.Linear(input_dim, hidden_dim // 2)
        self.bn_in = nn.BatchNorm1d(hidden_dim // 2)

        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim // 2,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Reshape for batch normalization
        x_flat = x.reshape(batch_size * seq_len, self.input_dim)
        x_feat = self.fc_in(x_flat)
        x_feat = self.bn_in(x_feat)
        x_feat = x_feat.reshape(batch_size, seq_len, -1)

        # LSTM forward
        lstm_out, _ = self.lstm(x_feat)
        last_out = lstm_out[:, -1, :]

        # Classification
        logits = self.fc_out(self.dropout(last_out))
        return logits


class SkeletonActionDetector:
    """
    Detects actions using skeleton pose sequences.
    Maintains skeleton buffers per tracked person and runs inference.
    """

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load skeleton action config
        self.skeleton_action_cfg = config.get("skeleton_action", {})
        self.model_path = self.skeleton_action_cfg.get("model_path", "models/skeleton_lstm.pth")
        self.seq_len = self.skeleton_action_cfg.get("sequence_length", 30)
        self.hidden_size = self.skeleton_action_cfg.get("hidden_size", 256)
        self.num_layers = self.skeleton_action_cfg.get("num_layers", 2)
        self.num_classes = self.skeleton_action_cfg.get("num_classes", 7)
        self.input_dim = self.skeleton_action_cfg.get("input_dim", 51)  # 17 keypoints * 3
        self.dropout = self.skeleton_action_cfg.get("dropout", 0.3)
        self.device = self.skeleton_action_cfg.get("device", "auto")
        
        self.classes = self.skeleton_action_cfg.get(
            "class_names",
            ["Walking", "Running", "Loitering", "Fall", "Lying_Still", "Fighting", "Panic"]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Postprocessing config
        post_cfg = self.skeleton_action_cfg.get("postprocess", {})
        self.confidence_threshold = post_cfg.get("confidence_threshold", 0.5)
        self.use_smoothing = post_cfg.get("use_smoothing", True)
        self.smoothing_alpha = post_cfg.get("smoothing_alpha", 0.4)
        
        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.model = SkeletonLSTMClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout
        )
        self.model = self.model.to(self.device)
        
        # Per-track skeleton buffers: track_id -> deque of normalized keypoints
        self._skeleton_buffers = {}
        
        # Smoothed predictions: track_id -> last smooth prediction
        self._smooth_predictions = {}
        
        self._load_model()

    def _load_model(self):
        """Load the skeleton-based LSTM model."""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.eval()
                print(f"[SkeletonActionDetector] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[SkeletonActionDetector] Error loading model: {e}")
        else:
            print(f"[SkeletonActionDetector] Model not found at {self.model_path}. Using untrained model.")

    def update_skeleton(self, track_id, normalized_keypoints):
        """
        Add normalized skeleton keypoints to the buffer for a tracked person.
        
        Args:
            track_id: int, unique person ID
            normalized_keypoints: ndarray of shape (17, 3) [x, y, confidence]
        """
        if track_id not in self._skeleton_buffers:
            self._skeleton_buffers[track_id] = deque(maxlen=self.seq_len)
        
        # Flatten keypoints to (51,) and add to buffer
        flattened = normalized_keypoints.flatten()
        self._skeleton_buffers[track_id].append(flattened)

    def predict(self, track_id):
        """
        Run inference on skeleton sequence for a person.
        
        Returns:
            dict with predictions, confidence, and class name
        """
        if track_id not in self._skeleton_buffers:
            return {
                'action': 'Unknown',
                'confidence': 0.0,
                'probabilities': {},
                'valid': False
            }
        
        buffer = self._skeleton_buffers[track_id]
        
        # Need at least seq_len frames
        if len(buffer) < self.seq_len:
            return {
                'action': 'Unknown',
                'confidence': 0.0,
                'probabilities': {},
                'valid': False
            }
        
        # Prepare batch
        sequence = np.array(list(buffer))  # (seq_len, input_dim)
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq_len, input_dim)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        
        probs = probs.cpu().numpy()[0]
        pred_class_idx = np.argmax(probs)
        pred_class = self.classes[pred_class_idx]
        confidence = float(probs[pred_class_idx])
        
        # Apply smoothing if enabled
        if self.use_smoothing and track_id in self._smooth_predictions:
            old_pred, old_conf = self._smooth_predictions[track_id]
            if old_pred == pred_class:
                confidence = self.smoothing_alpha * confidence + (1 - self.smoothing_alpha) * old_conf
        
        self._smooth_predictions[track_id] = (pred_class, confidence)
        
        # Build probability dict
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(self.num_classes)}
        
        return {
            'action': pred_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'valid': confidence >= self.confidence_threshold
        }

    def predict_batch(self, track_ids):
        """
        Run inference on multiple persons.
        
        Args:
            track_ids: list of track IDs
        
        Returns:
            dict mapping track_id -> prediction result
        """
        return {track_id: self.predict(track_id) for track_id in track_ids}

    def remove_track(self, track_id):
        """Clean up buffers for removed track."""
        if track_id in self._skeleton_buffers:
            del self._skeleton_buffers[track_id]
        if track_id in self._smooth_predictions:
            del self._smooth_predictions[track_id]

    def get_buffer_status(self, track_id):
        """Get status of skeleton buffer for debugging."""
        if track_id not in self._skeleton_buffers:
            return {'status': 'not_found', 'frames': 0}
        
        buffer = self._skeleton_buffers[track_id]
        return {
            'status': 'ready' if len(buffer) >= self.seq_len else 'accumulating',
            'frames': len(buffer),
            'required_frames': self.seq_len
        }

    def reset_all_buffers(self):
        """Clear all skeleton buffers and predictions."""
        self._skeleton_buffers.clear()
        self._smooth_predictions.clear()
