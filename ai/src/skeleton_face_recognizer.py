"""
STSMIRS — Enhanced Face Recognizer with Skeleton Context
Combines face recognition with body pose for improved accuracy and context-awareness.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path


class SkeletonAwareFaceRecognizer:
    """
    Enhanced face recognizer that uses skeleton pose data for contextual recognition.
    Provides additional signals for person identification when face recognition is ambiguous.
    """

    def __init__(self, face_recognizer, skeleton_extractor, config_path="config.json"):
        """
        Args:
            face_recognizer: FaceRecognizer instance
            skeleton_extractor: SkeletonExtractor instance
            config_path: Path to config.json
        """
        self.face_recognizer = face_recognizer
        self.skeleton_extractor = skeleton_extractor

        with open(config_path, "r") as f:
            config = json.load(f)

        self.skeleton_face_cfg = config.get("skeleton_face_recognition", {})
        self.use_skeleton_context = self.skeleton_face_cfg.get("use_skeleton_context", True)
        self.skeleton_confidence_weight = self.skeleton_face_cfg.get("skeleton_confidence_weight", 0.3)
        self.pose_similarity_threshold = self.skeleton_face_cfg.get("pose_similarity_threshold", 0.7)
        
        # Skeleton-based person profiles: store skeleton patterns per person
        self._skeleton_profiles = {}

    def recognize_person_enhanced(self, frame, bounding_box):
        """
        Recognize person using both face and skeleton context.
        
        Args:
            frame: ndarray BGR image
            bounding_box: (x1, y1, x2, y2) tuple
        
        Returns:
            dict with recognition result including face and skeleton confidence
        """
        # Traditional face recognition
        face_result = self.face_recognizer.recognize_person(frame, bounding_box)

        if not self.use_skeleton_context:
            return face_result

        # Extract skeleton and get contextual information
        skeleton_result = self._extract_and_analyze_skeleton(frame, bounding_box)

        # Fuse results
        enhanced_result = self._fuse_recognition_results(face_result, skeleton_result)

        return enhanced_result

    def _extract_and_analyze_skeleton(self, frame, bounding_box):
        """
        Extract and analyze skeleton within bounding box.
        
        Returns:
            dict with skeleton analysis
        """
        x1, y1, x2, y2 = bounding_box
        
        # Extract skeleton for the whole frame (will be filtered by bbox)
        result = self.skeleton_extractor.extract_skeleton(frame)
        keypoints = result['keypoints']

        if len(keypoints) == 0:
            return {
                'valid': False,
                'pose_features': None,
                'confidence': 0.0
            }

        # Get first person (ideally should match with tracked person)
        person_keypoints = keypoints[0]

        # Extract skeleton features
        features = self.skeleton_extractor.extract_skeleton_features(person_keypoints)

        # Compute pose signature (normalized features for matching)
        pose_signature = self._compute_pose_signature(person_keypoints, features)

        return {
            'valid': True,
            'pose_features': features,
            'pose_signature': pose_signature,
            'confidence': float(person_keypoints[:, 2].mean())
        }

    def _compute_pose_signature(self, keypoints, features):
        """
        Compute a normalized pose signature for person matching.
        
        Returns:
            ndarray with pose signature
        """
        signature = []

        # Add normalized limb lengths
        if 'limb_lengths' in features:
            limb_lengths = features['limb_lengths']
            max_len = np.max(limb_lengths) + 1e-6
            signature.extend(limb_lengths / max_len)

        # Add angles (normalized)
        if 'angles' in features and features['angles'] is not None:
            signature.extend(features['angles'] / np.pi)

        # Add distances (normalized)
        if 'distances' in features and features['distances'] is not None:
            distances = features['distances']
            max_dist = np.max(distances) + 1e-6
            signature.extend(distances / max_dist)

        return np.array(signature, dtype=np.float32) if signature else np.array([], dtype=np.float32)

    def _fuse_recognition_results(self, face_result, skeleton_result):
        """
        Fuse face recognition and skeleton context results.
        
        Returns:
            Enhanced recognition result
        """
        fused = face_result.copy()

        # Adjust confidence based on skeleton context
        if skeleton_result['valid']:
            face_conf = fused.get('confidence', 0.0)
            skeleton_conf = skeleton_result['confidence']
            
            # Weighted fusion
            fused['face_confidence'] = face_conf
            fused['skeleton_confidence'] = skeleton_conf
            fused['confidence'] = (
                face_conf * (1 - self.skeleton_confidence_weight) +
                skeleton_conf * self.skeleton_confidence_weight
            )

            # Add skeleton features for debugging/analysis
            fused['skeleton_features'] = skeleton_result['pose_features']
        else:
            fused['skeleton_confidence'] = 0.0

        return fused

    def update_skeleton_profile(self, person_id, pose_signature):
        """
        Update skeleton profile for a person for future recognition.
        
        Args:
            person_id: str, person identifier
            pose_signature: ndarray, pose signature
        """
        if person_id not in self._skeleton_profiles:
            self._skeleton_profiles[person_id] = []

        self._skeleton_profiles[person_id].append(pose_signature)

        # Keep only recent profiles (sliding window)
        max_profiles = 10
        if len(self._skeleton_profiles[person_id]) > max_profiles:
            self._skeleton_profiles[person_id] = self._skeleton_profiles[person_id][-max_profiles:]

    def match_pose_signature(self, pose_signature, person_id=None):
        """
        Match pose signature against known profiles.
        
        Args:
            pose_signature: ndarray, query pose signature
            person_id: optional, check against specific person
        
        Returns:
            dict with match result
        """
        if not self._skeleton_profiles:
            return {'match': None, 'confidence': 0.0}

        if pose_signature.size == 0:
            return {'match': None, 'confidence': 0.0}

        best_match = None
        best_score = 0.0

        persons_to_check = [person_id] if person_id else self._skeleton_profiles.keys()

        for p_id in persons_to_check:
            if p_id not in self._skeleton_profiles:
                continue

            profiles = self._skeleton_profiles[p_id]
            scores = []

            for profile in profiles:
                # Ensure same dimension
                if len(profile) != len(pose_signature):
                    continue

                # Cosine similarity
                score = self._cosine_similarity(pose_signature, profile)
                scores.append(score)

            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = p_id

        return {
            'match': best_match,
            'confidence': float(best_score) if best_score >= self.pose_similarity_threshold else 0.0
        }

    @staticmethod
    def _cosine_similarity(v1, v2):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)

    def get_person_skeleton_stats(self, person_id):
        """Get skeleton statistics for a person."""
        if person_id not in self._skeleton_profiles or not self._skeleton_profiles[person_id]:
            return None

        profiles = np.array(self._skeleton_profiles[person_id])
        
        return {
            'mean': profiles.mean(axis=0),
            'std': profiles.std(axis=0),
            'num_samples': len(profiles)
        }

    def visualize_skeleton_profiles(self, frame, person_ids):
        """
        Create visualization of skeleton profiles for multiple people.
        
        Args:
            frame: ndarray BGR image
            person_ids: list of person IDs to visualize
        
        Returns:
            annotated frame
        """
        frame = frame.copy()
        
        for i, person_id in enumerate(person_ids):
            stats = self.get_person_skeleton_stats(person_id)
            if stats is None:
                continue

            y_pos = 30 + i * 30
            text = f"{person_id}: samples={stats['num_samples']}"
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame
