"""
STSMIRS Action Detector (LSTM)

Runs the sequence model, then applies scene-aware guardrails so dangerous
labels like Fall and Fighting need believable posture or interaction evidence.
"""

import json
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ActionDetector] WARNING: PyTorch not installed. Action inference disabled.")


if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim=8, hidden_dim=256, num_layers=2, num_classes=7):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out)


class ActionDetector:
    """Extracts track features, runs the LSTM, and stabilizes live labels."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.lstm_cfg = config.get("lstm", {})
        self.model_path = self.lstm_cfg.get("model_path", "models/lstm_classifier.pth")
        self.seq_len = self.lstm_cfg.get("sequence_length", 30)
        self.hidden_size = self.lstm_cfg.get("hidden_size", 256)
        self.num_layers = self.lstm_cfg.get("num_layers", 2)
        self.num_classes = self.lstm_cfg.get("num_classes", 7)
        self.feature_dim = self.lstm_cfg.get("feature_dim", 8)
        self.classes = self.lstm_cfg.get(
            "class_names",
            ["Walking", "Running", "Loitering", "Fall", "Lying_Still", "Fighting", "Panic"],
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        post_cfg = config.get("action_postprocess", {})
        self.default_threshold = post_cfg.get("default_threshold", 0.50)
        self.class_thresholds = post_cfg.get("class_thresholds", {})
        self.default_stable_windows = post_cfg.get("default_stable_windows", 1)
        self.class_stable_windows = post_cfg.get("stable_windows", {})
        self.smoothing_alpha = post_cfg.get("smoothing_alpha", 0.35)
        self.safe_fallback_label = post_cfg.get("safe_fallback_label", "Loitering")
        self.transition_release_windows = int(post_cfg.get("transition_release_windows", 2))
        self.idle_reset_windows = int(post_cfg.get("idle_reset_windows", 4))
        self.min_training_samples = int(post_cfg.get("min_training_samples", 1))

        fighting_cfg = post_cfg.get("fighting", {})
        self.fighting_min_person_count = int(fighting_cfg.get("min_person_count", 2))
        self.fighting_min_confidence = float(fighting_cfg.get("min_confidence", 0.72))
        self.fighting_min_speed = float(fighting_cfg.get("min_speed_norm", 0.015))
        self.fighting_max_distance = float(fighting_cfg.get("max_center_distance_norm", 0.18))
        self.fighting_min_iou = float(fighting_cfg.get("min_iou", 0.01))
        self.fighting_min_partner_speed = float(fighting_cfg.get("min_partner_speed_norm", 0.01))

        panic_cfg = post_cfg.get("panic", {})
        self.panic_min_person_count = int(panic_cfg.get("min_person_count", 2))
        self.panic_min_confidence = float(panic_cfg.get("min_confidence", 0.68))
        self.panic_min_speed = float(panic_cfg.get("min_speed_norm", 0.008))
        self.panic_max_distance = float(panic_cfg.get("max_center_distance_norm", 0.35))

        fall_cfg = post_cfg.get("fall", {})
        self.fall_min_confidence = float(fall_cfg.get("min_confidence", 0.86))
        self.fall_baseline_window = max(6, int(fall_cfg.get("baseline_window", 20)))
        self.fall_min_height_ratio_to_baseline = float(
            fall_cfg.get("min_height_ratio_to_baseline", 0.32)
        )
        self.fall_min_aspect_ratio = float(fall_cfg.get("min_aspect_ratio", 0.78))
        self.fall_min_center_drop_norm = float(fall_cfg.get("min_center_drop_norm", 0.10))
        self.fall_min_downward_velocity_norm = float(
            fall_cfg.get("min_downward_velocity_norm", 0.018)
        )
        self.fall_max_horizontal_velocity_norm = float(
            fall_cfg.get("max_horizontal_velocity_norm", 0.060)
        )
        self.fall_min_low_posture_frames = max(
            2, int(fall_cfg.get("min_low_posture_frames", 3))
        )
        self.fall_max_settle_speed_norm = float(
            fall_cfg.get("max_settle_speed_norm", 0.030)
        )

        self._feature_buffers = {}
        self._smoothed_probs = {}
        self._track_states = {}
        self._class_training_counts = self._load_training_metadata()

        self.model = None
        self.device = None
        if TORCH_AVAILABLE:
            self._load_model()

    def _load_training_metadata(self):
        meta_path = os.path.splitext(self.model_path)[0] + "_meta.json"
        if not os.path.exists(meta_path):
            return {}

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            counts = meta.get("class_counts", {})
            if counts:
                print(f"[ActionDetector] Loaded training metadata from {meta_path}")
            return counts
        except Exception as exc:
            print(f"[ActionDetector] WARNING: could not read training metadata: {exc}")
            return {}

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ActionDetector] Loading LSTM model on {self.device}")

        self.model = LSTMClassifier(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
        ).to(self.device)

        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                try:
                    self.model.load_state_dict(state_dict)
                except RuntimeError:
                    current_state = self.model.state_dict()
                    compatible_state = {
                        key: value
                        for key, value in state_dict.items()
                        if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
                    }
                    current_state.update(compatible_state)
                    self.model.load_state_dict(current_state)
                    print(
                        f"[ActionDetector] Partial weight load from {self.model_path} "
                        f"({len(compatible_state)}/{len(current_state)} tensors matched)"
                    )
                self.model.eval()
                print(f"[ActionDetector] Loaded weights from {self.model_path}")
            except Exception as exc:
                print(f"[ActionDetector] WARNING: Error loading weights (using untrained model): {exc}")
                self.model.eval()
        else:
            print(f"[ActionDetector] WARNING: Model file not found at {self.model_path}. Using uninitialized weights.")
            self.model.eval()

    def extract_features(self, person, frame_w, frame_h, tracker):
        x1, y1, x2, y2 = person.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = person.width
        h = person.height

        cx_norm = cx / frame_w
        cy_norm = cy / frame_h
        w_norm = w / frame_w
        h_norm = h / frame_h
        aspect_ratio = w / float(h) if h > 0 else 0.0
        area_norm = (w * h) / float(frame_w * frame_h)

        vel_x, vel_y = tracker.get_velocity(person.track_id)
        vel_x_norm = vel_x / frame_w
        vel_y_norm = vel_y / frame_h

        return np.array(
            [cx_norm, cy_norm, w_norm, h_norm, aspect_ratio, vel_x_norm, vel_y_norm, area_norm],
            dtype=np.float32,
        )

    def update(self, persons, tracker, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        active_ids = {p.track_id for p in persons}
        detections = []
        raw_predictions = {}

        for person in persons:
            tid = person.track_id
            feat = self.extract_features(person, frame_w, frame_h, tracker)
            self._update_track_metrics(tid, person, frame_w, frame_h, tracker)

            self._feature_buffers.setdefault(tid, []).append(feat)
            if len(self._feature_buffers[tid]) > self.seq_len:
                self._feature_buffers[tid] = self._feature_buffers[tid][-self.seq_len:]

            if len(self._feature_buffers[tid]) == self.seq_len:
                raw_predictions[tid] = self._run_inference(self._feature_buffers[tid])

        for person in persons:
            tid = person.track_id
            if tid not in raw_predictions:
                continue

            pred_class, conf, raw_probs = raw_predictions[tid]
            if pred_class is None:
                continue

            final_class, final_conf, should_emit = self._postprocess_prediction(
                person, persons, tracker, frame_w, frame_h, pred_class, conf, raw_probs, raw_predictions
            )
            person.event_type = final_class
            person.action_confidence = float(final_conf)

            if should_emit:
                detections.append(
                    {"person": person, "event_type": final_class, "confidence": final_conf}
                )

        self._feature_buffers = {k: v for k, v in self._feature_buffers.items() if k in active_ids}
        self._smoothed_probs = {k: v for k, v in self._smoothed_probs.items() if k in active_ids}
        self._track_states = {k: v for k, v in self._track_states.items() if k in active_ids}
        return detections

    def _run_inference(self, sequence):
        if not TORCH_AVAILABLE or self.model is None:
            probs = np.zeros(len(self.classes), dtype=np.float32)
            probs[self.class_to_idx.get("Walking", 0)] = 1.0
            return "Walking", 1.0, probs

        seq_array = np.asarray(sequence, dtype=np.float32)
        seq_tensor = torch.from_numpy(seq_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        return self.classes[pred_idx], confidence, probs

    def _postprocess_prediction(
        self, person, persons, tracker, frame_w, frame_h, pred_class, conf, raw_probs, raw_predictions
    ):
        tid = person.track_id
        smoothed = self._smoothed_probs.get(tid)
        if smoothed is None:
            smoothed = raw_probs.astype(np.float32)
        else:
            smoothed = ((1.0 - self.smoothing_alpha) * smoothed) + (self.smoothing_alpha * raw_probs)
        smoothed = smoothed.astype(np.float32)
        smoothed /= max(float(smoothed.sum()), 1e-6)
        self._smoothed_probs[tid] = smoothed

        adjusted = smoothed.copy()
        adjusted = self._suppress_untrained_classes(adjusted)
        adjusted = self._apply_group_guardrails(
            adjusted, person, persons, tracker, frame_w, frame_h, raw_predictions
        )

        pred_idx = int(np.argmax(adjusted))
        pred_class = self.classes[pred_idx]
        confidence = float(adjusted[pred_idx])

        state = self._ensure_track_state(tid)
        if state["candidate"] == pred_class:
            state["streak"] += 1
        else:
            state["candidate"] = pred_class
            state["streak"] = 1

        threshold = float(self.class_thresholds.get(pred_class, self.default_threshold))
        required_windows = int(self.class_stable_windows.get(pred_class, self.default_stable_windows))

        candidate_ready = confidence >= threshold and state["streak"] >= required_windows
        if candidate_ready:
            state["stable_label"] = pred_class
            state["stable_confidence"] = confidence
            state["miss_streak"] = 0
        elif state["stable_label"] is None:
            fallback_idx = self.class_to_idx.get(self.safe_fallback_label, 0)
            state["stable_label"] = self.classes[fallback_idx]
            state["stable_confidence"] = float(adjusted[fallback_idx])
            state["miss_streak"] = 0
        else:
            if pred_class != state["stable_label"]:
                state["miss_streak"] += 1
            else:
                state["miss_streak"] = 0

            if state["miss_streak"] >= self.transition_release_windows:
                if confidence >= max(self.default_threshold, threshold * 0.9):
                    state["stable_label"] = pred_class
                    state["stable_confidence"] = confidence
                    state["miss_streak"] = 0
                elif state["miss_streak"] >= self.idle_reset_windows:
                    fallback_idx = self.class_to_idx.get(self.safe_fallback_label, 0)
                    state["stable_label"] = self.classes[fallback_idx]
                    state["stable_confidence"] = float(adjusted[fallback_idx])

        stable_label = state["stable_label"]
        stable_conf = float(state["stable_confidence"])
        person.action_probabilities = {
            self.classes[idx]: float(adjusted[idx]) for idx in range(len(self.classes))
        }
        should_emit = (
            stable_label in {"Fall", "Fighting", "Panic"}
            and pred_class == stable_label
            and confidence >= threshold
            and state["streak"] >= required_windows
        )
        return stable_label, stable_conf, should_emit

    def _ensure_track_state(self, tid):
        return self._track_states.setdefault(
            tid,
            {
                "candidate": None,
                "streak": 0,
                "stable_label": None,
                "stable_confidence": 0.0,
                "miss_streak": 0,
                "height_history": [],
                "width_history": [],
                "aspect_history": [],
                "cy_history": [],
                "speed_history": [],
                "vx_history": [],
                "vy_history": [],
            },
        )

    def _update_track_metrics(self, tid, person, frame_w, frame_h, tracker):
        state = self._ensure_track_state(tid)
        vx, vy = tracker.get_velocity(tid)
        vx_norm = float(vx / max(frame_w, 1))
        vy_norm = float(vy / max(frame_h, 1))
        speed_norm = float(np.hypot(vx_norm, vy_norm))
        width_norm = float(person.width / max(frame_w, 1))
        height_norm = float(person.height / max(frame_h, 1))
        cy_norm = float(person.center[1] / max(frame_h, 1))
        aspect_ratio = float(person.width / max(float(person.height), 1.0))

        max_len = max(self.seq_len * 2, self.fall_baseline_window + 10)
        history_specs = [
            ("height_history", height_norm),
            ("width_history", width_norm),
            ("aspect_history", aspect_ratio),
            ("cy_history", cy_norm),
            ("speed_history", speed_norm),
            ("vx_history", vx_norm),
            ("vy_history", vy_norm),
        ]

        for key, value in history_specs:
            state[key].append(float(value))
            if len(state[key]) > max_len:
                state[key] = state[key][-max_len:]

    def _apply_group_guardrails(self, probs, person, persons, tracker, frame_w, frame_h, raw_predictions):
        probs = self._apply_fall_guardrails(probs, person)
        probs = self._apply_fighting_guardrails(probs, person, persons, tracker, frame_w, frame_h)
        probs = self._apply_panic_guardrails(probs, person, persons, tracker, frame_w, frame_h, raw_predictions)
        return probs

    def _suppress_untrained_classes(self, probs):
        if not self._class_training_counts:
            return probs / max(float(probs.sum()), 1e-6)

        adjusted = probs.copy()
        changed = False
        for class_name, idx in self.class_to_idx.items():
            count = int(self._class_training_counts.get(class_name, 0))
            if count < self.min_training_samples:
                adjusted[idx] = 0.0
                changed = True

        if not changed:
            return adjusted / max(float(adjusted.sum()), 1e-6)
        return self._renormalize_with_safe_fallback(adjusted)

    def _apply_fall_guardrails(self, probs, person):
        fall_idx = self.class_to_idx.get("Fall")
        if fall_idx is None:
            return probs

        if probs[fall_idx] < self.fall_min_confidence:
            probs[fall_idx] = 0.0
            return self._renormalize_with_safe_fallback(probs)

        evidence = self._compute_fall_evidence(person)
        if not evidence["valid"]:
            probs[fall_idx] = 0.0
            return self._renormalize_with_safe_fallback(probs)

        return probs / max(float(probs.sum()), 1e-6)

    def _apply_fighting_guardrails(self, probs, person, persons, tracker, frame_w, frame_h):
        fighting_idx = self.class_to_idx.get("Fighting")
        if fighting_idx is None:
            return probs

        if probs[fighting_idx] < self.fighting_min_confidence:
            probs[fighting_idx] = 0.0
            return self._renormalize_with_safe_fallback(probs)

        evidence = self._compute_fighting_evidence(person, persons, tracker, frame_w, frame_h)
        if not evidence["valid"]:
            probs[fighting_idx] = 0.0
            return self._renormalize_with_safe_fallback(probs)

        return probs / max(float(probs.sum()), 1e-6)

    def _apply_panic_guardrails(self, probs, person, persons, tracker, frame_w, frame_h, raw_predictions):
        panic_idx = self.class_to_idx.get("Panic")
        if panic_idx is None:
            return probs

        if probs[panic_idx] < self.panic_min_confidence:
            probs[panic_idx] = 0.0
            return self._renormalize_with_safe_fallback(probs)

        evidence = self._compute_panic_evidence(person, persons, tracker, frame_w, frame_h, raw_predictions)
        if not evidence["valid"]:
            probs[panic_idx] = 0.0
            return self._renormalize_with_safe_fallback(probs)

        return probs / max(float(probs.sum()), 1e-6)

    def _compute_fall_evidence(self, person):
        state = self._track_states.get(person.track_id)
        if not state:
            return {"valid": False}

        height_history = state.get("height_history", [])
        aspect_history = state.get("aspect_history", [])
        cy_history = state.get("cy_history", [])
        speed_history = state.get("speed_history", [])
        vx_history = state.get("vx_history", [])
        vy_history = state.get("vy_history", [])

        if len(height_history) < self.fall_baseline_window:
            return {"valid": False}

        baseline_heights = height_history[:-self.fall_min_low_posture_frames] or height_history
        baseline_cy = cy_history[:-self.fall_min_low_posture_frames] or cy_history
        baseline_height = float(np.median(baseline_heights[-self.fall_baseline_window:]))
        baseline_center_y = float(np.median(baseline_cy[-self.fall_baseline_window:]))
        if baseline_height <= 1e-6:
            return {"valid": False}

        current_height = float(height_history[-1])
        current_aspect = float(aspect_history[-1])
        current_center_y = float(cy_history[-1])
        current_vx = float(vx_history[-1]) if vx_history else 0.0
        current_vy = float(vy_history[-1]) if vy_history else 0.0

        height_drop_ratio = max(0.0, (baseline_height - current_height) / baseline_height)
        center_drop = max(0.0, current_center_y - baseline_center_y)
        recent_aspects = aspect_history[-self.fall_min_low_posture_frames:]
        low_posture_frames = sum(1 for value in recent_aspects if value >= self.fall_min_aspect_ratio)
        recent_speeds = speed_history[-self.fall_min_low_posture_frames:]
        settled = bool(recent_speeds) and all(
            value <= self.fall_max_settle_speed_norm for value in recent_speeds
        )
        strong_collapse = (
            height_drop_ratio >= self.fall_min_height_ratio_to_baseline
            and current_aspect >= self.fall_min_aspect_ratio
        )
        motion_support = (
            center_drop >= self.fall_min_center_drop_norm
            or current_vy >= self.fall_min_downward_velocity_norm
        )
        limited_sideways_motion = abs(current_vx) <= self.fall_max_horizontal_velocity_norm
        valid = strong_collapse and motion_support and limited_sideways_motion and (
            low_posture_frames >= self.fall_min_low_posture_frames or settled
        )

        return {
            "valid": valid,
            "height_drop_ratio": height_drop_ratio,
            "center_drop": center_drop,
            "aspect_ratio": current_aspect,
            "low_posture_frames": low_posture_frames,
            "settled": settled,
        }

    def _compute_fighting_evidence(self, person, persons, tracker, frame_w, frame_h):
        if len(persons) < self.fighting_min_person_count:
            return {"valid": False}

        px, py = person.center
        self_speed = self._speed_norm(tracker.get_velocity(person.track_id), frame_w, frame_h)
        self_vx, self_vy = tracker.get_velocity(person.track_id)
        best = {
            "valid": False,
            "distance": 999.0,
            "iou": 0.0,
            "partner_speed": 0.0,
            "horizontal_overlap": 0.0,
            "approach_score": 0.0,
        }

        for other in persons:
            if other.track_id == person.track_id:
                continue

            ox, oy = other.center
            distance = float(np.hypot(px - ox, py - oy))
            distance_norm = distance / max(float(np.hypot(frame_w, frame_h)), 1.0)
            iou = self._bbox_iou(person.bbox, other.bbox)
            horizontal_overlap = self._axis_overlap_ratio(
                person.bbox[0], person.bbox[2], other.bbox[0], other.bbox[2]
            )
            partner_speed = self._speed_norm(tracker.get_velocity(other.track_id), frame_w, frame_h)
            other_vx, other_vy = tracker.get_velocity(other.track_id)
            approach_score = self._approach_score(
                person.center, other.center, (self_vx, self_vy), (other_vx, other_vy)
            )

            if distance_norm < best["distance"]:
                best = {
                    "valid": (
                        distance_norm <= self.fighting_max_distance
                        and (iou >= self.fighting_min_iou or horizontal_overlap >= 0.28)
                        and self_speed >= self.fighting_min_speed
                        and partner_speed >= self.fighting_min_partner_speed
                        and approach_score >= 0.10
                    ),
                    "distance": distance_norm,
                    "iou": iou,
                    "partner_speed": partner_speed,
                    "horizontal_overlap": horizontal_overlap,
                    "approach_score": approach_score,
                }

        return best

    def _compute_panic_evidence(self, person, persons, tracker, frame_w, frame_h, raw_predictions):
        if len(persons) < self.panic_min_person_count:
            return {"valid": False}

        panic_idx = self.class_to_idx.get("Panic")
        if panic_idx is None:
            return {"valid": False}

        self_speed = self._speed_norm(tracker.get_velocity(person.track_id), frame_w, frame_h)
        if self_speed < self.panic_min_speed:
            return {"valid": False}

        px, py = person.center
        best_distance = None

        for other in persons:
            if other.track_id == person.track_id:
                continue

            other_pred = raw_predictions.get(other.track_id)
            if other_pred is None:
                continue

            _, _, other_probs = other_pred
            other_conf = float(other_probs[panic_idx]) if panic_idx < len(other_probs) else 0.0
            if other_conf < self.panic_min_confidence:
                continue

            partner_speed = self._speed_norm(tracker.get_velocity(other.track_id), frame_w, frame_h)
            if partner_speed < self.panic_min_speed:
                continue

            ox, oy = other.center
            distance_norm = float(np.hypot(px - ox, py - oy)) / max(float(np.hypot(frame_w, frame_h)), 1.0)
            best_distance = distance_norm if best_distance is None else min(best_distance, distance_norm)
            if distance_norm <= self.panic_max_distance:
                return {"valid": True, "distance": distance_norm, "partner_speed": partner_speed}

        return {"valid": False, "distance": best_distance}

    @staticmethod
    def _speed_norm(velocity, frame_w, frame_h):
        vx, vy = velocity
        return float(np.hypot(vx / max(frame_w, 1), vy / max(frame_h, 1)))

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    @staticmethod
    def _axis_overlap_ratio(a1, a2, b1, b2):
        overlap = max(0.0, min(a2, b2) - max(a1, b1))
        denom = max(1.0, min(a2 - a1, b2 - b1))
        return float(overlap / denom)

    @staticmethod
    def _approach_score(center_a, center_b, vel_a, vel_b):
        ax, ay = center_a
        bx, by = center_b
        vx_a, vy_a = vel_a
        vx_b, vy_b = vel_b
        vec_ab = np.array([bx - ax, by - ay], dtype=np.float32)
        norm = float(np.linalg.norm(vec_ab))
        if norm <= 1e-6:
            return 0.0
        direction_ab = vec_ab / norm
        direction_ba = -direction_ab
        approach_a = float(np.dot(np.array([vx_a, vy_a], dtype=np.float32), direction_ab))
        approach_b = float(np.dot(np.array([vx_b, vy_b], dtype=np.float32), direction_ba))
        return max(0.0, approach_a + approach_b) / max(norm, 1.0)

    def _renormalize_with_safe_fallback(self, probs):
        safe_probs = probs.copy()
        if safe_probs.sum() <= 0:
            fallback_idx = self.class_to_idx.get(self.safe_fallback_label, 0)
            safe_probs[fallback_idx] = 1.0
        return safe_probs / max(float(safe_probs.sum()), 1e-6)
