"""
STSMIRS — Panic Detector (Heuristic-Based)
Detects panic situations without training data by analyzing crowd velocity and directional divergence.
"""

import numpy as np
import time

class PanicDetector:
    def __init__(self, speed_threshold=100.0, min_people=2, divergence_threshold=0.5, persistence_frames=5):
        """
        Args:
            speed_threshold: Min pixels/sec to be considered 'running'.
            min_people: Min number of runners to trigger panic.
            divergence_threshold: 0.0 (same direction) to 1.0 (perfectly scattered).
            persistence_frames: How many consecutive frames the condition must be met.
        """
        self.speed_threshold = speed_threshold
        self.min_people = min_people
        self.divergence_threshold = divergence_threshold
        self.persistence_frames = persistence_frames
        
        self.persistence_counter = 0
        self.is_panic = False
        
        # Debugging info
        self.last_metrics = {
            "runner_count": 0,
            "avg_speed": 0.0,
            "divergence": 0.0
        }

    def update(self, persons, tracker):
        """
        Analyze current frame for panic behavior.
        Handles both:
        1. Scattered Panic (running in different directions)
        2. Surge Panic (running fast in the same direction)
        """
        runners_v = []
        
        for person in persons:
            vx, vy = tracker.get_velocity(person.track_id)
            speed = np.sqrt(vx**2 + vy**2)
            
            if speed > self.speed_threshold:
                runners_v.append(np.array([vx, vy]))
        
        num_runners = len(runners_v)
        divergence = 0.0
        avg_speed = 0.0
        condition_met = False
        
        if num_runners >= self.min_people:
            # Calculate scattering (divergence)
            sum_v = np.sum(runners_v, axis=0)
            mag_sum = np.linalg.norm(sum_v)
            sum_mags = np.sum([np.linalg.norm(v) for v in runners_v])
            
            avg_speed = sum_mags / num_runners
            
            if sum_mags > 0:
                divergence = 1.0 - (mag_sum / sum_mags)
            
            # TRIGGER LOGIC:
            # Case A: Scattered (high divergence)
            # Case B: Surge (low divergence but high speed)
            if divergence > self.divergence_threshold:
                condition_met = True
            elif avg_speed > (self.speed_threshold * 1.5): # Surge needs higher speed to avoid jogging false positives
                condition_met = True
                
            if condition_met:
                self.persistence_counter += 1
            else:
                self.persistence_counter = max(0, self.persistence_counter - 1)
        else:
            self.persistence_counter = max(0, self.persistence_counter - 1)
            
        # Update panic state
        if self.persistence_counter >= self.persistence_frames:
            self.is_panic = True
        elif self.persistence_counter == 0:
            self.is_panic = False
            
        self.last_metrics = {
            "runner_count": num_runners,
            "avg_speed": avg_speed,
            "divergence": divergence,
            "persistence": self.persistence_counter,
            "panic_type": "Scattered" if divergence > self.divergence_threshold else "Surge"
        }
        
        return self.is_panic, self.last_metrics

    def reset(self):
        self.persistence_counter = 0
        self.is_panic = False
