"""
Session ML Evaluation: Action Detection & Event Triggers
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.action_detector import ActionDetector, TORCH_AVAILABLE
from src.event_trigger import EventTriggerEngine
from src.scoring_engine import ScoringEngine

class MockPerson:
    def __init__(self, tid):
        self.track_id = tid
        self.track_label = f"T-{tid:03d}"
        self.bbox = [100, 100, 200, 300]
        self.width = 100
        self.height = 200
        self.center = (150, 200)
        self.event_type = None
        self.zone_id = "ZONE-B"
        self.is_alert = False

class MockTracker:
    def get_velocity(self, tid):
        return (5.0, -2.0)

class FastTracker:
    def __init__(self, velocities=None):
        self.velocities = velocities or {}

    def get_velocity(self, tid):
        return self.velocities.get(tid, (25.0, 0.0))

def test():
    print("==================================================")
    print("  STSMIRS — Session ML Tests (LSTM & Triggers)")
    print("==================================================")

    print(f"PyTorch Available: {TORCH_AVAILABLE}")

    if not os.path.exists("config.json"):
        print("Skipping test: config.json missing.")
        return

    print("\n1. Testing ActionDetector Initialization...")
    action_detector = ActionDetector("config.json")
    print(f"  ✓ Initialized. Sequence length: {action_detector.seq_len}")

    print("\n2. Testing Feature Extraction & Inference Buffer...")
    tracker = MockTracker()
    p1 = MockPerson(1)
    frame_shape = (480, 640, 3)
    
    # Run 29 frames: buffer should not trigger inference yet
    for _ in range(29):
        dets = action_detector.update([p1], tracker, frame_shape)
    
    print(f"  ✓ Sent 29 frames. Detections: {len(dets)} (expected 0)")
    
    # Run frame 30: should trigger inference
    dets = action_detector.update([p1], tracker, frame_shape)
    print(f"  ✓ Sent frame 30. Detections returned: {len(dets)} (expected 0 or more depending on stabilization)")
    if dets:
        print(f"    Predicted action: {dets[0]['event_type']} (Conf: {dets[0]['confidence']:.2f})")

    print("\n2b. Testing Fighting Guardrails (single person should be suppressed)...")
    guarded_detector = ActionDetector("config.json")
    fighting_idx = guarded_detector.class_to_idx["Fighting"]
    loitering_idx = guarded_detector.class_to_idx["Loitering"]

    def fake_fighting(_sequence):
        probs = np.zeros(len(guarded_detector.classes), dtype=np.float32)
        probs[fighting_idx] = 0.92
        probs[loitering_idx] = 0.08
        return "Fighting", 0.92, probs

    guarded_detector._run_inference = fake_fighting
    solo_person = MockPerson(11)
    solo_person.bbox = [120, 120, 240, 360]
    solo_person.width = 120
    solo_person.height = 240
    fast_tracker = FastTracker({11: (40.0, 0.0)})

    solo_dets = []
    for _ in range(34):
        solo_dets = guarded_detector.update([solo_person], fast_tracker, frame_shape)

    print(f"  ✓ Single-person fighting detections: {len(solo_dets)} (expected 0)")
    print(f"    Stable label after suppression: {solo_person.event_type}")
    assert len(solo_dets) == 0
    assert solo_person.event_type != "Fighting"

    print("\n2c. Testing Fighting Guardrails (two nearby moving people can still trigger)...")
    interaction_detector = ActionDetector("config.json")

    def fake_pair_fighting(_sequence):
        probs = np.zeros(len(interaction_detector.classes), dtype=np.float32)
        probs[interaction_detector.class_to_idx["Fighting"]] = 0.93
        probs[interaction_detector.class_to_idx["Loitering"]] = 0.07
        return "Fighting", 0.93, probs

    interaction_detector._run_inference = fake_pair_fighting
    p2 = MockPerson(21)
    p3 = MockPerson(22)
    p2.bbox = [100, 100, 220, 340]
    p2.width = 120
    p2.height = 240
    p3.bbox = [150, 110, 270, 350]
    p3.width = 120
    p3.height = 240
    p2.center = (160, 220)
    p3.center = (210, 230)
    pair_tracker = FastTracker({21: (55.0, 0.0), 22: (-50.0, 0.0)})

    pair_dets = []
    for _ in range(34):
        pair_dets = interaction_detector.update([p2, p3], pair_tracker, frame_shape)

    print(f"  ✓ Two-person fighting detections: {len(pair_dets)} (expected >= 1)")
    assert any(det["event_type"] == "Fighting" for det in pair_dets)

    print("\n2d. Testing Action Transition (first action can yield to next action)...")
    transition_detector = ActionDetector("config.json")
    walking_idx = transition_detector.class_to_idx["Walking"]
    running_idx = transition_detector.class_to_idx["Running"]
    scripted_probs = []
    for _ in range(4):
        probs = np.zeros(len(transition_detector.classes), dtype=np.float32)
        probs[walking_idx] = 0.90
        probs[running_idx] = 0.10
        scripted_probs.append(("Walking", 0.90, probs))
    for _ in range(4):
        probs = np.zeros(len(transition_detector.classes), dtype=np.float32)
        probs[running_idx] = 0.92
        probs[walking_idx] = 0.08
        scripted_probs.append(("Running", 0.92, probs))

    def scripted_inference(_sequence):
        if scripted_probs:
            return scripted_probs.pop(0)
        probs = np.zeros(len(transition_detector.classes), dtype=np.float32)
        probs[running_idx] = 1.0
        return "Running", 1.0, probs

    transition_detector._run_inference = scripted_inference
    mover = MockPerson(31)
    mover.bbox = [100, 100, 220, 320]
    mover.width = 120
    mover.height = 220
    tracker_for_transition = FastTracker({31: (30.0, 0.0)})

    labels_seen = []
    for _ in range(38):
        transition_detector.update([mover], tracker_for_transition, frame_shape)
        labels_seen.append(mover.event_type)

    print(f"  ✓ Transition labels near end: {labels_seen[-8:]}")
    assert "Walking" in labels_seen
    assert labels_seen[-1] == "Running"
        
    print("\n2e. Testing Panic Guardrails (single person should be suppressed)...")
    panic_detector = ActionDetector("config.json")
    panic_idx = panic_detector.class_to_idx["Panic"]
    loitering_idx = panic_detector.class_to_idx["Loitering"]

    def fake_solo_panic(_sequence):
        probs = np.zeros(len(panic_detector.classes), dtype=np.float32)
        probs[panic_idx] = 0.91
        probs[loitering_idx] = 0.09
        return "Panic", 0.91, probs

    panic_detector._run_inference = fake_solo_panic
    solo_panic_person = MockPerson(41)
    solo_panic_tracker = FastTracker({41: (22.0, 10.0)})

    for _ in range(34):
        panic_detector.update([solo_panic_person], solo_panic_tracker, frame_shape)

    print(f"  âœ“ Solo panic label after suppression: {solo_panic_person.event_type}")
    assert solo_panic_person.event_type != "Panic"

    print("\n2f. Testing Panic Guardrails (two nearby people can trigger panic)...")
    group_panic_detector = ActionDetector("config.json")

    def fake_group_panic(_sequence):
        probs = np.zeros(len(group_panic_detector.classes), dtype=np.float32)
        probs[group_panic_detector.class_to_idx["Panic"]] = 0.88
        probs[group_panic_detector.class_to_idx["Loitering"]] = 0.12
        return "Panic", 0.88, probs

    group_panic_detector._run_inference = fake_group_panic
    p4 = MockPerson(51)
    p5 = MockPerson(52)
    p4.bbox = [100, 100, 220, 320]
    p5.bbox = [200, 110, 320, 330]
    p4.width = 120
    p4.height = 220
    p5.width = 120
    p5.height = 220
    p4.center = (160, 210)
    p5.center = (260, 220)
    panic_tracker = FastTracker({51: (18.0, 12.0), 52: (-16.0, 11.0)})

    panic_dets = []
    for _ in range(34):
        panic_dets = group_panic_detector.update([p4, p5], panic_tracker, frame_shape)

    panic_support = int(group_panic_detector._class_training_counts.get("Panic", 0))
    print(f"  ✓ Two-person panic detections: {len(panic_dets)} | trained Panic samples: {panic_support}")
    if panic_support > 0:
        assert any(det["event_type"] == "Panic" for det in panic_dets)
    else:
        assert all(det["event_type"] != "Panic" for det in panic_dets)

    print("\n3. Testing EventTriggerEngine Format Generation...")
    trigger_engine = EventTriggerEngine("config.json")
    scoring = ScoringEngine("config.json")
    
    # Manually inject a confident fighting event
    mock_action = [{"person": p1, "event_type": "Fighting", "confidence": 0.88}]
    
    events = trigger_engine.process_events(mock_action, [], scoring)
    print(f"  ✓ Events generated: {len(events)}")
    
    if events:
        ev = events[0]
        print("\n  --- EVENT PAYLOAD ---")
        for k, v in ev.items():
            print(f"  {k}: {v}")
        print("  ---------------------\n")
        
        # Verify schema core components
        assert "event_type" in ev
        assert "confidence_score" in ev
        assert "health_score_delta" in ev
        assert "timestamp" in ev
        print("  ✓ Schema verification passed.")

    print("\n✓ ALL SESSION ML TESTS COMPLETED.\n")


if __name__ == "__main__":
    test()
