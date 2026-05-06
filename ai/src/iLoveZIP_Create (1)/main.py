"""
STSMIRS — Main Pipeline Entry Point
Face Recognition + Live Action Tracking
"""

import os
import sys
import cv2
import time
import argparse
import traceback

# Assure correct import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stream_reader import StreamReader
from src.tracker import PersonTracker
from src.display import DisplayEngine
from src.zone_manager import ZoneManager
from src.scoring_engine import ScoringEngine
from src.action_detector import ActionDetector
from src.event_trigger import EventTriggerEngine
from src.backend_bridge import BackendBridge
from src.action_frame_classifier import ActionFrameClassifier
from src.face_recognizer import FaceRecognizer
from src.skeleton_extractor import SkeletonExtractor
from src.skeleton_action_detector import SkeletonActionDetector
from src.skeleton_face_recognizer import SkeletonAwareFaceRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="STSMIRS Core Pipeline")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None, help="Force video source (e.g., '0' for webcam, or URL)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    print("=====================================================")
    print("    STSMIRS — Core AI Pipeline Starting (SAFE MODE)  ")
    print("=====================================================")

    # ─── 1. Initialize Modules ─────────────────────────────────
    print("\n[Pipeline] Initializing modules...")
    try:
        reader = StreamReader(config_path)

        if args.source:
            if args.source == '0':
                reader.url = 0
                reader.fallback_url = 0
            else:
                reader.url = args.source

        tracker = PersonTracker(config_path)
        zone_manager = ZoneManager(config_path)
        action_detector = ActionDetector(config_path)
        scoring_engine = ScoringEngine(config_path)
        event_trigger = EventTriggerEngine(config_path)
        backend_bridge = BackendBridge(config_path)
        action_frame_classifier = ActionFrameClassifier(config_path)
        face_recognizer = FaceRecognizer(config_path)
        
        # Skeleton-based modules
        skeleton_extractor = SkeletonExtractor(config_path)
        skeleton_action_detector = SkeletonActionDetector(config_path)
        skeleton_face_recognizer = SkeletonAwareFaceRecognizer(face_recognizer, skeleton_extractor, config_path)
        
        display = DisplayEngine(config_path)

        print("[Pipeline] ✓ Modules initialized successfully.")

    except Exception as e:
        print(f"[Pipeline] ❌ Error initializing modules: {e}")
        traceback.print_exc()
        return

    # ─── 2. Start Stream ───────────────────────────────────────
    reader.start()

    wait_deadline = time.time() + 6.0
    first_frame_ready = False

    while time.time() < wait_deadline:
        ret, _ = reader.read()
        if ret:
            first_frame_ready = True
            break
        if not reader.running:
            break
        time.sleep(0.2)

    if not reader.connected or not first_frame_ready:
        print("[Pipeline] ❌ Failed to connect to camera. Exiting.")
        reader.stop()
        return

    cv2.namedWindow(display.window_name, cv2.WINDOW_NORMAL)

    print("\n=====================================================")
    print("    System Live. Press 'q' to quit.")
    print("=====================================================\n")

    # ─── 3. Main Loop ──────────────────────────────────────────
    try:
        while True:
            try:
                ret, frame = reader.read()

                if not ret or frame is None:
                    print("⚠ Frame not received")
                    time.sleep(0.01)
                    continue

                frame_shape = frame.shape

                # A. Detect & Track Persons
                persons = tracker.update(frame)

                # B. Face Recognition
                face_recognizer.identify_persons(frame, persons)

                # C. Skeleton Extraction & Enhanced Face Recognition
                skeleton_result = skeleton_extractor.extract_skeleton(frame)
                if len(skeleton_result['keypoints']) > 0:
                    for i, person in enumerate(persons):
                        if i < len(skeleton_result['keypoints']):
                            person_skeleton = skeleton_result['keypoints'][i]
                            # Normalize and store skeleton
                            normalized_kpts = skeleton_extractor.normalize_keypoints(
                                person_skeleton,
                                skeleton_result['frame_height'],
                                skeleton_result['frame_width']
                            )
                            # Update skeleton-based action detector
                            track_id = person.get('track_id', i)
                            skeleton_action_detector.update_skeleton(track_id, normalized_kpts)
                            
                            # Store skeleton features in person dict for other modules
                            person['skeleton_keypoints'] = normalized_kpts
                            skeleton_features = skeleton_extractor.extract_skeleton_features(person_skeleton)
                            person['skeleton_features'] = skeleton_features

                # D. Zone Management
                zone_violations = zone_manager.update(persons)

                # E. Action Detection (traditional + skeleton-based)
                action_detections = action_detector.update(persons, tracker, frame_shape)
                
                # Add skeleton-based predictions
                for person in persons:
                    track_id = person.get('track_id', 0)
                    skeleton_pred = skeleton_action_detector.predict(track_id)
                    if skeleton_pred['valid']:
                        person['skeleton_action'] = skeleton_pred['action']
                        person['skeleton_confidence'] = skeleton_pred['confidence']

                frame_action = action_frame_classifier.update(frame, persons)
                frame_action_detections = action_frame_classifier.apply_to_persons(persons, frame_action)

                if frame_action_detections:
                    action_detections = [
                        det for det in action_detections
                        if det.get("event_type") != "Fighting"
                    ] + frame_action_detections

                # E. Event Processing
                events = event_trigger.process_events(action_detections, zone_violations, scoring_engine)

                for detection in action_detections:
                    backend_bridge.ingest_action(detection)

                for ev in events:
                    print(f"[EVENT] {ev['event_type']} ({ev['confidence_level']})")

                # F. Scoring Updates
                emergencies = scoring_engine.update_persons(persons)

                # G. Display
                zones_overlay = zone_manager.get_zones_for_display()
                output_frame = display.render(frame, persons, zones_overlay, fps=reader.get_fps())

                cv2.imshow(display.window_name, output_frame)

                # Keyboard control
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[Pipeline] Quitting...")
                    break

            except Exception as inner_error:
                print("🔥 LOOP ERROR:", inner_error)
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted manually.")

    except Exception as e:
        print(f"\n🔥 FATAL ERROR: {e}")
        traceback.print_exc()

    finally:
        # ─── 4. Cleanup ───────────────────────────────────────
        reader.stop()
        cv2.destroyAllWindows()
        print("[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()