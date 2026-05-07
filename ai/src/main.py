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

# Panic Module — loaded at top level so it's available throughout main()
try:
    from src.panic_detector import PanicDetector as _PanicDetector
except ImportError:
    _PanicDetector = None


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
        # Note: skeleton_face_recognizer kept for future use
        # skeleton_face_recognizer = SkeletonAwareFaceRecognizer(face_recognizer, skeleton_extractor, config_path)

        display = DisplayEngine(config_path)

        print("[Pipeline] ✓ Modules initialized successfully.")

    except Exception as e:
        print(f"[Pipeline] ❌ Error initializing modules: {e}")
        traceback.print_exc()
        return

    # ─── Panic Detector Instance (outside try block so it's always available) ───
    external_panic_detector = None
    if _PanicDetector is not None:
        external_panic_detector = _PanicDetector(
            speed_threshold=120.0,
            min_people=2,
            divergence_threshold=0.6,
            persistence_frames=10
        )
        print("[Pipeline] ✓ Panic Detector initialized.")
    else:
        print("[Pipeline] ⚠ Panic Detector not available.")

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

    import threading
    import copy

    shared_state = {
        'frame': None,
        'persons_for_worker': [],
        'worker_results': {},
        'running': True
    }
    state_lock = threading.Lock()

    def ai_worker():
        heartbeat = 0
        while shared_state['running']:
            with state_lock:
                frame = shared_state['frame']
                if frame is not None:
                    frame = frame.copy()
                persons = shared_state['persons_for_worker']
            
            if frame is None or not persons:
                time.sleep(0.01)
                continue

            heartbeat += 1
            # if heartbeat % 30 == 1:
            #     print(f"[WORKER] Running... processed {heartbeat} frames, tracking {len(persons)} persons")

            try:
                # B. Face Recognition
                face_recognizer.identify_persons(frame, persons)

                # C. Skeleton Extraction
                skeleton_result = skeleton_extractor.extract_skeleton(frame)
                num_kpts = len(skeleton_result['keypoints'])
                # if heartbeat % 30 == 1:
                #     print(f"[WORKER] Skeleton keypoints detected: {num_kpts}")
                if num_kpts > 0:
                    for i, person in enumerate(persons):
                        if i < len(skeleton_result['keypoints']):
                            person_skeleton = skeleton_result['keypoints'][i]
                            normalized_kpts = skeleton_extractor.normalize_keypoints(
                                person_skeleton,
                                skeleton_result['frame_height'],
                                skeleton_result['frame_width']
                            )
                            track_id = getattr(person, 'track_id', i)
                            skeleton_action_detector.update_skeleton(track_id, normalized_kpts)
                            
                            skeleton_features = skeleton_extractor.extract_skeleton_features(person_skeleton)
                            
                            # Note: Skeleton action detector might be slow to trigger because it receives < 30 FPS
                            skeleton_pred = skeleton_action_detector.predict(track_id)
                            
                            with state_lock:
                                if track_id not in shared_state['worker_results']:
                                    shared_state['worker_results'][track_id] = {}
                                
                                shared_state['worker_results'][track_id]['skeleton_keypoints'] = normalized_kpts
                                shared_state['worker_results'][track_id]['skeleton_features'] = skeleton_features
                                shared_state['worker_results'][track_id]['skeleton_probabilities'] = skeleton_pred.get('probabilities', {})
                                
                                # ALWAYS report the action name, don't wait for 'valid'
                                shared_state['worker_results'][track_id]['skeleton_action'] = skeleton_pred['action']
                                shared_state['worker_results'][track_id]['skeleton_confidence'] = skeleton_pred['confidence']

                # D. Heavy Frame Classification (Fighting)
                frame_action = action_frame_classifier.update(frame, persons)
                frame_action_detections = action_frame_classifier.apply_to_persons(persons, frame_action)
                
                with state_lock:
                    for det in frame_action_detections:
                        tid = det["person"].track_id
                        if tid not in shared_state['worker_results']:
                            shared_state['worker_results'][tid] = {}
                        shared_state['worker_results'][tid]['frame_action'] = det["event_type"]
                        shared_state['worker_results'][tid]['frame_confidence'] = det["confidence"]

            except Exception as inner_error:
                print("[ERROR] AI WORKER ERROR:", inner_error)
                traceback.print_exc()
                time.sleep(0.1)

    # Start the background AI worker
    worker_thread = threading.Thread(target=ai_worker, daemon=True)
    worker_thread.start()

    # ─── 3. Main Loop (Capture & Tracker & Display) ───────────────
    frame_count = 0

    try:
        while True:
            ret, frame = reader.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            frame_shape = frame.shape
            ext_panic_status = False
            ext_panic_metrics = {}

            # A. High-Speed Tracking
            persons = tracker.update(frame)
            zone_violations = zone_manager.update(persons)
            
            # --- DEBUG: Print tracking status every 30 frames ---
            if frame_count % 30 == 0:
                print(f"[Pipeline] Tracking {len(persons)} persons...")
            frame_count += 1

            # --- NEW: Immediately lock names from cache to prevent 'T-001' flickering ---
            face_recognizer.apply_cached_names(persons)
            
            # [ADDED] Call External Panic Detection Logic (REUSING SAME FRAME)
            if external_panic_detector:
                try:
                    ext_panic_status, ext_panic_metrics = external_panic_detector.update(persons, tracker)
                except Exception as e:
                    print(f"[Pipeline] Panic Detection Error: {e}")

            # Apply worker results to freshly tracked persons
            with state_lock:
                for person in persons:
                    tid = person.track_id
                    res = shared_state['worker_results'].get(tid, {})
                    if 'skeleton_action' in res:
                        setattr(person, 'skeleton_action', res['skeleton_action'])
                        setattr(person, 'skeleton_confidence', res['skeleton_confidence'])
                    if 'skeleton_keypoints' in res:
                        setattr(person, 'skeleton_keypoints', res['skeleton_keypoints'])
                        setattr(person, 'skeleton_features', res['skeleton_features'])
                    if 'skeleton_probabilities' in res:
                        setattr(person, 'action_probabilities', res['skeleton_probabilities'])
                    if 'frame_action' in res:
                        setattr(person, 'event_type', res['frame_action'])
                        setattr(person, 'action_confidence', res['frame_confidence'])

            # B. High-Speed Action LSTM (Bounding Box Based)
            action_detections = action_detector.update(persons, tracker, frame_shape)
            
            # Merge Skeleton Actions from the background thread into the event pipeline
            for person in persons:
                skel_action = getattr(person, 'skeleton_action', None)
                if skel_action in ["Fall", "Panic"]:
                    # Avoid duplicate events if traditional detector already caught it
                    if not any(d["person"].track_id == person.track_id and d["event_type"] == skel_action for d in action_detections):
                        action_detections.append({
                            "person": person,
                            "event_type": skel_action,
                            "confidence": getattr(person, 'skeleton_confidence', 0.8)
                        })
            
            events = event_trigger.process_events(action_detections, zone_violations, scoring_engine, persons)

            for detection in action_detections:
                res = backend_bridge.ingest_action(detection)
                if res and "zone_score" in res:
                    detection["person"].health_score = res["zone_score"]
                    # Update crime score based on penalty applied
                    prev_crime = getattr(detection["person"], "crime_score", 0)
                    detection["person"].crime_score = prev_crime + res.get("penalty_applied", 0)

            # --- Aggregate and Process Events (Fall, Fight, Zone, etc.) ---
            active_emergencies = []
            for ev in events:
                etype = ev['event_type']
                tid = ev['tourist_id']
                
                # 1. Handle Zone Violations
                if etype == 'Zone_Violation':
                    res = backend_bridge.ingest_zone_violation(ev)
                    active_emergencies.append(f"ZONE BREACH ({ev['zone_id']})")
                    display.add_alert(f"RESTRICTED AREA: {tid}", level="warning")
                    # Update local person object score if possible
                    p_obj = ev.get("person")
                    if p_obj and res and "zone_score" in res:
                        p_obj.health_score = res["zone_score"]
                        p_obj.crime_score = getattr(p_obj, "crime_score", 0) + res.get("penalty_applied", 40)
                
                # 2. Handle Action Emergencies (Fall, Fighting, Panic)
                if etype in ["Fall", "Fighting", "Panic"] or ev['confidence_level'] == "Critical":
                    active_emergencies.append(etype.upper())
                    display.add_alert(f"{etype.upper()} DETECTED: {tid}", level="danger")
            
            # 3. Trigger Combined Emergency Banner if any events occurred
            if active_emergencies:
                combined_msg = " + ".join(sorted(list(set(active_emergencies))))
                print(f"[UI_ALERT] Triggering banner: {combined_msg}") # Debug print
                display.trigger_emergency(
                    f"!!! {combined_msg} !!! >>> Person: {tid} >>> Blockchain Audit Logged"
                )

            scoring_engine.update_persons(persons)

            # Give the worker thread the latest frame and a reference to the tracked persons
            with state_lock:
                shared_state['frame'] = frame
                # Shallow copy of the list (references to the same TrackedPerson objects)
                shared_state['persons_for_worker'] = list(persons)

            # G. Display
            zones_overlay = zone_manager.get_zones_for_display()
            display_frame = display.render(
                frame,
                persons=persons,
                zones=zones_overlay,
                fps=reader.get_fps(),
                external_panic_status=ext_panic_status,
                external_panic_metrics=ext_panic_metrics
            )
            cv2.imshow(display.window_name, display_frame)

            # Keyboard control
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[Pipeline] Quitting...")
                shared_state['running'] = False
                break

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted manually.")
        shared_state['running'] = False

    except Exception as e:
        print(f"\n[ERROR] FATAL ERROR: {e}")
        traceback.print_exc()
        shared_state['running'] = False

    finally:
        # ─── 4. Cleanup ───────────────────────────────────────
        reader.stop()
        cv2.destroyAllWindows()
        print("[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()
