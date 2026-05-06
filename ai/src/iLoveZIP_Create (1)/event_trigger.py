"""
STSMIRS — Event Trigger Engine
Accepts ML detections/violations and generates formatted JSON payloads
while handling confidence routing and alert cooldowns.
"""

import time
import json
from datetime import datetime

class EventTriggerEngine:
    """Generates Event JSONs and filters duplicate alerts."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        trigger_cfg = config.get("event_trigger", {})
        self.conf_high = trigger_cfg.get("confidence_high", 0.85)
        self.conf_medium = trigger_cfg.get("confidence_medium", 0.50)
        self.cooldown_sec = trigger_cfg.get("alert_cooldown_sec", 5)
        self.camera_id = config.get("camera", {}).get("id", "CAMERA-01")
        
        # State: track_id -> {event_type: last_trigger_time}
        self._last_events = {}

    def _can_trigger(self, track_id, event_type, now):
        """Check if enough time has passed since this person triggered this event."""
        if track_id not in self._last_events:
            self._last_events[track_id] = {}
            
        last_time = self._last_events[track_id].get(event_type, 0)
        if now - last_time >= self.cooldown_sec:
            self._last_events[track_id][event_type] = now
            return True
            
        return False

    def process_events(self, action_detections, zone_violations, scoring_engine):
        """
        Consolidate raw detections into standardized events, apply scoring, 
        and filter through cooldown logic.
        
        Returns:
            list of Event JSON dicts ready for server/display
        """
        events = []
        now = time.time()
        
        # 1. Process Action Detections (from LSTM)
        for det in action_detections:
            person = det["person"]
            event_type = det["event_type"]
            conf = det["confidence"]
            
            # Confidence routing
            if conf >= self.conf_high:
                level = "High"
            elif conf >= self.conf_medium:
                level = "Medium"
            else:
                continue # Discard low confidence
                
            # Filter duplicates via cooldown
            if self._can_trigger(person.track_id, event_type, now):
                # Update scores
                scores = scoring_engine.apply_event(person.track_id, event_type)
                person.is_alert = True
                
                # Build JSON payload
                events.append({
                    "event_type": event_type,
                    "confidence_level": level,
                    "confidence_score": float(conf),
                    "health_score_delta": scoring_engine.score_deltas.get(event_type, {}).get("health", 0),
                    "crime_score_delta": scoring_engine.score_deltas.get(event_type, {}).get("crime", 0),
                    "tourist_id": person.track_label,
                    "zone_id": person.zone_id or "Unknown",
                    "camera_id": self.camera_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "current_health": scores["health"],
                    "current_crime": scores["crime"]
                })

        # 2. Process Zone Violations
        for viol in zone_violations:
            person = viol["person"]
            zone_id = viol["zone_id"]
            event_type = "Zone_Violation"
            
            if self._can_trigger(person.track_id, event_type, now):
                scores = scoring_engine.apply_event(person.track_id, event_type)
                person.is_alert = True
                
                events.append({
                    "event_type": event_type,
                    "confidence_level": "High", # Definite algorithmic rule
                    "confidence_score": 1.0,
                    "health_score_delta": scoring_engine.score_deltas.get(event_type, {}).get("health", 0),
                    "crime_score_delta": scoring_engine.score_deltas.get(event_type, {}).get("crime", 0),
                    "tourist_id": person.track_label,
                    "zone_id": zone_id,
                    "camera_id": self.camera_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "current_health": scores["health"],
                    "current_crime": scores["crime"]
                })
                
        # Reset alert flag on people over time (handled natively by rendering engine expiring banners)

        return events


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/event_trigger.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.scoring_engine import ScoringEngine
    
    print("=" * 50)
    print("  STSMIRS — Event Trigger Test")
    print("=" * 50)
    
    trigger = EventTriggerEngine("config.json")
    scoring = ScoringEngine("config.json")
    
    class MockPerson:
        def __init__(self, tid):
            self.track_id = tid
            self.track_label = f"T-{tid:03d}"
            self.zone_id = "ZONE-A"
            self.health_score = 100
            self.crime_score = 0
            self.is_alert = False
            
    p1 = MockPerson(1)
    
    print("\n[Test 1: High Confidence Fall]")
    actions = [{"person": p1, "event_type": "Fall", "confidence": 0.95}]
    events = trigger.process_events(actions, [], scoring)
    print(f"Events generated: {len(events)}")
    if events: print(json.dumps(events[0], indent=2))
    
    print("\n[Test 2: Medium Confidence Fighting]")
    actions = [{"person": p1, "event_type": "Fighting", "confidence": 0.65}]
    events = trigger.process_events(actions, [], scoring)
    print(f"Events generated: {len(events)}")
    if events: print(f"  Confidence Level: {events[0]['confidence_level']}")
    
    print("\n[Test 3: Cooldown block (Fall again instantly)]")
    actions = [{"person": p1, "event_type": "Fall", "confidence": 0.95}]
    events = trigger.process_events(actions, [], scoring)
    print(f"Events generated: {len(events)} (Expected 0 due to cooldown)")
    
    print("\n[Done] Trigger test complete.")
