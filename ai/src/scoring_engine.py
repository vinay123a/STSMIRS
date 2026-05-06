"""
STSMIRS — Scoring Engine
Maintains persistent health and crime scores for active tracked persons.
"""

import json
import time

class ScoringEngine:
    """Tracks and updates Health and Crime scores per person."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
            
        trigger_cfg = config.get("event_trigger", {})
        
        self.health_init = trigger_cfg.get("health_score_init", 100)
        self.crime_init = trigger_cfg.get("crime_score_init", 0)
        
        self.health_emergency_threshold = trigger_cfg.get("health_emergency_threshold", 30)
        self.crime_emergency_threshold = trigger_cfg.get("crime_emergency_threshold", 70)
        
        self.score_deltas = trigger_cfg.get("score_deltas", {})
        
        # State: {"track_id": {"health": 100, "crime": 0, "last_updated": float}}
        self._scores = {}
        self._last_emergencies = {} # track_id -> timestamp (to prevent spamming)
        
    def _init_person(self, track_id):
        """Initialize scores for a new person."""
        if track_id not in self._scores:
            self._scores[track_id] = {
                "health": self.health_init,
                "crime": self.crime_init,
                "last_updated": time.time()
            }
            
    def apply_event(self, track_id, event_type):
        """
        Apply score deltas based on an event type.
        
        Args:
            track_id: int
            event_type: str (e.g., "Fall", "Fighting", "Zone_Violation")
            
        Returns:
            dict with updated health and crime scores
        """
        self._init_person(track_id)
        
        delta = self.score_deltas.get(event_type, {"health": 0, "crime": 0})
        
        # Apply deltas
        new_health = self._scores[track_id]["health"] + delta.get("health", 0)
        new_crime = self._scores[track_id]["crime"] + delta.get("crime", 0)
        
        # Clamp bounds
        self._scores[track_id]["health"] = max(0, min(100, new_health))
        self._scores[track_id]["crime"] = max(0, min(100, new_crime))
        self._scores[track_id]["last_updated"] = time.time()
        
        print(f"[ScoreEngine] Applied '{event_type}' to T-{track_id:03d} -> Health: {self._scores[track_id]['health']}, Crime: {self._scores[track_id]['crime']}")
        
        return self._scores[track_id]

    def update_persons(self, persons):
        """
        Sync scores to the TrackedPerson objects for display rendering.
        Also garbage collects stale tracks.
        
        Args:
            persons: list of TrackedPerson objects (modified in place)
            
        Returns:
            list of emergency trigger events (dicts)
        """
        emergencies = []
        now = time.time()
        active_ids = {p.track_id for p in persons}
        
        for person in persons:
            self._init_person(person.track_id)
            scores = self._scores[person.track_id]
            
            # Update person object
            person.health_score = scores["health"]
            person.crime_score = scores["crime"]
            
            # Check for emergencies (only trigger once every 10 seconds per person)
            last_emerg = self._last_emergencies.get(person.track_id, 0)
            if now - last_emerg > 10.0:
                is_emergency = False
                msg = ""
                
                if scores["health"] <= self.health_emergency_threshold:
                    is_emergency = True
                    msg = f"EMERGENCY >>> HEALTH CRITICAL ({scores['health']}) — T-{person.track_id:03d}"
                elif scores["crime"] >= self.crime_emergency_threshold:
                    is_emergency = True
                    msg = f"EMERGENCY >>> CRIME CRITICAL ({scores['crime']}) — T-{person.track_id:03d}"
                    
                if is_emergency:
                    self._last_emergencies[person.track_id] = now
                    emergencies.append({
                        "person": person,
                        "message": msg,
                        "timestamp": now,
                        "health": scores["health"],
                        "crime": scores["crime"]
                    })
                    
        # Garbage collect stale tracks (not seen in last 5 seconds)
        stale_threshold = now - 5.0
        keys_to_delete = []
        for tid, data in self._scores.items():
            if tid not in active_ids and data["last_updated"] < stale_threshold:
                keys_to_delete.append(tid)
                
        for tid in keys_to_delete:
            del self._scores[tid]
            if tid in self._last_emergencies:
                del self._last_emergencies[tid]
                
        return emergencies


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/scoring_engine.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    class MockPerson:
        def __init__(self, id):
            self.track_id = id
            self.health_score = 100
            self.crime_score = 0
            
    print("=" * 50)
    print("  STSMIRS — Scoring Engine Test")
    print("=" * 50)
    
    engine = ScoringEngine("config.json")
    p1 = MockPerson(1)
    
    print("\n[Initial Sync]")
    engine.update_persons([p1])
    print(f"P1: Health={p1.health_score}, Crime={p1.crime_score}")
    
    print("\n[Apply Zone Violation]")
    engine.apply_event(1, "Zone_Violation")
    engine.update_persons([p1])
    print(f"P1: Health={p1.health_score}, Crime={p1.crime_score}")
    
    print("\n[Apply Fall]")
    engine.apply_event(1, "Fall")
    engine.update_persons([p1])
    print(f"P1: Health={p1.health_score}, Crime={p1.crime_score}")
    
    print("\n[Apply Fighting x2 -> Should trigger emergency]")
    engine.apply_event(1, "Fighting")
    engine.apply_event(1, "Fighting")
    emergencies = engine.update_persons([p1])
    print(f"P1: Health={p1.health_score}, Crime={p1.crime_score}")
    print(f"Emergencies triggered: {len(emergencies)}")
    if emergencies:
        print(f"  {emergencies[0]['message']}")
        
    print("\n[Done] Score test complete.")
