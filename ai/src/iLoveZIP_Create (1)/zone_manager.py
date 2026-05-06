"""
STSMIRS — Zone Manager (Camera-Based Geofencing)
Loads zone polygons, performs point-in-polygon tests, detects zone violations.
"""

import json
import time
import cv2
import numpy as np

class ZoneManager:
    """Manages virtual zones and checks person locations against them."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.zones_cfg = self.config.get("zones", {})
        self.zones_file = self.zones_cfg.get("config_file", "zones_config.json")
        self.default_zone_type = self.zones_cfg.get("default_zone_type", "Normal")
        
        # Load zones
        self.zones = []
        self._load_zones()
        
        # Track who is in which zone {"track_id": "zone_id"}
        self._current_zones = {}

    def _load_zones(self):
        """Load zone polygons from the config file."""
        try:
            with open(self.zones_file, "r") as f:
                data = json.load(f)
                zones_data = data.get("zones", [])
                
            self.zones = []
            for z in zones_data:
                # Store points as numpy array for cv2 pointPolygonTest
                pts = np.array(z["polygon"], dtype=np.int32)
                
                self.zones.append({
                    "zone_id": z.get("zone_id"),
                    "type": z.get("type", "Normal"),
                    "polygon_pts": pts,
                    "raw_data": z
                })
            print(f"[ZoneManager] ✓ Loaded {len(self.zones)} zones from {self.zones_file}")
        except Exception as e:
            print(f"[ZoneManager] ⚠ Could not load zones from {self.zones_file}: {e}")
            self.zones = []

    def get_zones_for_display(self):
        """Return raw zone data for display engine."""
        return [z["raw_data"] for z in self.zones]

    def update(self, persons):
        """
        Check all tracked persons against zones.
        
        Args:
            persons: list of TrackedPerson objects (modified in place)
            
        Returns:
            list of zone violation events (dicts: {person, zone_id, type})
        """
        violations = []
        
        for person in persons:
            foot_point = person.foot_point
            assigned_zone = None
            assigned_type = self.default_zone_type
            
            # Check person against all zones
            # If overlapping multiple, last one drawn takes precedence
            for zone in self.zones:
                # pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
                dist = cv2.pointPolygonTest(zone["polygon_pts"], (float(foot_point[0]), float(foot_point[1])), measureDist=False)
                
                if dist >= 0:
                    assigned_zone = zone["zone_id"]
                    assigned_type = zone["type"]
            
            # Update person object state
            person.zone_id = assigned_zone
            
            # Detect zone transition
            prev_zone = self._current_zones.get(person.track_id)
            
            if assigned_zone and assigned_zone != prev_zone:
                # Person entered a new zone
                print(f"[ZoneManager] {'RESTRICTED ALERT' if assigned_type == 'Restricted' else 'Transition'}: {person.track_label} entered {assigned_zone} ({assigned_type})")
                
                # If Restricted, trigger an event
                if assigned_type == "Restricted":
                    violations.append({
                        "person": person,
                        "zone_id": assigned_zone,
                        "timestamp": time.time()
                    })
            
            # Update history
            self._current_zones[person.track_id] = assigned_zone
            
        # Clean up stale tracks from history (prevent memory leak)
        active_ids = {p.track_id for p in persons}
        self._current_zones = {k: v for k, v in self._current_zones.items() if k in active_ids}
        
        return violations


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/zone_manager.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    
    # Ensure import from root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create mock tracker person class for testing
    class MockPerson:
        def __init__(self, id, foot_x, foot_y):
            self.track_id = id
            self.track_label = f"T-{id:03d}"
            self.foot_point = (foot_x, foot_y)
            self.zone_id = None
            
    print("=" * 50)
    print("  STSMIRS — Zone Manager Test")
    print("=" * 50)
    
    zm = ZoneManager("config.json")
    
    print(f"\nEvaluating persons against {len(zm.zones)} loaded zones...")
    
    # Mock some people (based on zones_config.json coordinates)
    # ZONE-A (Restricted): ~ (100, 880) is inside
    # ZONE-B (Normal): ~ (1000, 600) is inside
    # Outside: (100, 100)
    
    p1 = MockPerson(1, 100, 880)
    p2 = MockPerson(2, 1000, 600)
    p3 = MockPerson(3, 100, 100)
    
    print("\n[Frame 1]")
    violations1 = zm.update([p1, p2, p3])
    print(f"P1 assigned to: {p1.zone_id}")
    print(f"P2 assigned to: {p2.zone_id}")
    print(f"P3 assigned to: {p3.zone_id}")
    print(f"Violations triggered: {len(violations1)}")
    
    print("\n[Frame 2 - no movement]")
    violations2 = zm.update([p1, p2, p3])
    print(f"Violations triggered: {len(violations2)} (should be 0, already recorded)")
    
    print("\n[Frame 3 - P3 enters Restricted Zone A]")
    p3.foot_point = (100, 880)
    violations3 = zm.update([p1, p2, p3])
    print(f"P3 assigned to: {p3.zone_id}")
    print(f"Violations triggered: {len(violations3)} (should be 1)")
    
    print("\n[Done] Zone test complete.")
