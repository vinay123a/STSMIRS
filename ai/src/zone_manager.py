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
            # Use center point for much higher sensitivity (any part of person entering)
            center_x = person.bbox[0] + person.bbox[2]/2
            center_y = person.bbox[1] + person.bbox[3]/2
            test_point = (center_x, center_y)
            
            assigned_zone = "Outside"
            assigned_type = self.default_zone_type
            
            for zone in self.zones:
                dist = cv2.pointPolygonTest(zone["polygon_pts"], (float(test_point[0]), float(test_point[1])), measureDist=False)
                if dist >= 0:
                    assigned_zone = zone["zone_id"]
                    assigned_type = zone["type"]
                    break
            
            person.zone_id = assigned_zone
            person.zone_type = assigned_type
            
            # LOUD DEBUG: Print to terminal if in a restricted area
            if assigned_type == "Restricted":
                print("\n" + "!"*60)
                print(f"!!! [RESTRICTED ZONE] Person {person.track_id} detected in {assigned_zone} !!!")
                print("!"*60 + "\n")
            
            # Detect zone transition
            prev_zone = self._current_zones.get(person.track_id)
            
            if assigned_zone and assigned_zone != prev_zone:
                # Log transition for console/debugging
                print(f"[ZoneManager] Transition: {person.track_label} entered {assigned_zone} ({assigned_type})")
            
            # If in Restricted zone, ALWAYS report it as a violation for the current frame.
            # (Throttling/cooldown is handled by EventTriggerEngine)
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
