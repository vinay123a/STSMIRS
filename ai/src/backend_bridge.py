"""
Bridge from the AI action pipeline to the STSMIRS backend/blockchain server.

The backend project lives beside this repo, so this adapter keeps integration
small and optional. Action detections are sent via HTTP POST to the central server.
"""

import json
import os
import sys
import time
import requests
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class BackendBridge:
    """Forwards high-confidence AI detections into the backend via HTTP REST API."""

    def __init__(self, config_path="config.json"):
        self.project_root = Path(config_path).resolve().parent
        with open(config_path, "r") as f:
            config = json.load(f)

        self.cfg = config.get("backend_bridge", {})
        self.enabled = self.cfg.get("enabled", False)
        self.api_url = self.cfg.get("api_url", "http://localhost:5000/api/detect")
        self.source = self.cfg.get("source", "AI_CAMERA")
        self.default_zone_id = self.cfg.get("default_zone_id", "A")
        self.cooldown_sec = float(self.cfg.get("event_cooldown_sec", 12))
        self.action_event_map = self.cfg.get(
            "action_event_map",
            {
                "Fall": "FALL",
                "Lying_Still": "PANIC",
                "Fighting": "AGGRESSION",
                "Panic": "PANIC"
            },
        )

        self._last_sent = {}

        if self.enabled:
            print(f"[BackendBridge] HTTP Bridge initialized targeting {self.api_url}")

    def ingest_action(self, detection):
        """Send one action detection dict from ActionDetector.update()."""
        if not self.enabled:
            return None

        event_name = detection.get("event_type")
        backend_event_name = self.action_event_map.get(event_name)
        if not backend_event_name:
            return None

        person = detection.get("person")
        if person is None:
            return None

        tourist_id = getattr(person, "track_label", None) or f"T-{person.track_id:03d}"
        zone_id = getattr(person, "zone_id", None) or self.default_zone_id
        confidence = float(detection.get("confidence", 0.0))

        key = (getattr(person, "track_id", tourist_id), backend_event_name, zone_id)
        now = time.time()
        if now - self._last_sent.get(key, 0.0) < self.cooldown_sec:
            return None
        self._last_sent[key] = now

        payload = {
            "source": self.source,
            "event_type": backend_event_name,
            "confidence": confidence,
            "zone_id": zone_id,
            "tourist_id": tourist_id
        }

        try:
            # Add a short timeout so the AI thread doesn't hang if the server is down
            response = requests.post(self.api_url, json=payload, timeout=2.0)
            if response.status_code == 200:
                result = response.json()
                print(f"[BackendBridge] Sent {event_name} -> {result.get('final_action', 'OK')}")
                return result
            else:
                print(f"[BackendBridge] Error from server: HTTP {response.status_code}")
                return None
        except requests.exceptions.RequestException as exc:
            print(f"[BackendBridge] Failed to send backend event (Server unreachable?): {exc}")
            return None
