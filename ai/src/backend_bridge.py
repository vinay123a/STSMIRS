"""
Bridge from the AI action pipeline to the STSMIRS backend/blockchain server.

The backend project lives beside this repo, so this adapter keeps integration
small and optional: action detections still work even if the backend is absent.
"""

import json
import os
import sys
import time
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class BackendBridge:
    """Forwards high-confidence AI detections into the backend CentralServer."""

    def __init__(self, config_path="config.json"):
        self.project_root = Path(config_path).resolve().parent
        with open(config_path, "r") as f:
            config = json.load(f)

        self.cfg = config.get("backend_bridge", {})
        self.enabled = self.cfg.get("enabled", False)
        self.source = self.cfg.get("source", "AI_CAMERA")
        self.state_dir = Path(self.cfg.get("state_dir", "data/backend_bridge"))
        if not self.state_dir.is_absolute():
            self.state_dir = self.project_root / self.state_dir
        self.default_zone_id = self.cfg.get("default_zone_id", "CAMERA_ZONE")
        self.cooldown_sec = float(self.cfg.get("event_cooldown_sec", 12))
        self.action_event_map = self.cfg.get(
            "action_event_map",
            {
                "Fall": "FALL",
                "Lying_Still": "PANIC",
                "Fighting": "AGGRESSION",
            },
        )

        self.server = None
        self.DetectionEvent = None
        self.EventType = None
        self._last_sent = {}

        if self.enabled:
            self._load_backend()

    def _load_backend(self):
        backend_path = Path(self.cfg.get("stsmirs_path", "")).expanduser()
        if not backend_path.exists():
            print(f"[BackendBridge] Backend path not found: {backend_path}")
            self.enabled = False
            return

        sys.path.insert(0, str(backend_path))
        old_cwd = os.getcwd()
        try:
            # central_server stores state next to its own file, but switching cwd
            # helps any relative imports in the backend behave like its demo.
            os.chdir(backend_path)
            from central_server import CentralServer, DetectionEvent, EventType

            self.server = CentralServer()
            self._redirect_backend_state()
            self.DetectionEvent = DetectionEvent
            self.EventType = EventType
            print(f"[BackendBridge] Connected to backend at {backend_path}")
        except Exception as exc:
            print(f"[BackendBridge] Backend disabled: {exc}")
            self.enabled = False
        finally:
            os.chdir(old_cwd)

    def _redirect_backend_state(self):
        """Keep backend event/zone logs in this AI project, not the external repo."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.server.state_file = str(self.state_dir / "zones_state.json")
        self.server.events_file = str(self.state_dir / "events_log.json")
        self.server._load_state()
        self.server._load_events()

    def ingest_action(self, detection):
        """Send one action detection dict from ActionDetector.update()."""
        if not self.enabled or self.server is None:
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

        try:
            event_type = getattr(self.EventType, backend_event_name)
            event = self.DetectionEvent(
                tourist_id=tourist_id,
                event_type=event_type,
                confidence=confidence,
                zone_id=zone_id,
                source=self.source,
            )
            result = self.server.ingest_event(event)
            print(f"[BackendBridge] Sent {event_name} -> {result.get('action')}")
            return result
        except Exception as exc:
            print(f"[BackendBridge] Failed to send backend event: {exc}")
            return None
