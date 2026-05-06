"""
STSMIRS — Stream Reader Module
Reads live video from phone camera (IP Webcam) via OpenCV.
Features: threaded frame buffer, auto-reconnect, FPS counter.

Usage (standalone test):
    python src/stream_reader.py
"""

import cv2
import time
import json
import threading
import numpy as np
import platform
from collections import deque


class StreamReader:
    """Threaded video stream reader with auto-reconnect."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)["stream"]

        self.url = self.config["url"]
        self.fallback_url = self.config.get("fallback_url", "0")
        self.target_fps = self.config.get("target_fps", 15)
        self.buffer_size = self.config.get("buffer_size", 2)
        self.reconnect_delay = self.config.get("reconnect_delay_sec", 3)
        self.max_reconnect = self.config.get("max_reconnect_attempts", 10)

        self.cap = None
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.thread = None
        self.last_source = None

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

    @staticmethod
    def _is_webcam_source(source):
        return isinstance(source, int) or (isinstance(source, str) and source.isdigit())

    def _open_capture(self, source):
        """Prefer DirectShow for Windows webcams because MSMF is flaky here."""
        if platform.system().lower().startswith("win") and self._is_webcam_source(source):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)

    def connect(self, use_fallback=False):
        """Connect to the video source."""
        source = self.fallback_url if use_fallback else self.url

        # If fallback is "0", use default webcam (integer)
        if source == "0":
            source = 0

        print(f"[StreamReader] Connecting to: {source}")

        if self.cap is not None:
            self.cap.release()

        self.cap = self._open_capture(source)
        self.last_source = source

        if not self.cap.isOpened():
            print(f"[StreamReader] Failed to connect to {source}")
            self.connected = False
            return False

        # Try to set resolution
        res = self.config.get("resolution", [720, 1280])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[1])

        first_good_frame = False
        for _ in range(8):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame_buffer.clear()
                    self.frame_buffer.append(frame)
                first_good_frame = True
                break
            time.sleep(0.12)

        if not first_good_frame:
            print(f"[StreamReader] Connected to {source}, but no readable frames arrived.")
            self.cap.release()
            self.cap = None
            self.connected = False
            return False

        self.connected = True
        print(f"[StreamReader] Connected! Resolution: "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True

    def _read_loop(self):
        """Background thread: continuously reads frames into buffer."""
        reconnect_count = 0

        while self.running:
            if not self.connected:
                # Attempt reconnection
                if reconnect_count >= self.max_reconnect:
                    print(f"[StreamReader] Max reconnect attempts reached. "
                          f"Trying fallback...")
                if not self.connect(use_fallback=True):
                    print("[StreamReader] Fallback also failed. Stopping.")
                    self.running = False
                    break
                    reconnect_count = 0
                else:
                    print(f"[StreamReader] Reconnecting in "
                          f"{self.reconnect_delay}s... "
                          f"(attempt {reconnect_count + 1}/{self.max_reconnect})")
                    time.sleep(self.reconnect_delay)
                    self.connect()
                    reconnect_count += 1
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("[StreamReader] Frame read failed. Reconnecting...")
                self.connected = False
                continue

            # Update buffer (thread-safe)
            with self.lock:
                self.frame_buffer.append(frame)

            # Update FPS
            self._fps_counter += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = time.time()

            # Throttle to target FPS
            time.sleep(max(0, 1.0 / self.target_fps - 0.001))

    def start(self):
        """Start the threaded stream reader."""
        if self.running:
            print("[StreamReader] Already running.")
            return

        # Try primary URL first, then fallback
        if not self.connect():
            print("[StreamReader] Primary URL failed. Trying fallback...")
            if not self.connect(use_fallback=True):
                print("[StreamReader] ✗ All connections failed.")
                return

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("[StreamReader] Stream reader started.")

    def read(self):
        """Get the latest frame. Returns (success, frame)."""
        with self.lock:
            if len(self.frame_buffer) == 0:
                return False, None
            return True, self.frame_buffer[-1].copy()

    def get_fps(self):
        """Get current FPS."""
        return self._current_fps

    def get_reference_frame(self):
        """Capture a single reference frame (for zone drawing)."""
        ret, frame = self.read()
        if ret:
            return frame
        # If buffer empty, try direct read
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def stop(self):
        """Stop the stream reader."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.cap:
            self.cap.release()
        print("[StreamReader] Stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/stream_reader.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="STSMIRS Stream Reader Test")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json")
    parser.add_argument("--source", default=None,
                        help="Override stream URL (or '0' for webcam)")
    args = parser.parse_args()

    # Load or override config
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        print("  Run from project root: python src/stream_reader.py")
        exit(1)

    reader = StreamReader(config_path)

    # Override URL if provided
    if args.source:
        reader.url = args.source
        if args.source == "0":
            reader.url = "0"
            reader.fallback_url = "0"

    print("=" * 50)
    print("  STSMIRS — Stream Reader Test")
    print("  Press 'q' to quit, 's' to save a snapshot")
    print("=" * 50)

    reader.start()
    time.sleep(1)  # Give it a moment to connect

    while True:
        ret, frame = reader.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Draw FPS on frame
        fps = reader.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "STSMIRS Stream Test", (10, 65),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to snapshot",
                     (10, frame.shape[0] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("STSMIRS Stream Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snap_path = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"[Snapshot] Saved: {snap_path}")

    reader.stop()
    cv2.destroyAllWindows()
    print("[Done] Stream reader test complete.")
