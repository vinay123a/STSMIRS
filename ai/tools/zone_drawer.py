"""
STSMIRS — Zone Drawer Tool
Interactive OpenCV GUI to draw zone polygons on a reference frame.

Usage:
    python tools/zone_drawer.py
"""

import cv2
import json
import os
import sys
import numpy as np
from datetime import datetime

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stream_reader import StreamReader

class ZoneDrawer:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.zones_file = self.config.get("zones", {}).get("config_file", "zones_config.json")
        self.camera_id = self.config.get("camera", {}).get("id", "CAMERA-01")
        
        # Colors
        zone_colors = self.config.get("zones", {}).get("colors", {})
        self.color_normal = tuple(zone_colors.get("Normal", [0, 255, 0]))
        self.color_restricted = tuple(zone_colors.get("Restricted", [0, 0, 255]))
        
        self.window_name = "STSMIRS Zone Drawer"
        
        # State
        self.zones = []            # list of dicts: {"zone_id": str, "type": str, "polygon": [[x,y],...], "label": str}
        self.current_polygon = []  # points of currently drawing polygon
        self.current_type = "Normal" # "Normal" or "Restricted"
        self.frame = None          # Original reference frame
        self.display_frame = None  # Frame with drawing overlays
        
        # Load existing if available
        self.load_zones()

    def load_zones(self):
        """Load existing zones from file if it exists."""
        if os.path.exists(self.zones_file):
            try:
                with open(self.zones_file, "r") as f:
                    data = json.load(f)
                    self.zones = data.get("zones", [])
                print(f"[ZoneDrawer] Loaded {len(self.zones)} existing zones from {self.zones_file}")
            except Exception as e:
                print(f"[ZoneDrawer] Error loading existing zones: {e}")

    def save_zones(self):
        """Save zones to JSON config."""
        data = {
            "zones": self.zones,
            "reference_frame": f"reference_{self.camera_id}.jpg",
            "created_at": datetime.now().isoformat(),
            "camera_id": self.camera_id
        }
        
        # Save JSON
        with open(self.zones_file, "w") as f:
            json.dump(data, f, indent=2)
            
        # Save reference frame image
        if self.frame is not None:
            cv2.imwrite(data["reference_frame"], self.frame)
            
        print(f"\n[ZoneDrawer] [SAVED] {len(self.zones)} zones to {self.zones_file}")
        print(f"[ZoneDrawer] [SAVED] Reference image to {data['reference_frame']}")

    def get_reference_frame(self):
        """Get a single frame from the camera using direct OpenCV capture."""
        import time
        
        # Get stream URL from config
        stream_url = self.config.get("stream", {}).get("url", "0")
        if stream_url == "0":
            stream_url = 0
            
        print(f"[ZoneDrawer] Connecting to: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print("[ZoneDrawer] [ERROR] Could not connect to stream. Please check IP Webcam.")
            return False
        
        print("[ZoneDrawer] Connected! Grabbing frame...")
        time.sleep(1.0)  # Let camera adjust
        
        # Try a few times to get a good frame
        frame = None
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.2)
        
        cap.release()
        
        if frame is None:
            print("[ZoneDrawer] [ERROR] Failed to grab frame.")
            return False
            
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        print(f"[ZoneDrawer] [OK] Grabbed reference frame: {frame.shape}")
        return True

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append([x, y])
            self.update_display()

    def update_display(self):
        """Redraw all zones and current drawing state."""
        if self.frame is None:
            return
            
        self.display_frame = self.frame.copy()
        overlay = self.frame.copy()
        
        # 1. Draw completed zones
        for idx, z in enumerate(self.zones):
            pts = np.array(z["polygon"], dtype=np.int32)
            z_type = z.get("type", "Normal")
            color = self.color_normal if z_type == "Normal" else self.color_restricted
            
            # Fill polygon semi-transparently
            cv2.fillPoly(overlay, [pts], color)
            # Outline
            cv2.polylines(self.display_frame, [pts], True, color, 2)
            
            # Label
            centroid = pts.mean(axis=0).astype(int)
            label = z.get("label", f"Zone {idx+1}")
            cv2.putText(self.display_frame, f"{label} ({z_type})", (centroid[0]-20, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)

        # 2. Draw current polygon in progress
        if len(self.current_polygon) > 0:
            color = self.color_normal if self.current_type == "Normal" else self.color_restricted
            pts = np.array(self.current_polygon, dtype=np.int32)
            
            # Points
            for pt in self.current_polygon:
                cv2.circle(self.display_frame, tuple(pt), 4, color, -1)
                
            # Lines connecting points
            if len(self.current_polygon) > 1:
                cv2.polylines(self.display_frame, [pts], False, color, 2)
                
            # Line from last point to mouse (would need mousemove event, keeping it simple)

        # 3. Draw HUD/Instructions
        hud_h = 100
        cv2.rectangle(self.display_frame, (0, 0), (self.display_frame.shape[1], hud_h), (30, 30, 30), -1)
        
        cv2.putText(self.display_frame, "STSMIRS Zone Drawer", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        mode_text = f"Current Mode: {self.current_type} Zone"
        mode_color = self.color_normal if self.current_type == "Normal" else self.color_restricted
        cv2.putText(self.display_frame, mode_text, (300, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                    
        y = 50
        cv2.putText(self.display_frame, "Left Click: Add Point", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(self.display_frame, "[ n ] Switch to Normal Zone", (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_normal, 1)
        cv2.putText(self.display_frame, "[ ENTER ] Finish Polygon", (500, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y = 75
        cv2.putText(self.display_frame, "[ u ] Undo last point", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(self.display_frame, "[ r ] Switch to Restricted", (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_restricted, 1)
        cv2.putText(self.display_frame, "[ s ] SAVE configs", (500, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(self.display_frame, "[ c ] Clear all", (700, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(self.window_name, self.display_frame)

    def run(self, image_path=None):
        """Run the interactive drawing loop."""
        if image_path and os.path.exists(image_path):
            # Load from saved image file
            self.frame = cv2.imread(image_path)
            if self.frame is not None:
                self.display_frame = self.frame.copy()
                print(f"[ZoneDrawer] [OK] Loaded image: {image_path} ({self.frame.shape})")
            else:
                print(f"[ZoneDrawer] [ERROR] Could not read image: {image_path}")
                return
        elif not self.get_reference_frame():
            # Try fallback to webcam 0
            print("[ZoneDrawer] Trying laptop webcam (0) as fallback...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, self.frame = cap.read()
                cap.release()
                if ret:
                    self.display_frame = self.frame.copy()
                    print("[ZoneDrawer] Using webcam successful.")
                else:
                    print("[ZoneDrawer] [ERROR] Webcam failed too.")
                    return
            else:
                return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.update_display()
        
        print("\n--- INSTRUCTIONS ---")
        print("1. Click points on the image to draw a polygon.")
        print("2. Press ENTER or SPACE to finish the current polygon.")
        print("3. Press 'n' to draw a Normal zone (Green).")
        print("4. Press 'r' to draw a Restricted zone (Red).")
        print("5. Press 'u' to undo the last point clicked.")
        print("6. Press 'c' to clear EVERYTHING and start over.")
        print("7. Press 's' to SAVE to zones_config.json.")
        print("8. Press 'q' or 'ESC' to quit.\n")

        while True:
            key = cv2.waitKey(10) & 0xFF
            
            if key in [27, ord('q')]: # ESC or q
                break
                
            elif key == ord('n'):
                self.current_type = "Normal"
                self.update_display()
                print("[Mode] Switched to Normal Zone")
                
            elif key == ord('r'):
                self.current_type = "Restricted"
                self.update_display()
                print("[Mode] Switched to Restricted Zone")
                
            elif key == ord('u'):
                if len(self.current_polygon) > 0:
                    self.current_polygon.pop()
                    self.update_display()
                    
            elif key == ord('c'):
                self.zones = []
                self.current_polygon = []
                self.update_display()
                print("[Cleared] All zones removed.")
                
            elif key in [13, 32]: # ENTER or SPACE - Finish polygon
                if len(self.current_polygon) >= 3:
                    zone_id = f"ZONE-{chr(65 + len(self.zones))}" # A, B, C...
                    new_zone = {
                        "zone_id": zone_id,
                        "type": self.current_type,
                        "polygon": self.current_polygon.copy(),
                        "label": "Main Area" if self.current_type == "Normal" else "Restricted Area"
                    }
                    self.zones.append(new_zone)
                    self.current_polygon = []
                    self.update_display()
                    print(f"[Added] {new_zone['zone_id']} ({new_zone['type']}) with {len(new_zone['polygon'])} points.")
                else:
                    print("[Warning] A polygon needs at least 3 points!")
                    
            elif key == ord('s'):
                self.save_zones()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STSMIRS Zone Drawer")
    parser.add_argument("--image", default=None, help="Path to reference frame image (skip live camera)")
    args = parser.parse_args()
    
    drawer = ZoneDrawer()
    drawer.run(image_path=args.image)
