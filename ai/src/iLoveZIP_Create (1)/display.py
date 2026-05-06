"""
STSMIRS — Display / Overlay Engine
Renders bounding boxes, tracking IDs, zone labels, event alerts,
health/crime score bars, and emergency banners on frames.

Usage (standalone test):
    python src/display.py
    python src/display.py --source 0
"""

import cv2
import json
import time
import numpy as np


class DisplayEngine:
    """Renders all STSMIRS overlays on video frames."""

    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)

        disp_cfg = config.get("display", {})
        self.window_name = disp_cfg.get("window_name", "STSMIRS Live Feed")
        self.show_fps = disp_cfg.get("show_fps", True)
        self.show_zones = disp_cfg.get("show_zones", True)
        self.show_scores = disp_cfg.get("show_scores", True)
        self.show_alerts = disp_cfg.get("show_alerts", True)
        self.alert_duration = disp_cfg.get("alert_display_duration_sec", 5)
        self.emergency_duration = disp_cfg.get("emergency_flash_duration_sec", 5)
        self.font_scale = disp_cfg.get("font_scale", 0.6)
        self.font_face = cv2.FONT_HERSHEY_DUPLEX
        self.font_face_secondary = cv2.FONT_HERSHEY_SIMPLEX
        self.box_thickness = disp_cfg.get("box_thickness", 2)

        # Colors (BGR)
        colors = disp_cfg.get("colors", {})
        self.color_normal = tuple(colors.get("normal_box", [0, 255, 0]))
        self.color_alert = tuple(colors.get("alert_box", [0, 0, 255]))
        self.color_text = tuple(colors.get("text", [255, 255, 255]))
        self.color_alert_banner = tuple(colors.get("alert_banner", [0, 0, 200]))
        self.color_emergency = tuple(colors.get("emergency_banner", [0, 0, 255]))

        # Zone colors
        zone_cfg = config.get("zones", {})
        zone_colors = zone_cfg.get("colors", {})
        self.zone_overlay_alpha = zone_cfg.get("overlay_alpha", 0.3)
        self.zone_colors = {
            "Normal": tuple(zone_colors.get("Normal", [0, 255, 0])),
            "Restricted": tuple(zone_colors.get("Restricted", [0, 0, 255]))
        }

        # Active alerts (auto-expire)
        self._active_alerts = []  # list of (message, expire_time, level)
        self._pending_notices = []  # list of {message, level, person, event, prompt_until}
        self._emergency_active = False
        self._emergency_expire = 0
        self._emergency_message = ""

        # Track ID color palette (consistent colors per person)
        self._id_colors = [
            (0, 255, 0), (255, 178, 0), (0, 178, 255), (255, 0, 178),
            (178, 255, 0), (0, 255, 255), (255, 0, 255), (178, 0, 255),
            (255, 255, 0), (0, 255, 178), (255, 178, 178), (178, 255, 255),
        ]

        self.panel_bg = (16, 18, 22)
        self.panel_border = (78, 86, 98)
        self.panel_text = (242, 245, 247)
        self.panel_muted = (166, 174, 184)
        self.panel_accent = (0, 196, 255)
        self.panel_shadow = (8, 10, 12)

    # ─── Person Overlays ──────────────────────────────────────

    def draw_person(self, frame, person, person_count=0):
        """
        Draw bounding box, tracking ID, and scores for one tracked person.

        Args:
            frame: BGR image (modified in-place)
            person: TrackedPerson object (from tracker.py)
        """
        x1, y1, x2, y2 = person.bbox
        is_alert = getattr(person, "is_alert", False)
        box_color = self.color_alert if is_alert else self._get_id_color(person.track_id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self.box_thickness)

        # Compact name label inside the upper part of the person box, near the face area.
        name = getattr(person, "track_label", f"T-{person.track_id:03d}")
        confidence_pct = self._format_percent(getattr(person, "action_confidence", 0.0))
        box_label = f"{name}  {confidence_pct}"
        self._draw_label(frame, box_label, (x1 + 4, y1 + 6), box_color, above=False, compact=True)

        # Zone label (if assigned)
        if person.zone_id:
            zone_text = self._clean_text(person.zone_id).upper()
            self._draw_label(frame, zone_text, (max(x1 + 6, x2 - 104), y1 - 6), (255, 200, 0), above=True, compact=True)

        # Foot point
        cv2.circle(frame, person.foot_point, 4, (0, 0, 255), -1)

        # Score bars (if enabled)
        if self.show_scores:
            self._draw_score_bars(frame, person, x1, y1, x2)

    def draw_persons(self, frame, persons):
        """Draw all tracked persons."""
        person_count = len(persons)
        for person in persons:
            self.draw_person(frame, person, person_count=person_count)

    def draw_person_info_panel(self, frame, persons):
        """Draw ordered person info in a fixed corner: name and primary action only."""
        if not persons:
            return

        h, w = frame.shape[:2]
        panel_w = min(360, max(260, w // 4))
        row_h = 34
        header_h = 34
        rows = min(len(persons), 4)
        panel_h = header_h + rows * row_h + 12
        x0 = 14
        y0 = max(52, h - panel_h - 36)

        self._draw_glass_panel(frame, x0, y0, panel_w, panel_h, accent=self.panel_accent, accent_width=3, alpha=0.74)

        cv2.putText(frame, "LIVE SUBJECTS", (x0 + 14, y0 + 22),
                    self.font_face_secondary, 0.50, self.panel_accent, 1, cv2.LINE_AA)

        for idx, person in enumerate(persons[:rows]):
            name = self._clean_text(getattr(person, "track_label", f"T-{person.track_id:03d}"))
            action = self._clean_text(getattr(person, "event_type", None) or "Unknown")
            action_conf = self._format_percent(getattr(person, "action_confidence", 0.0))
            left = self._fit_text(name.upper(), int(panel_w * 0.44), 0.60, 1)
            center = self._fit_text(f"{action.replace('_', ' ').upper()}  {action_conf}", int(panel_w * 0.46), 0.52, 1)
            y = y0 + header_h + 22 + idx * row_h
            if idx > 0:
                cv2.line(frame, (x0 + 14, y - 16), (x0 + panel_w - 14, y - 16), (44, 48, 54), 1)
            cv2.putText(frame, left, (x0 + 14, y),
                        self.font_face, 0.60, self.panel_text, 1, cv2.LINE_AA)
            cv2.putText(frame, center, (x0 + int(panel_w * 0.44), y),
                        self.font_face, 0.52, self.panel_muted, 1, cv2.LINE_AA)

    def draw_risk_panel(self, frame, persons):
        """Draw a separate highlighted panel for risky action percentages."""
        if not persons:
            return

        h, w = frame.shape[:2]
        panel_w = min(460, max(360, w // 3))
        metric_gap = 42
        subject_block_h = 156
        header_h = 42
        rows = min(len(persons), 2)
        panel_h = header_h + rows * subject_block_h + 18
        x0 = w - panel_w - 16
        y0 = h - panel_h - 42

        self._draw_glass_panel(frame, x0, y0, panel_w, panel_h, accent=(255, 186, 0), accent_width=4, alpha=0.76)
        cv2.putText(frame, "RISK INDICATORS", (x0 + 18, y0 + 28),
                    self.font_face, 0.68, (255, 210, 90), 2, cv2.LINE_AA)
        cv2.putText(frame, "Threat confidence overview", (x0 + 18, y0 + 40),
                    self.font_face_secondary, 0.42, self.panel_muted, 1, cv2.LINE_AA)

        for idx, person in enumerate(persons[:rows]):
            probs = getattr(person, "action_probabilities", {}) or {}
            subject = self._clean_text(getattr(person, "track_label", f"T-{person.track_id:03d}")).upper()
            base_y = y0 + header_h + 22 + idx * subject_block_h
            if idx > 0:
                cv2.line(frame, (x0 + 18, base_y - 24), (x0 + panel_w - 18, base_y - 24), (44, 48, 54), 1)

            cv2.putText(frame, subject, (x0 + 18, base_y),
                        self.font_face, 0.72, self.panel_text, 2, cv2.LINE_AA)
            cv2.putText(frame, "Live action risk levels", (x0 + 18, base_y + 18),
                        self.font_face_secondary, 0.42, self.panel_muted, 1, cv2.LINE_AA)

            metric_y = base_y + 42
            self._draw_metric_bar(frame, x0 + 18, metric_y, panel_w - 36, "FIGHTING", probs.get("Fighting", 0.0), (80, 110, 255))
            self._draw_metric_bar(frame, x0 + 18, metric_y + metric_gap, panel_w - 36, "PANIC", probs.get("Panic", 0.0), (0, 210, 255))
            self._draw_metric_bar(frame, x0 + 18, metric_y + metric_gap * 2, panel_w - 36, "FALL", probs.get("Fall", 0.0), (0, 190, 120))

    # ─── Score Bars ───────────────────────────────────────────

    def _draw_score_bars(self, frame, person, x1, y1, x2):
        """Draw health and crime score bars on the right side of bounding box."""
        bar_width = 6
        bar_max_height = min(60, (person.bbox[3] - person.bbox[1]) // 2)
        bar_x = x2 + 4

        health = getattr(person, "health_score", 100)
        crime = getattr(person, "crime_score", 0)

        # Health bar (green → red as it drops)
        h_height = int(bar_max_height * (health / 100))
        h_color = self._score_to_color(health, high_is_good=True)
        bar_top = y1
        cv2.rectangle(frame, (bar_x, bar_top + bar_max_height - h_height),
                      (bar_x + bar_width, bar_top + bar_max_height), h_color, -1)
        cv2.rectangle(frame, (bar_x, bar_top),
                      (bar_x + bar_width, bar_top + bar_max_height), (100, 100, 100), 1)

        # Crime bar (green → red as it rises)
        c_height = int(bar_max_height * (crime / 100))
        c_color = self._score_to_color(crime, high_is_good=False)
        c_bar_x = bar_x + bar_width + 3
        cv2.rectangle(frame, (c_bar_x, bar_top + bar_max_height - c_height),
                      (c_bar_x + bar_width, bar_top + bar_max_height), c_color, -1)
        cv2.rectangle(frame, (c_bar_x, bar_top),
                      (c_bar_x + bar_width, bar_top + bar_max_height), (100, 100, 100), 1)

        # Tiny labels
        cv2.putText(frame, "H", (bar_x, bar_top - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(frame, "C", (c_bar_x, bar_top - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    @staticmethod
    def _score_to_color(score, high_is_good=True):
        """Convert 0-100 score to BGR color (green=good, red=bad)."""
        if high_is_good:
            ratio = score / 100.0
        else:
            ratio = 1.0 - (score / 100.0)
        # Green (0,255,0) → Yellow (0,255,255) → Red (0,0,255)
        if ratio > 0.5:
            g = 255
            r = int(255 * (1 - ratio) * 2)
        else:
            g = int(255 * ratio * 2)
            r = 255
        return (0, g, r)

    # ─── Zone Overlays ────────────────────────────────────────

    def draw_zones(self, frame, zones):
        """
        Draw semi-transparent zone polygons on the frame.

        Args:
            frame: BGR image (modified in-place)
            zones: list of zone dicts from zones_config.json
                   Each: {"zone_id": str, "type": str, "polygon": [[x,y],...], "label": str}
        """
        if not self.show_zones or not zones:
            return

        overlay = frame.copy()
        for zone in zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            zone_type = zone.get("type", "Normal")
            color = self.zone_colors.get(zone_type, (0, 255, 0))

            # Fill polygon
            cv2.fillPoly(overlay, [pts], color)

            # Zone label at centroid
            centroid = pts.mean(axis=0).astype(int)
            label = zone.get("label", zone.get("zone_id", "ZONE"))
            zone_label = f"{label} ({zone_type})"
            label_size, _ = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                          (centroid[0] - 4, centroid[1] - label_size[1] - 4),
                          (centroid[0] + label_size[0] + 4, centroid[1] + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, zone_label,
                        (centroid[0], centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Blend overlay
        cv2.addWeighted(overlay, self.zone_overlay_alpha, frame,
                        1 - self.zone_overlay_alpha, 0, frame)

        # Draw polygon outlines (on top, not transparent)
        for zone in zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            zone_type = zone.get("type", "Normal")
            color = self.zone_colors.get(zone_type, (0, 255, 0))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    # ─── Alert Banners ────────────────────────────────────────

    def add_alert(self, message, level="warning", person_label=None, event_type=None, delay_before_danger=15):
        """
        Add an alert to display on screen.
        Level: 'warning' (yellow), 'danger' (red), 'info' (blue)
        """
        if level == "danger":
            if person_label is None or event_type is None:
                parsed_event, parsed_person = self._parse_alert_message(message)
                person_label = person_label or parsed_person
                event_type = event_type or parsed_event
            self._pending_notices.append({
                "message": message,
                "level": level,
                "person_label": person_label,
                "event_type": event_type,
                "prompt_until": time.time() + delay_before_danger,
            })
            return

        expire = time.time() + self.alert_duration
        self._active_alerts.append((message, expire, level))

    def acknowledge_notice(self):
        """Mark all pending notice prompts as acknowledged and remove them."""
        if self._pending_notices:
            self._pending_notices.clear()
            self._active_alerts.append(("Acknowledged. No red alert raised.", time.time() + 2, "info"))

    def trigger_emergency(self, message=None):
        """Trigger the emergency banner (blockchain simulation)."""
        self._emergency_active = True
        self._emergency_expire = time.time() + self.emergency_duration
        self._emergency_message = message or \
            "EMERGENCY >>> Smart Contract Called >>> ADM Identity Release"

    def draw_alerts(self, frame):
        """Draw active alerts as a compact corner panel."""
        now = time.time()

        # Clean expired alerts
        self._active_alerts = [(m, e, l) for m, e, l in self._active_alerts if e > now]
        escalated = []
        still_pending = []
        for notice in self._pending_notices:
            if now >= notice["prompt_until"]:
                escalated.append(notice)
            else:
                still_pending.append(notice)
        self._pending_notices = still_pending
        for notice in escalated:
            expire = now + self.alert_duration
            self._active_alerts.append((notice["message"], expire, "danger"))

        h, w = frame.shape[:2]
        panel_w = min(460, max(280, w // 3))
        x0 = w - panel_w - 14
        y0 = 52
        row_h = 30

        pending_rows = self._pending_notices[:2]
        for idx, notice in enumerate(pending_rows):
            remaining = max(0, int(round(notice["prompt_until"] - now)))
            y = y0 + idx * (row_h + 8)
            self._draw_glass_panel(frame, x0, y, panel_w, row_h, accent=(0, 200, 255), accent_width=3, alpha=0.78)

            person_label = self._clean_text(notice.get("person_label") or "Person")
            event_type = self._clean_text(notice.get("event_type") or "Event")
            text = f"NOTICE  {person_label.upper()}  CONFIRM {event_type.upper()}  Y / {remaining:02d}S"
            text = self._fit_text(text, panel_w - 18, 0.50, 1)
            cv2.putText(frame, text, (x0 + 12, y + 21),
                        self.font_face_secondary, 0.48, self.panel_text, 1, cv2.LINE_AA)

        # Regular alerts in top-right corner
        alerts_y0 = y0 + len(pending_rows) * (row_h + 8)
        for idx, (message, expire, level) in enumerate(self._active_alerts[:4]):
            if level == "danger":
                accent = (70, 70, 230)
                title = "ALERT"
            elif level == "warning":
                accent = (0, 170, 255)
                title = "WARNING"
            else:
                accent = (255, 180, 80)
                title = "INFO"

            y = alerts_y0 + idx * (row_h + 8)
            self._draw_glass_panel(frame, x0, y, panel_w, row_h, accent=accent, accent_width=3, alpha=0.76)

            clean_message = self._clean_text(message)
            text = f"{title}  {clean_message.upper()}"
            text = self._fit_text(text, panel_w - 18, 0.50, 1)
            cv2.putText(frame, text, (x0 + 12, y + 21),
                        self.font_face_secondary, 0.48, self.panel_text, 1, cv2.LINE_AA)

        # Emergency banner (full-width flashing)
        if self._emergency_active:
            if now > self._emergency_expire:
                self._emergency_active = False
            else:
                # Flashing effect (alternates every 0.5s)
                flash = int(now * 2) % 2 == 0
                if flash:
                    bar_h = 50
                    y_pos = h // 2 - bar_h // 2

                    cv2.rectangle(frame, (0, y_pos), (w, y_pos + bar_h),
                                  (12, 18, 32), -1)
                    cv2.rectangle(frame, (0, y_pos), (w, y_pos + 4),
                                  (0, 0, 220), -1)
                    cv2.rectangle(frame, (2, y_pos + 2), (w - 2, y_pos + bar_h - 2),
                                  (70, 70, 220), 1)

                    text = self._emergency_message
                    text_size, _ = cv2.getTextSize(text, self.font_face, 0.7, 2)
                    tx = (w - text_size[0]) // 2
                    cv2.putText(frame, text, (tx, y_pos + 33),
                                self.font_face, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ─── HUD (Heads-Up Display) ──────────────────────────────

    def draw_hud(self, frame, fps=0.0, person_count=0, extra_info=None):
        """Draw the top HUD bar with FPS, person count, and status."""
        h, w = frame.shape[:2]

        # Top bar background
        top_h = 44
        self._draw_glass_panel(frame, 0, 0, w, top_h, accent=None, accent_width=0, alpha=0.68)

        # Title
        cv2.putText(frame, "STSMIRS SURVEILLANCE FEED", (14, 28),
                    self.font_face_secondary, 0.62, self.panel_accent, 1, cv2.LINE_AA)

        # FPS
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            fps_color = (0, 255, 0) if fps >= 10 else (0, 165, 255) if fps >= 5 else (0, 0, 255)
            cv2.putText(frame, fps_text, (w - 142, 28),
                        self.font_face_secondary, 0.54, fps_color, 1, cv2.LINE_AA)

        # Person count
        count_text = f"SUBJECTS {person_count}"
        cv2.putText(frame, count_text, (w - 290, 28),
                    self.font_face_secondary, 0.54, self.panel_text, 1, cv2.LINE_AA)

        # Extra info
        if extra_info:
            cv2.putText(frame, extra_info, (300, 26),
                        self.font_face, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom bar
        bottom_h = 24
        self._draw_glass_panel(frame, 0, h - bottom_h, w, bottom_h, accent=None, accent_width=0, alpha=0.60)
        cv2.putText(frame, "Y ACKNOWLEDGE   Q EXIT   E EMERGENCY TEST",
                    (12, h - 7), self.font_face_secondary, 0.42, self.panel_muted, 1, cv2.LINE_AA)

    # ─── Full Frame Render ────────────────────────────────────

    def render(self, frame, persons=None, zones=None, fps=0.0):
        """
        Render all overlays on a frame. This is the main method to call.

        Args:
            frame: BGR image (will be copied, original not modified)
            persons: list of TrackedPerson objects
            zones: list of zone dicts from zones_config.json
            fps: current FPS value

        Returns:
            annotated frame (BGR)
        """
        output = frame.copy()

        # Layer 1: Zone overlays (lowest)
        if zones:
            self.draw_zones(output, zones)

        # Layer 2: Person overlays
        if persons:
            self.draw_persons(output, persons)
            self.draw_person_info_panel(output, persons)
            self.draw_risk_panel(output, persons)

        # Layer 3: Alert banners
        if self.show_alerts:
            self.draw_alerts(output)

        # Layer 4: HUD (topmost)
        person_count = len(persons) if persons else 0
        self.draw_hud(output, fps=fps, person_count=person_count)

        return output

    # ─── Helpers ──────────────────────────────────────────────

    def _get_id_color(self, track_id):
        """Get consistent color for a track ID."""
        return self._id_colors[track_id % len(self._id_colors)]

    def _draw_label(self, frame, text, position, color, above=True, compact=False):
        """Draw a text label with background."""
        x, y = position
        text = self._clean_text(text)
        scale = 0.46 if compact else self.font_scale
        thickness = 1
        padding_x = 8 if compact else 6
        padding_y = 6 if compact else 8
        size, _ = cv2.getTextSize(text, self.font_face_secondary if compact else self.font_face, scale, thickness)
        tw, th = size

        if above:
            y1 = y - th - padding_y
            self._draw_glass_panel(frame, x, y1, tw + padding_x, th + padding_y, accent=color, accent_width=2, alpha=0.72)
            cv2.putText(frame, text, (x + 5, y - 4),
                        self.font_face_secondary if compact else self.font_face, scale, self.panel_text, thickness, cv2.LINE_AA)
        else:
            self._draw_glass_panel(frame, x, y, tw + padding_x, th + padding_y, accent=color, accent_width=2, alpha=0.74)
            cv2.putText(frame, text, (x + 5, y + th + 1),
                        self.font_face_secondary if compact else self.font_face, scale, self.panel_text, thickness, cv2.LINE_AA)

    def _draw_glass_panel(self, frame, x, y, w, h, accent=None, accent_width=3, alpha=0.74):
        """Draw a cinematic semi-transparent panel with subtle border and shadow."""
        fh, fw = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        if w <= 0 or h <= 0:
            return

        shadow_x1 = min(fw - 1, x + 3)
        shadow_y1 = min(fh - 1, y + 3)
        shadow_x2 = min(fw, x + w + 3)
        shadow_y2 = min(fh, y + h + 3)
        if shadow_x2 > shadow_x1 and shadow_y2 > shadow_y1:
            shadow = frame[shadow_y1:shadow_y2, shadow_x1:shadow_x2].copy()
            cv2.rectangle(shadow, (0, 0), (shadow_x2 - shadow_x1, shadow_y2 - shadow_y1), self.panel_shadow, -1)
            cv2.addWeighted(shadow, 0.20, frame[shadow_y1:shadow_y2, shadow_x1:shadow_x2], 0.80, 0,
                            frame[shadow_y1:shadow_y2, shadow_x1:shadow_x2])

        panel = frame[y:y + h, x:x + w].copy()
        cv2.rectangle(panel, (0, 0), (w, h), self.panel_bg, -1)
        cv2.addWeighted(panel, alpha, frame[y:y + h, x:x + w], 1.0 - alpha, 0, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.panel_border, 1)
        if accent is not None and accent_width > 0:
            cv2.rectangle(frame, (x, y), (x + accent_width, y + h), accent, -1)

    def _draw_metric_bar(self, frame, x, y, width, label, value, accent):
        """Draw a bold cinematic metric bar with percentage."""
        value = max(0.0, min(1.0, float(value)))
        label_w = 132
        value_w = 76
        track_x = x + label_w
        track_w = width - label_w - value_w
        bar_y = y + 9
        bar_h = 14
        fill_w = max(2, int(track_w * value)) if track_w > 0 else 0

        cv2.putText(frame, label, (x, y),
                    self.font_face, 0.58, self.panel_text, 2, cv2.LINE_AA)
        cv2.putText(frame, self._format_percent(value), (x + width - value_w + 8, y),
                    self.font_face, 0.58, accent, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (track_x, bar_y), (track_x + track_w, bar_y + bar_h), (38, 42, 48), -1)
        cv2.rectangle(frame, (track_x, bar_y), (track_x + track_w, bar_y + bar_h), (70, 76, 82), 1)
        cv2.rectangle(frame, (track_x, bar_y), (track_x + fill_w, bar_y + bar_h), accent, -1)

    @staticmethod
    def _clean_text(text):
        """Keep overlay text ASCII-friendly because OpenCV Hershey fonts do not render Unicode."""
        text = str(text)
        replacements = {
            "⚠": "ALERT",
            "—": "-",
            "–": "-",
            "âš ": "ALERT",
            "â€”": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.encode("ascii", errors="ignore").decode("ascii")

    def _fit_text(self, text, max_width, scale, thickness):
        """Trim a string so it fits in the requested pixel width."""
        text = self._clean_text(text)
        if cv2.getTextSize(text, self.font_face, scale, thickness)[0][0] <= max_width:
            return text

        ellipsis = "..."
        while text and cv2.getTextSize(text + ellipsis, self.font_face, scale, thickness)[0][0] > max_width:
            text = text[:-1]
        return text + ellipsis if text else ellipsis

    def _parse_alert_message(self, message):
        """Best-effort parser for existing alert strings."""
        clean = self._clean_text(message)
        parts = [part.strip() for part in clean.replace("-", "|").split("|") if part.strip()]
        event_type = parts[0].title() if parts else "Event"
        person_label = parts[1] if len(parts) > 1 else "Person"
        return event_type, person_label

    @staticmethod
    def _format_percent(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(1.0, value))
        return f"{int(round(value * 100.0)):02d}%"


# ═══════════════════════════════════════════════════════════════
#  STANDALONE TEST — Run: python src/display.py
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="STSMIRS Display Engine Test")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--source", default=None,
                        help="Video source: URL, file path, or '0' for webcam")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        exit(1)

    # Import tracker
    from src.tracker import PersonTracker

    # Load zone config
    with open(args.config, "r") as f:
        cfg = json.load(f)
    zones_file = cfg.get("zones", {}).get("config_file", "zones_config.json")
    zones = []
    if os.path.exists(zones_file):
        with open(zones_file, "r") as f:
            zones = json.load(f).get("zones", [])
        print(f"[Display] Loaded {len(zones)} zones from {zones_file}")

    # Initialize
    tracker = PersonTracker(args.config)
    display = DisplayEngine(args.config)

    # Video source
    if args.source:
        source = 0 if args.source == "0" else args.source
    else:
        source = cfg["stream"]["url"]

    print("=" * 50)
    print("  STSMIRS — Display Engine Test")
    print(f"  Source: {source}")
    print("  Keys: 'q' quit | 'e' emergency | 'a' alert")
    print("=" * 50)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[WARN] Could not open {source}, trying webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] No video source.")
            exit(1)

    fps_time = time.time()
    fps_count = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        # Run tracker
        persons = tracker.update(frame)

        # Simulate some scores for demo (in real pipeline, event_trigger sets these)
        for p in persons:
            p.health_score = max(30, 100 - (p.track_id * 15) % 70)
            p.crime_score = min(60, (p.track_id * 10) % 50)

        # Render everything
        output = display.render(frame, persons=persons, zones=zones, fps=current_fps)

        # FPS
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time = time.time()

        cv2.imshow(display.window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            display.trigger_emergency()
            print("[Test] Emergency triggered!")
        elif key == ord('a'):
            display.add_alert("FALL DETECTED - T-001 - ZONE-A - 97%", level="danger")
            print("[Test] Alert added!")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Display test complete.")
