"""
Annotator - Bounding box + overlay text renderer for monitoring frames.
Color scheme:
  Green  (0,255,0)   : fully compliant (Hardhat + Safety Vest)
  Orange (0,165,255) : missing ONE critical item (Hardhat or Safety Vest)
  Red    (0,0,255)   : missing BOTH critical items / no protection
"""
import cv2
import numpy as np


class Annotator:
    """Draws detection boxes and overlay on frames."""

    # BGR colors
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_CYAN = (255, 255, 0)

    PPE_COLORS = {
        'Helmet': (0, 255, 0),           # Green
        'Safety Vest': (0, 255, 255),    # Yellow
    }

    def __init__(self, font_scale=0.45, thickness=2, text_thickness=1):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.text_thickness = text_thickness

    def _person_box_color(self, ppe_status):
        """
        Determine person bounding box color based on PPE compliance.
        ppe_status: dict like {'Hardhat': True, 'Safety Vest': False}
        """
        has_helmet = ppe_status.get('Helmet', False)
        has_vest = ppe_status.get('Safety Vest', False)

        if has_helmet and has_vest:
            return self.COLOR_GREEN   # Fully compliant
        elif has_helmet or has_vest:
            return self.COLOR_ORANGE  # Missing ONE critical item
        else:
            return self.COLOR_RED     # Missing BOTH critical items → danger

    def annotate(self, frame, person_boxes, ppe_statuses, ppe_boxes,
                 alarm_level=0, total_persons=0, violation_count=0,
                 scene_name='', location='', track_ids=None, track_ages=None,
                 entry_exit_counts=None):
        """
        Draw all annotations on a copy of the frame.
        Returns annotated frame.

        Args:
            track_ids: optional list of track IDs matching person_boxes
            track_ages: optional list of track ages (frames) for stabilization display
        """
        import config as cfg
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Draw PPE item boxes (thin lines) ─────────────────────────
        for ppe_type, (x1, y1, x2, y2) in ppe_boxes:
            is_violation = ppe_type.startswith('NO-')
            color = self.PPE_COLORS.get(ppe_type, self.COLOR_WHITE)
            thickness = 2 if is_violation else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = ppe_type.replace('NO-', '!')[:8]
            cv2.putText(annotated, label, (x1, y1 - 4),
                        self.font, 0.35, color, 1)

        # ── Draw person boxes (thicker lines, color-coded) ───────────
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            ppe_status = ppe_statuses.get(i, {})
            age = track_ages[i] if track_ages and i < len(track_ages) else 999
            is_new = age < cfg.PERSON_MIN_TRACK_AGE  # still stabilizing

            # Border style: dashed for new, solid for stable
            if is_new:
                color = self.COLOR_CYAN  # blue-ish — still tracking
                thickness = 1
            else:
                color = self._person_box_color(ppe_status)
                thickness = self.thickness

            # Draw person box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Compose status label
            if is_new:
                tid = track_ids[i] if track_ids and i < len(track_ids) else i + 1
                label = f"T{tid} [NEW {age}/{cfg.PERSON_MIN_TRACK_AGE}]"
            else:
                status_parts = []
                if ppe_status.get('Helmet', False):
                    status_parts.append('H')
                else:
                    status_parts.append('-')
                if ppe_status.get('Safety Vest', False):
                    status_parts.append('V')
                else:
                    status_parts.append('-')

                tid_label = f"T{track_ids[i]}" if track_ids and i < len(track_ids) else f"P{i+1}"
                label = f"{tid_label} [{''.join(status_parts)}]"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, self.text_thickness)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        self.font, self.font_scale, self.COLOR_WHITE, self.text_thickness)

        # ── Counting line (entry/exit gate) ────────────────────────
        if entry_exit_counts:
            line_y = int(h * 0.40)
            for x in range(0, w, 16):
                cv2.line(annotated, (x, line_y), (min(x+8, w), line_y),
                         (0, 200, 200), 1)
            counts = entry_exit_counts
            cv2.putText(annotated, f"IN:{counts['in']} OUT:{counts['out']} CUR:{counts['current']}",
                        (w-260, line_y-8), self.font, 0.4, (0, 220, 220), 1)

        # ── Center ROI indicator ────────────────────────────────────
        if cfg.CENTER_ONLY:
            roi_m = cfg.CENTER_ROI_MARGIN
            rx1, ry1 = int(w * roi_m), int(h * roi_m)
            rx2, ry2 = int(w * (1 - roi_m)), int(h * (1 - roi_m))
            # Dashed yellow border
            for x in range(rx1, rx2, 16):
                cv2.line(annotated, (x, ry1), (min(x+8, rx2), ry1), (0, 220, 220), 1)
                cv2.line(annotated, (x, ry2), (min(x+8, rx2), ry2), (0, 220, 220), 1)
            for y in range(ry1, ry2, 16):
                cv2.line(annotated, (rx1, y), (rx1, min(y+8, ry2)), (0, 220, 220), 1)
                cv2.line(annotated, (rx2, y), (rx2, min(y+8, ry2)), (0, 220, 220), 1)
            cv2.putText(annotated, "ROI", (rx1+4, ry2-6), self.font, 0.35, (0, 180, 180), 1)

        # ── Top overlay bar ──────────────────────────────────────────
        bar_height = 40
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), self.COLOR_BLACK, -1)
        annotated = cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0)

        # Row 1: Scene | Location
        if scene_name:
            scene_text = f"{scene_name} / {location}" if location else scene_name
            cv2.putText(annotated, scene_text, (8, 16),
                        self.font, 0.4, self.COLOR_WHITE, 1)

        # Row 2: Persons | Violations | Alarm status
        y2 = 36
        cv2.putText(annotated, f"PERSONS: {total_persons}",
                    (8, y2), self.font, 0.55, self.COLOR_CYAN, 2)

        v_color = self.COLOR_RED if violation_count > 0 else self.COLOR_GREEN
        cv2.putText(annotated, f"VIOLATIONS: {violation_count}",
                    (170, y2), self.font, 0.55, v_color, 2)

        # Alarm level badge
        if alarm_level == 0:
            alarm_text = "STATUS: NORMAL"
            alarm_color = self.COLOR_GREEN
        elif alarm_level == 1:
            alarm_text = "ALARM: L1 WARNING"
            alarm_color = self.COLOR_YELLOW
        else:
            alarm_text = "ALARM: L2 CRITICAL"
            alarm_color = self.COLOR_RED

        cv2.putText(annotated, alarm_text,
                    (w - 330, y2), self.font, 0.55, alarm_color, 2)

        # ── Alarm popup overlay (bottom-center) ──────────────────────
        if alarm_level >= 2:
            popup_text = "⚠ WARNING: CRITICAL PPE VIOLATION DETECTED!"
            popup_w = 500
            popup_h = 40
            px = (w - popup_w) // 2
            py = h - 60
            cv2.rectangle(annotated, (px, py), (px + popup_w, py + popup_h),
                          (0, 0, 200), -1)
            cv2.rectangle(annotated, (px, py), (px + popup_w, py + popup_h),
                          self.COLOR_RED, 2)
            (tw, th), _ = cv2.getTextSize(popup_text, self.font, 0.55, 1)
            cv2.putText(annotated, popup_text,
                        (px + (popup_w - tw) // 2, py + th + 10),
                        self.font, 0.55, self.COLOR_WHITE, 1)

        elif alarm_level == 1:
            popup_text = "⚠ NOTICE: Missing recommended PPE (Safety Vest)"
            popup_w = 460
            popup_h = 32
            px = (w - popup_w) // 2
            py = h - 55
            cv2.rectangle(annotated, (px, py), (px + popup_w, py + popup_h),
                          (0, 140, 140), -1)
            cv2.rectangle(annotated, (px, py), (px + popup_w, py + popup_h),
                          self.COLOR_YELLOW, 2)
            (tw, th), _ = cv2.getTextSize(popup_text, self.font, 0.45, 1)
            cv2.putText(annotated, popup_text,
                        (px + (popup_w - tw) // 2, py + th + 8),
                        self.font, 0.45, self.COLOR_WHITE, 1)

        return annotated
