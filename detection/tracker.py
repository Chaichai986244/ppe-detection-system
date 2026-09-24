"""
Person Tracker — ByteTrack (Kalman + IoU) + DetectionsSmoother.
Uses supervision library for production-grade temporal consistency.
Falls back to simple IoU tracking if supervision is unavailable.
"""
import logging
import numpy as np

import config

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Production-grade person tracker using ByteTrack + temporal smoothing.
    Handles frame-to-frame ID consistency, bounding box stabilization.

    Interface is intentionally simple to match the rest of the pipeline:
        update(person_boxes) -> {track_id: {'box': [x1,y1,x2,y2], 'is_new': bool}}
    """

    def __init__(self, iou_threshold=None, max_lost_frames=None):
        self.iou_threshold = iou_threshold or config.TRACKER_IOU_THRESHOLD
        self.max_lost_frames = max_lost_frames or config.TRACKER_MAX_LOST_FRAMES
        self._byte_tracker = None
        self._smoother = None
        self._use_bytetrack = False
        self._frame_count = 0
        self._track_ages = {}  # {track_id: frame_count_since_first_seen}

        # Try to init ByteTrack
        try:
            from supervision.tracker.byte_tracker.core import ByteTrack
            self._byte_tracker = ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=60,            # ~2s at 30fps — keep tracks alive longer
                minimum_matching_threshold=0.30, # moderate matching to avoid ID switch
                frame_rate=30,
            )
            self._use_bytetrack = True
            logger.info("PersonTracker: ByteTrack initialized (buffer=60, match=0.30)")
        except ImportError:
            logger.warning("PersonTracker: ByteTrack not available, using simple IoU")

        # Try to init DetectionsSmoother
        try:
            from supervision.detection.tools.smoother import DetectionsSmoother
            self._smoother = DetectionsSmoother()
            logger.info(f"PersonTracker: DetectionsSmoother initialized (window={config.SMOOTHER_WINDOW})")
        except ImportError:
            logger.warning("PersonTracker: DetectionsSmoother not available")

        # Re-id: recently lost tracks for fast-motion recovery
        self._lost_tracks = {}  # {track_id: {'box': [], 'lost_frames': int, 'age': int}}
        self._last_box = {}     # {track_id: [x1,y1,x2,y2]} — last known box per track
        self._reid_distance = 250  # pixels — max distance to consider same person

        # Fallback simple IoU tracker
        self._simple_tracks = {}
        self._next_simple_id = 0
        self._prev_boxes = []

    def update(self, person_boxes):
        """
        Match new detections to existing tracks using ByteTrack + smoothing.

        Args:
            person_boxes: list of [x1, y1, x2, y2] boxes for current frame

        Returns:
            dict: {track_id: {'box': [x1,y1,x2,y2], 'is_new': bool}}
        """
        self._frame_count += 1

        if not person_boxes:
            if self._byte_tracker:
                # Feed empty to let tracker decay lost tracks
                import supervision as sv
                empty_det = sv.Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    confidence=np.empty(0, dtype=np.float32),
                    class_id=np.empty(0, dtype=np.int64),
                )
                self._byte_tracker.update_with_detections(empty_det)
            return {}

        if self._use_bytetrack:
            return self._update_bytetrack(person_boxes)
        else:
            return self._update_simple(person_boxes)

    # ── ByteTrack path ────────────────────────────────────────────

    def _update_bytetrack(self, person_boxes):
        import supervision as sv

        # Note which IDs were active last frame
        prev_active = set(self._track_ages.keys())

        # Build supervision Detections object
        boxes_np = np.array(person_boxes, dtype=np.float32)
        n = len(person_boxes)
        detections = sv.Detections(
            xyxy=boxes_np,
            confidence=np.full(n, 0.85, dtype=np.float32),
            class_id=np.zeros(n, dtype=np.int64),
        )

        # Update ByteTrack
        tracked = self._byte_tracker.update_with_detections(detections)

        # Apply DetectionsSmoother if available
        if self._smoother is not None:
            tracked = self._smoother.update_with_detections(tracked)

        # Build result dict
        result = {}
        now_active = set()
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid_int = int(tid)
                now_active.add(tid_int)
                box = tracked.xyxy[i].tolist()
                is_new = tid_int not in self._track_ages
                if is_new:
                    self._track_ages[tid_int] = 0
                self._track_ages[tid_int] += 1
                self._last_box[tid_int] = box  # save for re-id
                result[tid_int] = {'box': box, 'is_new': is_new}

        # ── Re-id: detect lost tracks for next frame ──────────────
        newly_lost = prev_active - now_active
        for lost_id in newly_lost:
            age = self._track_ages.get(lost_id, 0)
            if age >= 5:
                self._lost_tracks[lost_id] = {
                    'box': self._last_box.get(lost_id, [0, 0, 0, 0]),
                    'lost_frames': 0,
                    'age': age,
                }

        # ── Re-id: check if new tracks match recently-lost ones ──
        for tid in list(result.keys()):
            info = result[tid]
            if not info.get('is_new'):
                continue
            box = info['box']
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            best_match = None
            best_dist = self._reid_distance
            for lost_id, lost_info in list(self._lost_tracks.items()):
                if lost_info['lost_frames'] > 30:  # too long ago
                    del self._lost_tracks[lost_id]
                    continue
                lb = lost_info['box']
                lx = (lb[0] + lb[2]) / 2 if isinstance(lb, list) else cx + 1000
                ly = (lb[1] + lb[3]) / 2 if isinstance(lb, list) else cy + 1000
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match = lost_id

            if best_match is not None:
                # Merge: reuse lost ID
                lost_info = self._lost_tracks.pop(best_match)
                old_id = best_match
                # Replace result entry
                result[old_id] = {'box': box, 'is_new': False}  # not new, re-id'd
                del result[tid]
                self._track_ages[old_id] = lost_info['age'] + 1
                if tid in self._track_ages:
                    del self._track_ages[tid]
                logger.debug(f"Re-id: track {old_id} recovered (was lost {lost_info['lost_frames']} frames, dist={best_dist:.0f}px)")

        # ── Update lost track counters ────────────────────────────
        for lost_id in list(self._lost_tracks.keys()):
            self._lost_tracks[lost_id]['lost_frames'] += 1
            if self._lost_tracks[lost_id]['lost_frames'] > 30:
                del self._lost_tracks[lost_id]

        # Prune ages
        active_ids = set(result.keys())
        self._track_ages = {k: v for k, v in self._track_ages.items() if k in active_ids}

        return result

    # ── Simple IoU fallback ───────────────────────────────────────

    def _update_simple(self, person_boxes):
        """Simple IoU-based tracking (fallback when ByteTrack unavailable)."""
        from .tracker import _box_iou
        matched = {}

        if not self._simple_tracks:
            for box in person_boxes:
                tid = self._get_simple_id()
                self._simple_tracks[tid] = {'box': box, 'lost': 0, 'total_frames': 1}
                matched[tid] = {'box': box, 'is_new': True}
            return matched

        track_ids = list(self._simple_tracks.keys())
        num_tracks = len(track_ids)
        num_boxes = len(person_boxes)

        # Compute IoU matrix
        iou_matrix = np.zeros((num_tracks, num_boxes), dtype=np.float32)
        for ti, tid in enumerate(track_ids):
            for bi, box in enumerate(person_boxes):
                iou_matrix[ti, bi] = _box_iou(self._simple_tracks[tid]['box'], box)

        # Greedy matching
        matched_ti, matched_bi = set(), set()
        pairs = [(iou_matrix[ti, bi], ti, bi) for ti in range(num_tracks) for bi in range(num_boxes)
                 if iou_matrix[ti, bi] >= self.iou_threshold]
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, ti, bi in pairs:
            if ti not in matched_ti and bi not in matched_bi:
                tid = track_ids[ti]
                self._simple_tracks[tid]['box'] = person_boxes[bi]
                self._simple_tracks[tid]['lost'] = 0
                self._simple_tracks[tid]['total_frames'] += 1
                matched[tid] = {'box': person_boxes[bi], 'is_new': False}
                matched_ti.add(ti)
                matched_bi.add(bi)

        # New tracks for unmatched boxes
        for bi in range(num_boxes):
            if bi not in matched_bi:
                tid = self._get_simple_id()
                self._simple_tracks[tid] = {'box': person_boxes[bi], 'lost': 0, 'total_frames': 1}
                matched[tid] = {'box': person_boxes[bi], 'is_new': True}

        # Increment lost for unmatched tracks
        for ti in range(num_tracks):
            if ti not in matched_ti:
                tid = track_ids[ti]
                self._simple_tracks[tid]['lost'] += 1
                if self._simple_tracks[tid]['lost'] > self.max_lost_frames:
                    del self._simple_tracks[tid]

        return matched

    def _get_simple_id(self):
        tid = self._next_simple_id
        self._next_simple_id += 1
        return tid

    # ── Common ────────────────────────────────────────────────────

    def get_track_age(self, track_id):
        """How many frames a track has been active."""
        if self._use_bytetrack:
            return self._track_ages.get(track_id, 0)
        t = self._simple_tracks.get(track_id)
        return t['total_frames'] if t else 0

    def reset(self):
        """Clear all tracks."""
        self._track_ages.clear()
        self._lost_tracks.clear()
        self._last_box.clear()
        self._simple_tracks.clear()
        self._next_simple_id = 0
        self._frame_count = 0
        # Re-init ByteTrack
        try:
            from supervision.tracker.byte_tracker.core import ByteTrack
            self._byte_tracker = ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=60,
                minimum_matching_threshold=0.30,
                frame_rate=30,
            )
        except ImportError:
            pass
        try:
            from supervision.detection.tools.smoother import DetectionsSmoother
            self._smoother = DetectionsSmoother()
        except ImportError:
            pass

    @property
    def active_count(self):
        if self._use_bytetrack:
            return len(self._track_ages)
        return sum(1 for t in self._simple_tracks.values() if t['lost'] == 0)


def _box_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
