"""
Detection Engine - YOLO model loader + inference loop
Runs in a daemon thread, processes frames and emits results via callback.
"""
import os
import time
import threading
import queue
import logging
import cv2
import numpy as np
from ultralytics import YOLO

import config
from .annotator import Annotator
from .tracker import PersonTracker
from .temporal_voter import TemporalVoter
from .preprocessor import (preprocess, tta_inference, penalize_dark_helmets,
                            diou_nms, temperature_scale, box_ema_smooth)
from .ppe_association import associate_ppe, compute_violations_per_person
from alarm.rules import (determine_frame_alarm, should_screenshot,
                         get_violation_display_name)
from alarm.logger import AlarmLogger
from notification.dingtalk import DingTalkNotifier

logger = logging.getLogger(__name__)


class DetectionEngine:
    """YOLO-based PPE detection engine running in a background thread."""

    def __init__(self, model_path=config.MODEL_PATH, device=config.DEVICE):
        self.model_path = model_path
        self.device = device
        self.model = None  # Ultralytics model (RT-DETR/YOLO)
        self.frcnn = None  # Faster R-CNN model (torchvision)
        self.model_type = config.MODELS[config.ACTIVE_MODEL]['type']
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Dynamic settings
        self.confidence = config.CONFIDENCE_THRESHOLD
        self.active_scene = config.ACTIVE_SCENE
        self.active_location = config.ACTIVE_LOCATION
        # Detect if model has Person class
        self._has_person = 'Person' in config.CLASS_NAMES
        # Init active classes from scene rules
        rules = config.get_scene_rules(config.ACTIVE_SCENE)
        self.active_classes = set()
        if self._has_person:
            self.active_classes.add('Person')
        for ppe in rules.get('mandatory', []) + rules.get('recommended', []):
            self.active_classes.add(ppe)
        config.ACTIVE_CLASSES = self.active_classes

        # Frame source
        self.source_type = None          # 'webcam' | 'video'
        self.video_path = None
        self.cap = None

        # Frame queue (from webcam socket.io or video file)
        self._frame_queue = queue.Queue(maxsize=config.MAX_PROCESSING_BACKLOG)

        # Temporal voter (weighted positive/negative label debouncing)
        self.voter = TemporalVoter(
            window_size=config.VOTER_WINDOW,
            alarm_no_direct=config.VOTER_ALARM_NO_DIRECT,
            alarm_absent=config.VOTER_ALARM_ABSENT,
            clear_present=config.VOTER_CLEAR_PRESENT,
            clear_no_absent=config.VOTER_CLEAR_NO_ABSENT,
            conf_threshold=config.VOTER_CONF_THRESHOLD,
            adaptive_conf=config.VOTER_ADAPTIVE_CONF,
            rate_limit_sec=config.VOTER_RATE_LIMIT_SEC,
            ema_alpha=config.VOTER_EMA_ALPHA,
            rearm_delay_sec=config.VOTER_REARM_DELAY_SEC,
            global_rate_limit_sec=config.VOTER_GLOBAL_RATE_LIMIT_SEC,
            escalation_sec=config.VOTER_ESCALATION_SEC,
        )

        # Person tracker (persistent IDs across frames)
        self.person_tracker = PersonTracker(
            iou_threshold=config.TRACKER_IOU_THRESHOLD,
            max_lost_frames=config.TRACKER_MAX_LOST_FRAMES,
        )

        # Entry/Exit counter (supervision LineZone)
        import supervision as sv
        from supervision.geometry.core import Point
        self.line_zone = None  # initialized on first frame with actual width
        self._zone_initialized = False

        # Annotator
        self.annotator = Annotator()

        # Alarm logger
        self.alarm_logger = AlarmLogger()

        # DingTalk notifier
        self.dingtalk = DingTalkNotifier(
            webhook_url=config.DINGTALK_WEBHOOK,
            enabled=config.DINGTALK_ENABLED,
        )

        # Periodic report accumulator (event-based stats)
        self._period_start = None
        self._period_person_ids = set()       # all unique track IDs seen
        self._period_violator_ids = set()     # track IDs that had ≥1 violation
        self._period_violation_events = []    # [(tid, vtype), ...] unique events
        self._period_violation_types = {}     # {vtype: count}
        self._last_report_time = 0
        self._video_start_time = None

        # Stats
        self.stats = {
            'fps': 0.0,
            'total_detections': 0,
            'total_violations': 0,
            'persons_count': 0,
            'violations_count': 0,
            'alarm_level': 0,
            'uptime': 0,
            'model_device': str(device),
        }
        self._frame_times = []  # rolling FPS calculation
        self._start_time = None

        # Callback for results: func(annotated_frame_bgr, stats_dict, alarm_events)
        self.on_result = None
        # Callback for alarm: func(alarm_event_dict)
        self.on_alarm = None

    # ── Lifecycle ───────────────────────────────────────────────────

    def load_model(self):
        """Load the active model. Call once before start()."""
        logger.info(f"Loading model ({self.model_type}) from {self.model_path} on {self.device}")
        if self.model_type == 'torchvision':
            from .faster_rcnn import FasterRCNNInference
            self.frcnn = FasterRCNNInference(
                model_path=self.model_path,
                device=self.device,
                class_names=config.CLASS_NAMES,
                conf_threshold=self.confidence,
            )
            self.frcnn.load_model()
            logger.info("Faster R-CNN loaded")
        else:
            self.model = YOLO(self.model_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, verbose=False)
            logger.info("Ultralytics model loaded and warmed up")

    def switch_model(self, model_key):
        """Switch to a different model. Requires stopping detection first."""
        if model_key not in config.MODELS:
            return {'error': f'Unknown model: {model_key}'}
        was_running = self._running
        if was_running:
            self.stop()
        # Switch
        config.switch_model(model_key)
        self.model_path = config.MODEL_PATH
        self.model_type = config.MODELS[model_key]['type']
        self.model = None
        self.frcnn = None
        # Re-init active classes from scene
        rules = config.get_scene_rules(self.active_scene)
        self.active_classes = set(config.CLASS_NAMES)
        self.voter.reset()
        self.load_model()
        if was_running:
            self.start(self.source_type, self.video_path)
        return {
            'model': model_key,
            'name': config.MODELS[model_key]['name'],
            'classes': config.CLASS_NAMES,
            'active_classes': sorted(self.active_classes),
        }

    def start(self, source='webcam', video_path=None):
        """
        Start the detection engine.
        source: 'webcam' (frames pushed via push_frame) or 'video' (reads file).
        """
        if self._running:
            logger.warning("Engine already running")
            return

        if self.model is None:
            self.load_model()

        self.source_type = source
        self.video_path = video_path
        self._running = True
        self._start_time = time.time()

        # Reset period accumulator
        self._period_start = time.time()
        self._period_person_ids = set()
        self._period_violator_ids = set()
        self._period_violation_events = []
        self._period_violation_types = {}
        self._last_report_time = time.time()
        if source == 'video':
            self._video_start_time = time.time()

        # Open video capture if using a video file
        if source == 'video' and video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

        # Start background thread
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info(f"Detection engine started (source={source})")

    def stop(self):
        """Stop the detection engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.alarm_logger.flush()
        logger.info("Detection engine stopped")

    def push_frame(self, frame_bgr):
        """
        Push a frame from external source (webcam socket) into the processing queue.
        Non-blocking: drops frame if queue is full.
        """
        try:
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass  # drop frame, processing is backlogged

    def switch_source(self, source, video_path=None):
        """Switch the input source on-the-fly."""
        was_running = self._running
        self.stop()
        self._frame_queue = queue.Queue(maxsize=config.MAX_PROCESSING_BACKLOG)
        self.voter.reset()
        self.person_tracker.reset()
        self._zone_initialized = False
        if was_running:
            self.start(source, video_path)

    def set_confidence(self, value):
        """Set YOLO confidence threshold dynamically (0.10 ~ 0.90)."""
        value = max(config.CONFIDENCE_MIN, min(config.CONFIDENCE_MAX, float(value)))
        self.confidence = value
        config.CONFIDENCE_THRESHOLD = value
        return self.confidence

    def set_scene(self, scene_key):
        """Switch active scene for PPE rules. Also syncs detection labels."""
        if scene_key in config.SCENES:
            self.active_scene = scene_key
            config.ACTIVE_SCENE = scene_key
            # Update location to first preset of new scene
            presets = config.get_camera_presets(scene_key)
            if presets:
                self.active_location = presets[0]
                config.ACTIVE_LOCATION = presets[0]

            # Auto-sync active classes
            rules = config.get_scene_rules(scene_key)
            synced_classes = set()
            if self._has_person:
                synced_classes.add('Person')
            for ppe in rules.get('mandatory', []) + rules.get('recommended', []):
                synced_classes.add(ppe)
            self.active_classes = synced_classes
            config.ACTIVE_CLASSES = synced_classes

            self.voter.reset()
            self.person_tracker.reset()
            return {
                'scene': scene_key,
                'location': self.active_location,
                'active_classes': sorted(synced_classes),
            }
        return None

    def set_location(self, location_name):
        """Set camera preset location within current scene."""
        self.active_location = location_name
        config.ACTIVE_LOCATION = location_name
        return self.active_location

    def set_active_classes(self, class_names):
        """Set which classes to detect."""
        classes = set(class_names)
        if self._has_person:
            classes.add('Person')
        self.active_classes = classes
        config.ACTIVE_CLASSES = classes
        return sorted(self.active_classes)

    # ── Inference Loop ──────────────────────────────────────────────

    def _inference_loop(self):
        """Main detection loop running in background thread."""
        frame_count = 0

        while self._running:
            frame = None
            source = self.source_type

            if source == 'webcam':
                # Get frame from queue (block with timeout)
                try:
                    frame = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

            elif source == 'video' and self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Video ended")
                    self._running = False
                    # Send video-end DingTalk report
                    if config.DINGTALK_ON_VIDEO_END:
                        self._send_video_end_report()
                    if self.on_result:
                        self.stats['video_ended'] = True
                        self.on_result(None, dict(self.stats))
                    break

            if frame is None:
                continue

            frame_count += 1

            # Skip frames for performance
            if frame_count % config.INFERENCE_SKIP_FRAMES != 0:
                continue

            # Process this frame
            try:
                annotated_frame, stats, alarms = self._process_frame(frame)
                # Update stats
                with self._lock:
                    self.stats.update(stats)
                    # Track FPS
                    now = time.time()
                    self._frame_times.append(now)
                    self._frame_times = [t for t in self._frame_times if now - t < 5.0]
                    if len(self._frame_times) >= 2:
                        elapsed = self._frame_times[-1] - self._frame_times[0]
                        self.stats['fps'] = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0
                    self.stats['uptime'] = int(now - self._start_time) if self._start_time else 0

                # Emit result via callback
                if self.on_result:
                    self.on_result(annotated_frame, dict(self.stats))

                # Emit alarms via callback
                for alarm in alarms:
                    if self.on_alarm:
                        self.on_alarm(alarm)

                # ── Accumulate event-based period stats ─────────
                for alarm in alarms:
                    if alarm.get('cleared'):
                        continue
                    tid = alarm.get('track_id', -1)
                    raw_type = alarm.get('type', '')
                    # Split merged types (e.g. 'NO-Helmet|NO-Safety Vest')
                    for vtype in raw_type.split('|'):
                        self._period_violation_events.append((tid, vtype))
                        self._period_violation_types[vtype] = \
                            self._period_violation_types.get(vtype, 0) + 1
                    if tid >= 0:
                        self._period_violator_ids.add(tid)

                # ── Periodic report (webcam mode only) ──────────
                now = time.time()
                if (self.source_type == 'webcam' and
                    now - self._last_report_time >= config.DINGTALK_REPORT_INTERVAL_SEC):
                    self._send_periodic_report()
                    self._last_report_time = now

            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        self.alarm_logger.flush()
        logger.info("Inference loop exited")

    def _process_frame(self, frame_bgr):
        """
        Full single-frame pipeline: YOLO detect → associate → debounce → alarm → annotate.
        Returns: (annotated_frame_bgr, stats_dict, alarm_events_list)
        """
        # Resize for consistent inference
        h, w = frame_bgr.shape[:2]
        scale = config.FRAME_RESIZE_WIDTH / w
        new_w = config.FRAME_RESIZE_WIDTH
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame_bgr, (new_w, new_h))

        # ── Image preprocessing (frame-skip for speed) ─────────────
        if not hasattr(self, '_preprocess_counter'):
            self._preprocess_counter = 0
        self._preprocess_counter += 1
        if self._preprocess_counter % config.PREPROCESS_EVERY_N == 0:
            frame_processed = preprocess(
                frame_resized,
                clahe=config.PREPROCESS_CLAHE,
                color_boost=config.PREPROCESS_COLOR_BOOST,
                sharpen_enabled=config.PREPROCESS_SHARPEN,
            )
        else:
            frame_processed = frame_resized  # skip preprocessing this frame

        # ── Inference (Ultralytics or torchvision) ───────────────────
        if self.model_type == 'torchvision' and self.frcnn:
            self.frcnn.conf_threshold = self.confidence
            boxes_data = self.frcnn.infer(frame_resized)
        elif config.TTA_ENABLED:
            boxes_data = tta_inference(
                self.model, frame_processed,
                device=self.device,
                conf=self.confidence,
                iou=config.IOU_THRESHOLD,
            )
            if config.HELMET_DARK_PENALTY < 1.0:
                boxes_data = penalize_dark_helmets(
                    boxes_data, frame_processed, penalty=config.HELMET_DARK_PENALTY
                )
        else:
            results = self.model(
                frame_processed,
                conf=self.confidence,
                iou=config.IOU_THRESHOLD,
                device=self.device,
                verbose=False,
            )
            boxes_data = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_idx = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    boxes_data.append({
                        'class_idx': cls_idx,
                        'class_name': config.CLASS_IDX_TO_NAME.get(cls_idx, 'unknown'),
                        'confidence': conf,
                        'xyxy': [int(v) for v in xyxy],
                    })

        # ── Temperature scaling: sharpen confidence ──────────────────
        if config.TEMPERATURE_SCALE_ENABLED:
            boxes_data = temperature_scale(boxes_data, T=config.TEMPERATURE_SCALE_T)

        # ── DIoU-NMS: better box dedup (keeps nearby persons separate) ─
        if config.DIOU_NMS_ENABLED:
            boxes_data = diou_nms(boxes_data, threshold=config.DIOU_NMS_THRESHOLD)

        # ── Box EMA: smooth bounding box coordinates across frames ────
        if config.BOX_EMA_ENABLED:
            boxes_data = box_ema_smooth(boxes_data, alpha=config.BOX_EMA_ALPHA)

        # Separate persons (if available) and PPE items
        person_boxes_list = []
        ppe_boxes = []
        for b in boxes_data:
            cls_name = b['class_name']
            if cls_name not in self.active_classes:
                continue
            if self._has_person and cls_name == 'Person':
                if b['confidence'] >= config.PERSON_CONFIDENCE_MIN:
                    person_boxes_list.append(b['xyxy'])
            else:
                ppe_boxes.append((cls_name, b['xyxy'], b['confidence']))

        # ── Person tracking (if Person class available) ────────────
        if self._has_person:
            tracked = self.person_tracker.update(person_boxes_list)
            for tid, info in tracked.items():
                int_box = [int(v) for v in info['box']]
            total_detections = len(tracked)
            tracked_person_boxes = [[int(v) for v in info['box']] for info in tracked.values()]
            tracked_person_ids = list(tracked.keys())
            self._period_person_ids.update(tracked_person_ids)
        else:
            tracked = {}
            total_detections = sum(1 for t, _, _ in ppe_boxes)
            tracked_person_boxes = []
            tracked_person_ids = [0]  # synthetic frame ID

        # ── Entry/Exit counting (person mode only) ──────────────────
        frame_h, frame_w = frame_resized.shape[:2]
        entry_exit_counts = {}
        if self._has_person:
            import supervision as sv
            import numpy as np
            from supervision.geometry.core import Point
            line_y = int(frame_h * 0.40)
            if not self._zone_initialized:
                self.line_zone = sv.LineZone(
                    start=Point(0, line_y), end=Point(frame_w, line_y),
                    triggering_anchors=(sv.Position.BOTTOM_CENTER,),
                )
                self._zone_initialized = True
            if tracked_person_ids and tracked:
                boxes_list = [info['box'] for info in tracked.values()]
                ids_list = list(tracked.keys())
                dets = sv.Detections(
                    xyxy=np.array(boxes_list, dtype=np.float32),
                    tracker_id=np.array(ids_list, dtype=np.int64),
                )
                self.line_zone.trigger(dets)
            entry_exit_counts = {
                'in': self.line_zone.in_count,
                'out': self.line_zone.out_count,
                'current': len(tracked_person_ids),
                'total': self.line_zone.in_count + self.line_zone.out_count,
            }

        # ── Edge/ROI filtering (person mode only) ────────────────────
        edge_ids = set()
        if self._has_person:
            for tid, info in tracked.items():
                x1, y1, x2, y2 = info['box']
                if (x1 <= config.EDGE_MARGIN or y1 <= config.EDGE_MARGIN or
                    x2 >= frame_w - config.EDGE_MARGIN or y2 >= frame_h - config.EDGE_MARGIN):
                    edge_ids.add(tid)
                if config.CENTER_ONLY:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    roi_m = config.CENTER_ROI_MARGIN
                    if not (frame_w * roi_m <= cx <= frame_w * (1 - roi_m) and
                            frame_h * roi_m <= cy <= frame_h * (1 - roi_m)):
                        edge_ids.add(tid)

        # ── PPE detection + temporal voting ──────────────────────────
        scene_key = self.active_scene
        rules = config.get_scene_rules(scene_key)
        if self._has_person:
            ppe_confs = associate_ppe(tracked_person_boxes, ppe_boxes, margin=config.PPE_ASSOC_MARGIN)
            for idx, person_box in enumerate(tracked_person_boxes):
                px1, py1, px2, py2 = person_box
                head_max_y = py1 + (py2 - py1) * 0.35
                for ppe_type, (bx1, by1, bx2, by2), conf in ppe_boxes:
                    if ppe_type not in ('Helmet', 'Hardhat'):
                        continue
                    ppe_center_y = (by1 + by2) / 2
                    if ppe_center_y > head_max_y + 30:
                        if idx in ppe_confs and ppe_confs[idx].get(ppe_type, 0) > 0:
                            ppe_confs[idx][ppe_type] = 0.05
            for idx, tid in enumerate(tracked_person_ids):
                self.voter.update(tid, ppe_confs.get(idx, {}))
        else:
            ppe_confs_frame = {}
            for ppe_type, _, conf in ppe_boxes:
                if conf > ppe_confs_frame.get(ppe_type, 0):
                    ppe_confs_frame[ppe_type] = conf
            for ppe_type in rules.get('mandatory', []) + rules.get('recommended', []):
                if ppe_type not in ppe_confs_frame:
                    ppe_confs_frame[ppe_type] = 0.0
            self.voter.update(0, ppe_confs_frame)

        # ── Hysteresis voting ──────────────────────────────────────
        alarm_events = []
        new_violations = []
        cleared_violations = []

        ppe_check = ['Helmet', 'Safety Vest'] if 'Helmet' in config.CLASS_NAMES else ['Hardhat', 'Safety Vest']
        for tid in tracked_person_ids:
            if self._has_person:
                if tid in edge_ids:
                    continue
                track_age = self.person_tracker.get_track_age(tid)
                if track_age < config.PERSON_MIN_TRACK_AGE:
                    continue

            for ppe_type in ppe_check:
                level = 2 if ppe_type in rules.get('mandatory', []) else (
                    1 if ppe_type in rules.get('recommended', []) else 0)
                if level == 0:
                    continue
                if self.voter.check_clear(tid, ppe_type):
                    cleared_violations.append((tid, f'NO-{ppe_type}'))
                if self.voter.check_violation(tid, ppe_type):
                    new_violations.append((tid, f'NO-{ppe_type}', level))

        self.voter.prune(tracked_person_ids)

        # ── Determine alarm level and log ────────────────────────────
        alarm_level = determine_frame_alarm([(v, l) for _, v, l in new_violations])
        violations_count = self.voter.active_violations  # unique violators

        # Merge new violations by person (one alarm event per person)
        person_violations = {}  # {tid: (max_level, [vtype, ...])}
        for tid, vtype, level in new_violations:
            if tid not in person_violations:
                person_violations[tid] = (level, [])
            person_violations[tid] = (max(person_violations[tid][0], level),
                                       person_violations[tid][1] + [vtype])

        if person_violations:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            for tid, (max_level, vtypes) in person_violations.items():
                screenshot_path = ''
                for vtype in vtypes:
                    if should_screenshot(vtype, scene_key=scene_key):
                        img_name = f"alarm_{vtype}_{time.strftime('%Y%m%d_%H%M%S')}_t{tid}.jpg"
                        img_path = os.path.join(config.ALARM_IMG_DIR, img_name)
                        cv2.imwrite(img_path, frame_bgr)
                        screenshot_path = screenshot_path or img_path
                    self.alarm_logger.log(
                        timestamp=timestamp, violation_type=vtype, level=max_level,
                        screenshot_path=screenshot_path, persons_count=total_detections,
                        violations_count=violations_count, scene=self.active_scene,
                        location=self.active_location,
                    )
                escalated = any(self.voter.is_escalated(tid, v.replace('NO-', '')) for v in vtypes)
                display = '、'.join(get_violation_display_name(v) for v in vtypes)
                alarm_events.append({
                    'level': max_level + (1 if escalated else 0),
                    'type': '|'.join(vtypes),
                    'display_name': display,
                    'timestamp': timestamp, 'screenshot': screenshot_path,
                    'scene': self.active_scene, 'location': self.active_location,
                    'scene_name': config.SCENES.get(self.active_scene, {}).get('name', ''),
                    'track_id': tid, 'escalated': escalated,
                    'duration': max(self.voter.get_violation_duration(tid, v.replace('NO-', '')) for v in vtypes),
                })

        # ── Emit clear events ────────────────────────────────────────
        if cleared_violations:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            for tid, vtype in cleared_violations:
                ppe_type = vtype.replace('NO-', '')
                presence_rate = self.voter.get_presence_rate(tid, ppe_type)
                logger.info(f"VIOLATION CLEARED: person={tid}, {vtype}, presence_rate={presence_rate:.2f}")
                alarm_events.append({
                    'level': 0, 'type': vtype,
                    'display_name': f"{get_violation_display_name(vtype)} → 已佩戴",
                    'timestamp': timestamp, 'screenshot': '',
                    'scene': self.active_scene, 'location': self.active_location,
                    'scene_name': config.SCENES.get(self.active_scene, {}).get('name', ''),
                    'track_id': tid, 'cleared': True,
                })

        # ── Annotate frame ───────────────────────────────────────────
        scene_info = config.SCENES.get(scene_key, {})
        ppe_statuses_bool = {}
        track_ages_list = []
        if self._has_person:
            for idx, tid in enumerate(tracked_person_ids):
                ppe_statuses_bool[idx] = self.voter.get_ppe_statuses(tid)
                track_ages_list.append(self.person_tracker.get_track_age(tid))
        else:
            ppe_statuses_bool = {0: self.voter.get_ppe_statuses(0)}
            track_ages_list = [999]

        annotated = self.annotator.annotate(
            frame_resized,
            person_boxes=tracked_person_boxes,
            ppe_statuses=ppe_statuses_bool,
            ppe_boxes=[(t, b) for t, b, _ in ppe_boxes],
            alarm_level=alarm_level,
            total_persons=total_detections,
            violation_count=violations_count,
            scene_name=scene_info.get('name', ''),
            location=self.active_location,
            track_ids=tracked_person_ids if self._has_person else [],
            track_ages=track_ages_list,
            entry_exit_counts=entry_exit_counts,
        )

        # Stats
        stats = {
            'total_detections': self.stats.get('total_detections', 0) + total_detections,
            'total_violations': self.stats.get('total_violations', 0) + violations_count,
            'persons_count': len(tracked) if self._has_person else 0,
            'violations_count': violations_count,
            'alarm_level': alarm_level,
            'scene': scene_key,
            'scene_name': scene_info.get('name', ''),
            'location': self.active_location,
            'entry_exit': entry_exit_counts,
        }

        return annotated, stats, alarm_events

    # ── DingTalk Reports ─────────────────────────────────────────

    def _send_periodic_report(self):
        """Send periodic summary with violation event counts."""
        now = time.time()
        period_sec = int(now - self._period_start) if self._period_start else 0
        total_events = len(self._period_violation_events)
        violation_types = dict(self._period_violation_types)

        scene_name = config.SCENES.get(self.active_scene, {}).get('name', '')
        self.dingtalk.send_periodic_report(
            scene_name=scene_name, location=self.active_location,
            active_labels=sorted(self.active_classes),
            period_seconds=period_sec,
            total_events=total_events,
            violation_types=violation_types,
        )
        # Reset
        self._period_start = now
        self._period_person_ids = set()
        self._period_violator_ids = set()
        self._period_violation_events = []
        self._period_violation_types = {}

    def _send_video_end_report(self):
        """Send one-shot report after video ends."""
        total_events = len(self._period_violation_events)
        violation_types = dict(self._period_violation_types)
        duration = time.time() - self._video_start_time if self._video_start_time else 0

        scene_name = config.SCENES.get(self.active_scene, {}).get('name', '')
        self.dingtalk.send_video_end_report(
            scene_name=scene_name, location=self.active_location,
            active_labels=sorted(self.active_classes),
            total_events=total_events,
            violation_types=violation_types,
            video_duration_sec=duration,
        )
