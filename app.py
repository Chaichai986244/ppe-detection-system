"""
PPE Violation Detection System - Main Application
Flask + Socket.IO server with real-time video detection, multi-level alarm, and dashboard API.
"""
import os
import sys
import time
import base64
import logging
import threading
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import config
from detection.engine import DetectionEngine

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('app')

# ── Flask App ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    async_mode='threading',
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB for base64 frames
    ping_timeout=60,
    ping_interval=25,
)

# ── Detection Engine (singleton) ────────────────────────────────────
engine = None
engine_lock = threading.Lock()

def get_engine():
    """Get or create the detection engine singleton."""
    global engine
    with engine_lock:
        if engine is None:
            engine = DetectionEngine(
                model_path=config.MODEL_PATH,
                device=config.DEVICE,
            )
            engine.load_model()

            # Set callbacks
            def on_result(annotated_frame, stats):
                """Called every detection cycle: emit annotated frame + stats."""
                # Handle video-ended signal (frame is None)
                if annotated_frame is None:
                    socketio.emit('detection_status', {
                        'running': False,
                        'source': engine.source_type if engine else 'video',
                        'message': '视频播放完毕',
                    })
                    return

                # Encode frame as JPEG base64
                _, buffer = cv2.imencode('.jpg', annotated_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_JPEG_QUALITY])
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('detection_frame', {
                    'image_b64': f'data:image/jpeg;base64,{img_b64}',
                    'persons': stats.get('persons_count', 0),
                    'violations': stats.get('violations_count', 0),
                    'alarm_level': stats.get('alarm_level', 0),
                    'fps': round(stats.get('fps', 0), 1),
                    'device': stats.get('model_device', str(config.DEVICE)),
                    'scene': stats.get('scene', 'construction'),
                    'scene_name': stats.get('scene_name', ''),
                    'location': stats.get('location', ''),
                    'entry_exit': stats.get('entry_exit', {}),
                })

                # Also emit stats update every 2 seconds (throttled)
                now = time.time()
                if not hasattr(on_result, '_last_stats_emit'):
                    on_result._last_stats_emit = 0
                if now - on_result._last_stats_emit > 2.0:
                    on_result._last_stats_emit = now
                    socketio.emit('stats_update', {
                        'fps': round(stats.get('fps', 0), 1),
                        'uptime': stats.get('uptime', 0),
                        'model_device': stats.get('model_device', str(config.DEVICE)),
                        'persons': stats.get('persons_count', 0),
                        'violations': stats.get('violations_count', 0),
                        'alarm_level': stats.get('alarm_level', 0),
                        'scene': stats.get('scene', 'construction'),
                        'scene_name': stats.get('scene_name', ''),
                        'location': stats.get('location', ''),
                        })

            def on_alarm(alarm_event):
                """Called when a confirmed alarm fires or a violation clears."""
                is_cleared = alarm_event.get('cleared', False)
                event_name = 'alarm_cleared' if is_cleared else 'alarm_alert'
                socketio.emit(event_name, {
                    'level': alarm_event['level'],
                    'type': alarm_event['type'],
                    'display_name': alarm_event.get('display_name', alarm_event['type']),
                    'timestamp': alarm_event['timestamp'],
                    'screenshot': alarm_event['screenshot'].replace('\\', '/') if alarm_event.get('screenshot') else '',
                    'scene': alarm_event.get('scene', ''),
                    'location': alarm_event.get('location', ''),
                    'scene_name': alarm_event.get('scene_name', ''),
                    'track_id': alarm_event.get('track_id'),
                    'cleared': is_cleared,
                })

            engine.on_result = on_result
            engine.on_alarm = on_alarm

    return engine


# ══════════════════════════════════════════════════════════════════════
#  Page Routes
# ══════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Navigation hub."""
    device_label = 'GPU (CUDA)' if str(config.DEVICE) == '0' else str(config.DEVICE)
    return render_template('index.html', device_label=device_label)


@app.route('/monitor')
def monitor():
    """Live monitoring page."""
    scenes = config.get_scene_list()
    current_scene = config.ACTIVE_SCENE
    presets = config.get_camera_presets(current_scene)
    return render_template('monitor.html', scenes=scenes, current_scene=current_scene, presets=presets,
                          models=config.get_model_list(), active_model=config.ACTIVE_MODEL)


@app.route('/dashboard')
def dashboard():
    """Analytics dashboard page."""
    return render_template('dashboard.html')


@app.route('/training')
def training():
    """Model training progress page."""
    return render_template('training.html')


# ── Video Upload ────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(config.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload a video file for detection."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    # Save to uploads folder
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    logger.info(f"Video uploaded: {save_path}")
    return jsonify({'path': save_path, 'filename': file.filename})


# ══════════════════════════════════════════════════════════════════════
#  Dashboard REST API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/stats/overview')
def api_stats_overview():
    """Get summary statistics."""
    try:
        eng = get_engine()
        csv_stats = eng.alarm_logger.get_stats()
        active_v = eng.voter.active_violations
        return jsonify({
            'total_detections': csv_stats['total_detections'],
            'total_violations': csv_stats['total_violations'],
            'active_violations': active_v,
            'compliance_rate': csv_stats['compliance_rate'],
            'today_alarms': csv_stats['today_alarms'],
        })
    except Exception as e:
        logger.error(f"API error /api/stats/overview: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/violations')
def api_stats_violations():
    """Get violation counts by type (for bar chart)."""
    try:
        eng = get_engine()
        csv_stats = eng.alarm_logger.get_stats()
        by_type = csv_stats.get('by_type', {})

        # Ensure all known types are represented
        all_types = ['NO-Helmet', 'NO-Safety Vest']
        result = []
        for t in all_types:
            count = by_type.get(t, 0)
            # Friendly name
            name_map = {'NO-Hardhat': '未戴安全帽', 'NO-Safety Vest': '未穿反光衣'}
            name = name_map.get(t, t)
            result.append({'type': t, 'name': name, 'count': count})
        return jsonify(result)
    except Exception as e:
        logger.error(f"API error /api/stats/violations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/hourly')
def api_stats_hourly():
    """Get hourly violation trend (for line chart)."""
    try:
        eng = get_engine()
        csv_stats = eng.alarm_logger.get_stats()
        return jsonify(csv_stats.get('by_hour', []))
    except Exception as e:
        logger.error(f"API error /api/stats/hourly: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/compliance')
def api_stats_compliance():
    """Get compliant vs violating ratio (for pie chart)."""
    try:
        eng = get_engine()
        csv_stats = eng.alarm_logger.get_stats()

        total_v = csv_stats['total_violations']
        total_d = csv_stats['total_detections']
        total_persons = total_d + total_v

        compliant = total_d
        violating = total_v

        return jsonify({
            'compliant': compliant,
            'violating': violating,
            'total': total_persons,
            'compliance_rate': csv_stats['compliance_rate'],
        })
    except Exception as e:
        logger.error(f"API error /api/stats/compliance: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alarms/recent')
def api_alarms_recent():
    """Get recent alarm records."""
    try:
        limit = request.args.get('limit', 20, type=int)
        eng = get_engine()
        records = eng.alarm_logger.read_all()
        # Return most recent first
        recent = records[-limit:] if len(records) > limit else records
        return jsonify(list(reversed(recent)))
    except Exception as e:
        logger.error(f"API error /api/alarms/recent: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alarms/today')
def api_alarms_today():
    """Get today's alarm summary."""
    try:
        eng = get_engine()
        csv_stats = eng.alarm_logger.get_stats()
        return jsonify({
            'total': csv_stats['today_alarms'],
            'by_type': csv_stats.get('by_type', {}),
            'compliance_rate': csv_stats['compliance_rate'],
        })
    except Exception as e:
        logger.error(f"API error /api/alarms/today: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alarm_img/<path:filename>')
def serve_alarm_image(filename):
    """Serve alarm screenshot images."""
    return send_from_directory(config.ALARM_IMG_DIR, filename)


@app.route('/api/detection/status')
def api_detection_status():
    """Get current detection engine status."""
    try:
        eng = get_engine()
        return jsonify({
            'running': eng._running,
            'source': eng.source_type,
            'stats': eng.stats,
        })
    except Exception as e:
        return jsonify({'running': False, 'error': str(e)})


# ══════════════════════════════════════════════════════════════════════
#  Scene & Confidence API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/scenes')
def api_scenes():
    """List all available scenes with their PPE rules."""
    return jsonify(config.get_scene_list())


@app.route('/api/scene/current')
def api_scene_current():
    """Get current scene and location."""
    return jsonify({
        'scene': config.ACTIVE_SCENE,
        'location': config.ACTIVE_LOCATION,
        'scene_name': config.SCENES.get(config.ACTIVE_SCENE, {}).get('name', ''),
        'rules': config.get_scene_rules(config.ACTIVE_SCENE),
        'presets': config.get_camera_presets(config.ACTIVE_SCENE),
    })


@app.route('/api/scene/switch', methods=['POST'])
def api_scene_switch():
    """Switch active scene."""
    try:
        data = request.get_json()
        scene_key = data.get('scene', 'construction')
        eng = get_engine()
        result = eng.set_scene(scene_key)
        if result is None:
            return jsonify({'error': f'Unknown scene: {scene_key}'}), 400
        return jsonify({
            'scene': result['scene'],
            'location': result['location'],
            'scene_name': config.SCENES.get(result['scene'], {}).get('name', ''),
            'presets': config.get_camera_presets(result['scene']),
            'active_classes': result.get('active_classes', []),
            'rules': config.get_scene_rules(result['scene']),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/confidence', methods=['POST'])
def api_set_confidence():
    """Set YOLO confidence threshold."""
    try:
        data = request.get_json()
        value = float(data.get('value', config.CONFIDENCE_THRESHOLD))
        eng = get_engine()
        new_val = eng.set_confidence(value)
        return jsonify({'confidence': new_val})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/location', methods=['POST'])
def api_set_location():
    """Set camera preset location within current scene."""
    try:
        data = request.get_json()
        location = data.get('location', config.ACTIVE_LOCATION)
        eng = get_engine()
        result = eng.set_location(location)
        return jsonify({'location': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════
#  Class Filter API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/classes')
def api_classes():
    """Get all available classes and their current active state."""
    return jsonify({
        'all': config.CLASS_NAMES,
        'active': sorted(config.ACTIVE_CLASSES),
        'display_names': config.CLASS_DISPLAY_NAMES,
        'categories': config.CLASS_CATEGORIES,
    })


@app.route('/api/classes', methods=['POST'])
def api_set_classes():
    """Set which classes to detect."""
    try:
        data = request.get_json()
        class_names = data.get('classes', config.CLASS_NAMES)
        eng = get_engine()
        result = eng.set_active_classes(class_names)
        return jsonify({'active': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════
#  Filtered Alarm Query API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/alarms/filter')
def api_alarms_filter():
    """
    Filter alarm records by type, scene, location, and time range.
    Query params: type, scene, location, date_from, date_to, limit
    """
    try:
        eng = get_engine()
        records = eng.alarm_logger.read_all()

        # Parse filters
        filter_type = request.args.get('type')
        filter_scene = request.args.get('scene')
        filter_location = request.args.get('location')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        limit = request.args.get('limit', 100, type=int)

        filtered = []
        for r in records:
            if filter_type and r.get('violation_type') != filter_type:
                continue
            if filter_scene and r.get('scene') != filter_scene:
                continue
            if filter_location and r.get('location') != filter_location:
                continue
            if date_from and r.get('timestamp', '') < date_from:
                continue
            if date_to and r.get('timestamp', '') > date_to + ' 23:59:59':
                continue
            filtered.append(r)

        # Return most recent first, limited
        filtered = filtered[-limit:]
        filtered.reverse()

        return jsonify(filtered)
    except Exception as e:
        logger.error(f"API error /api/alarms/filter: {e}")
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════
#  PyEcharts Chart Data API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/charts/bar')
def api_chart_bar():
    """Return ECharts bar chart option for violation types."""
    try:
        eng = get_engine()
        stats = eng.alarm_logger.get_stats()
        by_type = stats.get('by_type', {})

        type_names = {'NO-Helmet':'未戴安全帽','NO-Safety Vest':'未穿反光衣'}
        colors = ['#ff3d00','#ffab00']
        all_types = ['NO-Helmet','NO-Safety Vest']

        x_data = [type_names.get(t, t) for t in all_types]
        y_data = [by_type.get(t, 0) for t in all_types]

        return jsonify({
            'xAxis': [{'type': 'category', 'data': x_data, 'axisLabel': {'color': '#7a8ea0', 'fontSize': 10}}],
            'yAxis': [{'type': 'value', 'axisLabel': {'color': '#7a8ea0', 'fontSize': 10}, 'splitLine': {'lineStyle': {'color': '#1e3040'}}}],
            'series': [{'type': 'bar', 'data': [{'value': v, 'itemStyle': {'color': colors[i]}} for i, v in enumerate(y_data)],
                        'label': {'show': True, 'position': 'top', 'color': '#d4dce6', 'fontSize': 11}}],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/line')
def api_chart_line():
    """Return ECharts line chart option for hourly trend."""
    try:
        eng = get_engine()
        stats = eng.alarm_logger.get_stats()
        by_hour = stats.get('by_hour', [])

        if not by_hour:
            return jsonify({'xAxis':[{'type':'category','data':[]}],'yAxis':[{'type':'value'}],'series':[]})

        hours = [d.get('hour','') for d in by_hour]
        name_map = {'NO-Helmet':'未戴安全帽','NO-Safety Vest':'未穿反光衣'}
        color_map = {'NO-Helmet':'#ff3d00','NO-Safety Vest':'#ffab00'}

        all_keys = set()
        for d in by_hour:
            for k in d:
                if k != 'hour': all_keys.add(k)

        series = []
        for key in sorted(all_keys):
            series.append({
                'name': name_map.get(key, key),
                'type': 'line',
                'smooth': True,
                'data': [d.get(key, 0) for d in by_hour],
                'lineStyle': {'color': color_map.get(key, '#fff'), 'width': 2},
                'itemStyle': {'color': color_map.get(key, '#fff')},
                'symbol': 'circle', 'symbolSize': 5,
            })

        return jsonify({
            'xAxis': [{'type': 'category', 'data': hours, 'axisLabel': {'color': '#7a8ea0', 'fontSize': 10}}],
            'yAxis': [{'type': 'value', 'axisLabel': {'color': '#7a8ea0', 'fontSize': 10}, 'splitLine': {'lineStyle': {'color': '#1e3040'}}}],
            'series': series,
            'tooltip': {'trigger': 'axis'},
            'legend': {'textStyle': {'color': '#7a8ea0', 'fontSize': 10}, 'top': 0},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/pie')
def api_chart_pie():
    """Return ECharts pie chart option for compliance ratio."""
    try:
        eng = get_engine()
        stats = eng.alarm_logger.get_stats()

        # Use compliance_rate directly from stats
        compliance = stats.get('compliance_rate', 100.0)
        violation_pct = round(100.0 - compliance, 1)

        return jsonify({
            'series': [{
                'type': 'pie',
                'radius': ['45%', '70%'],
                'center': ['50%', '55%'],
                'data': [
                    {'value': round(compliance, 1), 'name': '合规率', 'itemStyle': {'color': '#00c853'}},
                    {'value': violation_pct, 'name': '违规率', 'itemStyle': {'color': '#ff3d00'}},
                ],
                'label': {'color': '#d4dce6', 'fontSize': 12, 'formatter': '{b}\n{d}%'},
                'emphasis': {'itemStyle': {'shadowBlur': 10, 'shadowColor': 'rgba(0,0,0,0.5)'}},
            }],
            'legend': {'textStyle': {'color': '#7a8ea0', 'fontSize': 10}, 'top': 5},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════
#  Training Progress API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/training/status')
def api_training_status():
    """Get live training progress from train_output."""
    import pandas as pd
    results_dir = os.path.join(config.OUTPUT_DIR, 'ppe_detection')
    csv_path = os.path.join(results_dir, 'results.csv')

    if not os.path.exists(csv_path):
        return jsonify({'error': 'No training results yet', 'running': False})

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        current_epoch = int(last['epoch']) if 'epoch' in df.columns else len(df)

        # Best metrics
        map50_col = [c for c in df.columns if 'mAP50' in c and 'mAP50-95' not in c]
        map5095_col = [c for c in df.columns if 'mAP50-95' in c]

        best_map50 = float(df[map50_col[0]].max()) if map50_col else 0
        best_map50_epoch = int(df[map50_col[0]].idxmax()) + 1 if map50_col else 0
        best_map5095 = float(df[map5095_col[0]].max()) if map5095_col else 0

        # Current epoch metrics
        current = {
            'epoch': current_epoch,
            'box_loss': float(last.get('train/box_loss', 0)),
            'cls_loss': float(last.get('train/cls_loss', 0)),
            'dfl_loss': float(last.get('train/dfl_loss', 0)),
            'precision': float(last.get('metrics/precision(B)', 0)),
            'recall': float(last.get('metrics/recall(B)', 0)),
            'map50': float(last.get(map50_col[0], 0)) if map50_col else 0,
            'map50_95': float(last.get(map5095_col[0], 0)) if map5095_col else 0,
        }

        # Full history for charts
        history = {
            'epochs': df['epoch'].tolist() if 'epoch' in df.columns else list(range(1, len(df)+1)),
        }
        for col in df.columns:
            if col != 'epoch' and col != 'time':
                history[col] = df[col].tolist()

        return jsonify({
            'running': True,
            'current_epoch': current_epoch,
            'total_epochs': 100,
            'best_map50': round(best_map50, 4),
            'best_map50_epoch': best_map50_epoch,
            'best_map5095': round(best_map5095, 4),
            'current': current,
            'history': history,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'running': False})


# ══════════════════════════════════════════════════════════════════════
#  DingTalk Settings API
# ══════════════════════════════════════════════════════════════════════

@app.route('/api/dingtalk/settings', methods=['GET'])
def api_dingtalk_get():
    """Get current DingTalk settings."""
    return jsonify({
        'enabled': config.DINGTALK_ENABLED,
        'webhook': config.DINGTALK_WEBHOOK[:60] + '...' if len(config.DINGTALK_WEBHOOK) > 60 else config.DINGTALK_WEBHOOK,
        'interval_sec': config.DINGTALK_REPORT_INTERVAL_SEC,
        'on_video_end': config.DINGTALK_ON_VIDEO_END,
    })


@app.route('/api/dingtalk/settings', methods=['POST'])
def api_dingtalk_set():
    """Update DingTalk settings."""
    try:
        data = request.get_json()
        if 'enabled' in data:
            config.DINGTALK_ENABLED = bool(data['enabled'])
            eng = get_engine()
            if eng:
                eng.dingtalk.enabled = config.DINGTALK_ENABLED
        if 'interval_sec' in data:
            config.DINGTALK_REPORT_INTERVAL_SEC = int(data['interval_sec'])
        if 'on_video_end' in data:
            config.DINGTALK_ON_VIDEO_END = bool(data['on_video_end'])
        if 'webhook' in data:
            config.DINGTALK_WEBHOOK = str(data['webhook'])
            eng = get_engine()
            if eng:
                eng.dingtalk.webhook_url = config.DINGTALK_WEBHOOK
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dingtalk/test', methods=['POST'])
def api_dingtalk_test():
    """Send a test notification to verify webhook."""
    try:
        from notification.dingtalk import DingTalkNotifier
        n = DingTalkNotifier(webhook_url=config.DINGTALK_WEBHOOK, enabled=True)
        ok = n.send_periodic_report(
            scene_name='系统测试', location='测试机位',
            active_labels=['Person', 'Hardhat', 'Safety Vest'],
            period_seconds=0, unique_persons=0, violator_count=0,
            violation_events=0, compliance=100.0, scene_violations={},
        )
        return jsonify({'ok': ok})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════
#  Socket.IO Handlers
# ══════════════════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('server_info', {
        'message': 'Connected to PPE Detection System',
        'device': str(config.DEVICE),
        'model': 'YOLOv8s',
        'classes': config.CLASS_NAMES,
    })


@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('start_detection')
def on_start_detection(data):
    """Start or switch the detection engine."""
    try:
        source = data.get('source', 'webcam')
        video_path = data.get('video_path', None)
        eng = get_engine()

        # If already running, stop first then restart with new source
        if eng._running:
            eng.stop()

        eng.start(source=source, video_path=video_path)
        emit('detection_status', {
            'running': True,
            'source': source,
            'message': f'Detection started ({source})',
        })
        logger.info(f"Detection started: source={source}, path={video_path}")
    except Exception as e:
        logger.error(f"Failed to start detection: {e}")
        emit('server_error', {'message': str(e)})


@socketio.on('stop_detection')
def on_stop_detection():
    """Stop the detection engine."""
    try:
        eng = get_engine()
        eng.stop()
        emit('detection_status', {
            'running': False,
            'message': 'Detection stopped',
        })
        logger.info("Detection stopped by client")
    except Exception as e:
        logger.error(f"Failed to stop detection: {e}")
        emit('server_error', {'message': str(e)})


@socketio.on('webcam_frame')
def on_webcam_frame(data):
    """
    Receive a frame from the browser webcam.
    data: {image_b64: 'data:image/jpeg;base64,...', width: int, height: int}
    """
    try:
        # Decode base64 image
        img_b64 = data.get('image_b64', '')
        # Strip data URL prefix if present
        if 'base64,' in img_b64:
            img_b64 = img_b64.split('base64,')[1]

        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is not None:
            eng = get_engine()
            eng.push_frame(frame)
    except Exception as e:
        logger.error(f"Error processing webcam frame: {e}")


@socketio.on('switch_source')
def on_switch_source(data):
    """Switch detection source on-the-fly."""
    try:
        source = data.get('source', 'webcam')
        video_path = data.get('video_path', None)
        eng = get_engine()
        eng.switch_source(source=source, video_path=video_path)
        emit('detection_status', {
            'running': True,
            'source': source,
            'message': f'Switched to {source}',
        })
    except Exception as e:
        logger.error(f"Failed to switch source: {e}")
        emit('server_error', {'message': str(e)})


@socketio.on('request_stats')
def on_request_stats():
    """Send current stats immediately (for page load)."""
    try:
        eng = get_engine()
        emit('stats_update', {
            'fps': round(eng.stats.get('fps', 0), 1),
            'uptime': eng.stats.get('uptime', 0),
            'model_device': eng.stats.get('model_device', str(config.DEVICE)),
            'persons': eng.stats.get('persons_count', 0),
            'violations': eng.stats.get('violations_count', 0),
            'alarm_level': eng.stats.get('alarm_level', 0),
        })
    except Exception as e:
        emit('server_error', {'message': str(e)})


@socketio.on('switch_scene')
def on_switch_scene(data):
    """Switch active detection scene."""
    try:
        scene_key = data.get('scene', 'construction')
        eng = get_engine()
        result = eng.set_scene(scene_key)
        if result is None:
            emit('server_error', {'message': f'Unknown scene: {scene_key}'})
            return
        emit('scene_changed', {
            'scene': result['scene'],
            'location': result['location'],
            'scene_name': config.SCENES.get(result['scene'], {}).get('name', ''),
            'presets': config.get_camera_presets(result['scene']),
            'active_classes': result.get('active_classes', []),
            'rules': {
                'mandatory': config.get_scene_rules(result['scene']).get('mandatory', []),
                'recommended': config.get_scene_rules(result['scene']).get('recommended', []),
                'optional': config.get_scene_rules(result['scene']).get('optional', []),
            },
        })
    except Exception as e:
        emit('server_error', {'message': str(e)})


@socketio.on('set_location')
def on_set_location(data):
    """Set camera preset location."""
    try:
        location = data.get('location', config.ACTIVE_LOCATION)
        eng = get_engine()
        eng.set_location(location)
        emit('location_changed', {'location': location})
    except Exception as e:
        emit('server_error', {'message': str(e)})


@socketio.on('set_confidence')
def on_set_confidence(data):
    """Set YOLO confidence threshold dynamically."""
    try:
        value = float(data.get('value', config.CONFIDENCE_THRESHOLD))
        eng = get_engine()
        new_val = eng.set_confidence(value)
        emit('confidence_changed', {'confidence': new_val})
    except Exception as e:
        emit('server_error', {'message': str(e)})


@socketio.on('set_classes')
def on_set_classes(data):
    """Set which classes to detect."""
    try:
        class_names = data.get('classes', config.CLASS_NAMES)
        eng = get_engine()
        result = eng.set_active_classes(class_names)
        emit('classes_changed', {'active': result, 'display_names': config.CLASS_DISPLAY_NAMES})
    except Exception as e:
        emit('server_error', {'message': str(e)})


@socketio.on('switch_model')
def on_switch_model(data):
    """Switch detection model."""
    try:
        model_key = data.get('model', config.ACTIVE_MODEL)
        eng = get_engine()
        result = eng.switch_model(model_key)
        emit('model_changed', result)
    except Exception as e:
        emit('server_error', {'message': str(e)})


@app.route('/api/models')
def api_models():
    """Get available models and current active one."""
    return jsonify({
        'models': config.get_model_list(),
        'active': config.ACTIVE_MODEL,
        'classes': config.CLASS_NAMES,
    })


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info(f"Starting PPE Detection System on {config.FLASK_HOST}:{config.FLASK_PORT}")
    logger.info(f"Model: {config.MODEL_PATH}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Alarm images: {config.ALARM_IMG_DIR}")
    logger.info(f"CSV log: {config.CSV_LOG_PATH}")

    # Pre-load engine
    get_engine()

    socketio.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
