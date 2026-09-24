"""
PPE Detection System - Central Configuration
Real-time video surveillance violation detection + multi-level alarm system
+ Scene-based detection rules + camera presets
"""
import os
import torch

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ── Model registry (switch via UI) ───────────────────────────────────
MODELS = {
    'rtdetr': {
        'name': 'RT-DETR',
        'path': os.path.join(BASE_DIR, 'best_rfdter.pt'),
        'class_names': ['Helmet', 'Mask', 'Safety Vest', 'boots', 'glove'],
        'type': 'ultralytics',
    },
    'frcnn': {
        'name': 'Faster R-CNN',
        'path': os.path.join(BASE_DIR, 'faster_rcnn_epoch_5.pth'),
        'class_names': ['Safety-Helmet', 'Reflective-Jacket'],
        'type': 'torchvision',
    },
}
ACTIVE_MODEL = 'rtdetr'  # default
MODEL_PATH = MODELS[ACTIVE_MODEL]['path']
ALARM_IMG_DIR = os.path.join(BASE_DIR, 'alarm_img')
CSV_LOG_PATH = os.path.join(BASE_DIR, 'alarm_log.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# ── Model ───────────────────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.30        # YOLO detection confidence (lowered for better recall)
CONFIDENCE_MIN = 0.10
CONFIDENCE_MAX = 0.90
CONFIDENCE_STEP = 0.05
IOU_THRESHOLD = 0.45
CLASS_NAMES = ['Helmet', 'Mask', 'Safety Vest', 'boots', 'glove']

CLASS_IDX_TO_NAME = dict(enumerate(CLASS_NAMES))
CLASS_NAME_TO_IDX = {v: k for k, v in CLASS_IDX_TO_NAME.items()}

# Active detection classes — 安全帽+反光衣. Toggle via UI.
ACTIVE_CLASSES = {'Helmet', 'Safety Vest'}
CLASS_DISPLAY_NAMES = {
    'Helmet': '安全帽', 'Safety Vest': '反光衣', 'Mask': '口罩',
    'boots': '安全靴', 'glove': '手套',
}
CLASS_CATEGORIES = {
    'critical': ['Helmet', 'Safety Vest'],
}

# ── Scenes ──────────────────────────────────────────────────────────
# Each scene defines which PPE is mandatory (Level 2), recommended (Level 1), optional (ignored)
# Only 2 presets. Switching scene auto-syncs the active detection labels.
SCENES = {
    'construction': {
        'name': '建筑工地',
        'name_en': 'Construction Site',
        'description': '强制：安全帽 | 建议：反光衣',
        'mandatory': ['Helmet'],
        'recommended': ['Safety Vest'],
        'optional': [],
        'camera_presets': ['基坑', '塔吊区', '材料堆场', '施工通道'],
    },
    'welding': {
        'name': '焊接车间',
        'name_en': 'Welding Workshop',
        'description': '强制：安全帽 | 建议：反光衣',
        'mandatory': ['Helmet'],
        'recommended': ['Safety Vest'],
        'optional': [],
        'camera_presets': ['焊接工位A', '焊接工位B', '切割区', '打磨区'],
    },
}

ACTIVE_SCENE = 'construction'
ACTIVE_LOCATION = '基坑'

# ── Detection ───────────────────────────────────────────────────────
DEBOUNCE_WINDOW = 3
INFERENCE_SKIP_FRAMES = 1
PPE_ASSOC_MARGIN = 50
PERSON_CONFIDENCE_MIN = 0.20
PERSON_MIN_TRACK_AGE = 8
MAX_PROCESSING_BACKLOG = 5
FRAME_RESIZE_WIDTH = 640

# ── Edge / ROI filtering ────────────────────────────────────────────
EDGE_MARGIN = 25                   # px — ignore persons within N px of frame border
CENTER_ONLY = True                 # only evaluate persons in center ROI
CENTER_ROI_MARGIN = 0.25           # fraction — ignore outer 25% of frame on each side (center 50%)
HELMET_DARK_PENALTY = 0.6          # confidence multiplier for Hardhat detections in dark areas

# ── Image Preprocessing ──────────────────────────────────────────────
PREPROCESS_CLAHE = False           # CLAHE: heavy (~20ms). Enable only for dark scenes.
PREPROCESS_COLOR_BOOST = False     # disabled: NO-Safety Vest label catches violations
PREPROCESS_SHARPEN = False         # Sharpening: mild cost, mild benefit. Off by default.
PREPROCESS_EVERY_N = 3             # preprocess every 3rd frame (performance)
TTA_ENABLED = False
TTA_CONFIDENCE_PENALTY = 0.95

# ── Post-processing ─────────────────────────────────────────────────
DIOU_NMS_ENABLED = True            # DIoU-NMS for better box dedup
DIOU_NMS_THRESHOLD = 0.65          # DIoU threshold for suppression
TEMPERATURE_SCALE_ENABLED = False  # disabled: minimal benefit, saves compute
TEMPERATURE_SCALE_T = 1.5          # T>1 sharpens (high→higher, low→lower)
BOX_EMA_ENABLED = False            # EMA smooth boxes (O(n²), costly. Use Smoother instead)
BOX_EMA_ALPHA = 0.25               # smoothing factor (lower=smoother)

# ── Temporal Voting (weighted positive/negative label debouncing) ───
VOTER_WINDOW = 20                    # frames of EMA history for display (~2s at 10fps)
VOTER_ALARM_NO_DIRECT = 8            # NO-* label: 需持续检出 8 帧才报警 (~0.8s, 防误报)
VOTER_ALARM_ABSENT = 3               # compliant PPE 缺失: 3 帧即报警 (~0.3s, 高置信)
VOTER_CLEAR_PRESENT = 5              # compliant PPE 出现: 5 帧清除 (~0.5s, 快速响应)
VOTER_CLEAR_NO_ABSENT = 10           # NO-* label 消失: 需 10 帧才清除 (~1.0s, 防闪烁)
VOTER_CONF_THRESHOLD = 0.25          # base min YOLO confidence (fallback)
VOTER_ADAPTIVE_CONF = True           # use 50% of historical avg confidence as threshold
VOTER_EMA_ALPHA = 0.30               # EMA smoothing for display only
VOTER_RATE_LIMIT_SEC = 20            # per-PPE-type cooldown
VOTER_REARM_DELAY_SEC = 5            # after clear, block re-alarm for N seconds
VOTER_GLOBAL_RATE_LIMIT_SEC = 3      # per-person: max 1 alarm (any type) per N seconds
VOTER_ESCALATION_SEC = 30            # violation lasting >N seconds → escalated

# ── Person Tracking ──────────────────────────────────────────────────
TRACKER_IOU_THRESHOLD = 0.35          # Min IoU to match same person across frames
TRACKER_MAX_LOST_FRAMES = 15          # Frames before removing a lost track
TRACKER_TYPE = 'bytetrack'            # 'bytetrack' (supervision) or 'iou' (simple)
SMOOTHER_WINDOW = 5                   # DetectionsSmoother frame window (0=disabled)
ALARM_COOLDOWN_SEC = 20               # Seconds before same violation re-fires for same person
ALARM_COOLDOWN_PER_TYPE = True        # Cooldown per violation type (not global per person)

# ── Alarm ───────────────────────────────────────────────────────────
# Alarm levels are now determined per scene based on mandatory/recommended PPE

# ── Streaming ───────────────────────────────────────────────────────
SOCKET_EMIT_FPS = 5
WEBCAM_SEND_FPS = 5
FRAME_JPEG_QUALITY = 50

# ── Flask Server ────────────────────────────────────────────────────
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
SECRET_KEY = 'ppe-detection-system-2026'

# ── Dashboard ───────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC = 60

# ── DingTalk Notification ───────────────────────────────────────────
DINGTALK_ENABLED = True
DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=c401a1252fdfba3dc090898ea33bf53211d5e109a45e0eb6823a090828c103eb"
DINGTALK_REPORT_INTERVAL_SEC = 300   # webcam: push every N seconds
DINGTALK_ON_VIDEO_END = True         # push report when video ends

# ── Auto-create directories ─────────────────────────────────────────
os.makedirs(ALARM_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CSV Columns (extended with scene metadata) ──────────────────────
CSV_COLUMNS = [
    'timestamp',
    'violation_type',
    'level',
    'screenshot_path',
    'persons_count',
    'violations_count',
    'scene',
    'location',
]


# ── Helper ─────────────────────────────────────────────────────────
def get_scene_rules(scene_key=None):
    """Get PPE rules for a scene. Falls back to 'construction'."""
    if scene_key is None:
        scene_key = ACTIVE_SCENE
    return SCENES.get(scene_key, SCENES['construction'])


def get_scene_list():
    """Return list of scene keys with names for UI."""
    return [
        {'key': k, 'name': v['name'], 'name_en': v['name_en'], 'description': v['description']}
        for k, v in SCENES.items()
    ]


def get_camera_presets(scene_key=None):
    """Return camera presets for a scene."""
    rules = get_scene_rules(scene_key)
    return rules.get('camera_presets', [])

def get_model_list():
    """Return available models for UI."""
    return [{'key': k, 'name': v['name'], 'type': v['type']} for k, v in MODELS.items()]

def switch_model(model_key):
    """Switch active model and update class names."""
    global ACTIVE_MODEL, MODEL_PATH, CLASS_NAMES, CLASS_IDX_TO_NAME, CLASS_NAME_TO_IDX, ACTIVE_CLASSES
    if model_key not in MODELS:
        return False
    ACTIVE_MODEL = model_key
    MODEL_PATH = MODELS[model_key]['path']
    CLASS_NAMES = MODELS[model_key]['class_names']
    CLASS_IDX_TO_NAME = dict(enumerate(CLASS_NAMES))
    CLASS_NAME_TO_IDX = {v: k for k, v in CLASS_IDX_TO_NAME.items()}
    ACTIVE_CLASSES = set(CLASS_NAMES)
    return True
