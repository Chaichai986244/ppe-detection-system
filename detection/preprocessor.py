"""
Image Preprocessor — Fast single-pass enhancement for PPE detection.
All ops consolidated: 2 color-space conversions max (LAB + HSV).
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Fluorescent vest color ranges in HSV
VEST_HSV_RANGES = [
    (15, 40, 60, 255, 80, 255),   # Fluorescent yellow
    (5, 20, 80, 255, 80, 255),    # Fluorescent orange
    (40, 80, 50, 255, 60, 255),   # Fluorescent green
]

# ── Fast single-pass preprocessing ─────────────────────────────────

def preprocess(frame, clahe=False, color_boost=False, sharpen_enabled=False):
    """
    Single-pass preprocessing. Only converts BGR→LAB (for CLAHE)
    and BGR→HSV (for color boost) once each.
    Returns preprocessed BGR frame.
    """
    result = frame

    # CLAHE: LAB conversion once
    if clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe_obj.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Color boost: HSV conversion once
    if color_boost:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        for h_low, h_high, s_low, s_high, v_low, v_high in VEST_HSV_RANGES:
            range_mask = (
                (hsv[:, :, 0] >= h_low) & (hsv[:, :, 0] <= h_high) &
                (hsv[:, :, 1] >= s_low) & (hsv[:, :, 1] <= s_high) &
                (hsv[:, :, 2] >= v_low) & (hsv[:, :, 2] <= v_high)
            )
            mask = np.maximum(mask, range_mask.astype(np.float32))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.where(mask > 0.5, 1.3, 1.0)[:, :, np.newaxis].squeeze(), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.where(mask > 0.5, 1.3, 1.0)[:, :, np.newaxis].squeeze(), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Sharpen: fast gaussian + addWeighted
    if sharpen_enabled:
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

    return result


# ── Dark helmet penalty ────────────────────────────────────────────

def penalize_dark_helmets(boxes_data, frame, penalty=0.6):
    """
    Reduce confidence of helmet detections in very dark regions (hair).
    Uses a cached HSV conversion if frame is already available.
    """
    import config as cfg
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for b in boxes_data:
        if b['class_name'] != 'Helmet':
            continue
        x1, y1, x2, y2 = [int(v) for v in b['xyxy']]
        if x2 <= x1 or y2 <= y1:
            continue
        region = hsv[max(0, y1):min(hsv.shape[0], y2),
                     max(0, x1):min(hsv.shape[1], x2)]
        if region.size == 0:
            continue
        mean_v = region[:, :, 2].mean()
        if mean_v < 80:
            b['confidence'] *= penalty
    return boxes_data


# ── DIoU-NMS (fast path for small N) ────────────────────────────────

def diou_nms(boxes_data, threshold=0.65):
    """DIoU-NMS. O(n²) but fast for typical PPE scenes (n<30)."""
    n = len(boxes_data)
    if n <= 1:
        return boxes_data
    if n > 50:
        # Too many boxes: use simple class-aware dedup to avoid O(n²)
        boxes_data.sort(key=lambda b: b['confidence'], reverse=True)
        seen, kept = set(), []
        for b in boxes_data:
            key = (b['class_idx'], int(b['xyxy'][0]/20), int(b['xyxy'][1]/20))
            if key not in seen:
                kept.append(b)
                seen.add(key)
        return kept

    boxes_data.sort(key=lambda b: b['confidence'], reverse=True)
    kept = []
    for b in boxes_data:
        suppress = False
        for k in kept:
            if b['class_idx'] != k['class_idx']:
                continue
            if _fast_diou(b['xyxy'], k['xyxy']) > threshold:
                suppress = True
                break
        if not suppress:
            kept.append(b)
    return kept


def _fast_diou(a, b):
    """Compute DIoU quickly using int arithmetic."""
    # Intersection
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    # Areas
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0
    # Center distance / enclosing diagonal
    ax_c = (a[0] + a[2]) * 0.5
    ay_c = (a[1] + a[3]) * 0.5
    bx_c = (b[0] + b[2]) * 0.5
    by_c = (b[1] + b[3]) * 0.5
    center_dist = (ax_c - bx_c) ** 2 + (ay_c - by_c) ** 2
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    diag = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    return iou - (center_dist / diag) if diag > 0 else iou


# ── Temperature scaling ────────────────────────────────────────────

def temperature_scale(boxes_data, T=1.5):
    """Sharpen confidence: conf_new = conf^(1/T). O(n)."""
    inv_t = 1.0 / T
    for b in boxes_data:
        b['confidence'] = b['confidence'] ** inv_t
    return boxes_data


# ── Box EMA (disabled by default, O(n²) per frame) ─────────────────

_box_ema_state = {}

def box_ema_smooth(boxes_data, alpha=0.25):
    """EMA smooth box coords. Disabled by default — costs O(n²)."""
    from .tracker import _box_iou
    global _box_ema_state
    new_state = {}
    for b in boxes_data:
        cls = b['class_name']
        box = b['xyxy']
        key = f"{cls}_{int(box[0]/40)}_{int(box[1]/40)}"
        prev_key = None
        best_iou = 0.3
        for pk, pbox in _box_ema_state.items():
            if pk.startswith(cls + '_'):
                iou = _box_iou(box, pbox)
                if iou > best_iou:
                    best_iou = iou; prev_key = pk
        if prev_key:
            smoothed = [alpha * bv + (1 - alpha) * pv
                        for bv, pv in zip(box, _box_ema_state[prev_key])]
            b['xyxy'] = smoothed
            new_state[key] = smoothed
        else:
            new_state[key] = box
    _box_ema_state = new_state
    return boxes_data


# ── TTA ────────────────────────────────────────────────────────────

def tta_inference(model, frame, device, conf, iou):
    """Test-Time Augmentation: original + horizontal flip."""
    w = frame.shape[1]
    boxes_data = []
    # Original
    results_orig = model(frame, conf=conf, iou=iou, device=device, verbose=False)
    if results_orig[0].boxes is not None:
        for box in results_orig[0].boxes:
            boxes_data.append({
                'class_idx': int(box.cls.item()),
                'class_name': model.names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'xyxy': box.xyxy[0].cpu().numpy().tolist(),
                'source': 'orig',
            })
    # Flip
    flipped = cv2.flip(frame, 1)
    results_flip = model(flipped, conf=conf, iou=iou, device=device, verbose=False)
    if results_flip[0].boxes is not None:
        for box in results_flip[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            boxes_data.append({
                'class_idx': int(box.cls.item()),
                'class_name': model.names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'xyxy': [w - x2, y1, w - x1, y2],
                'source': 'flip',
            })
    # Dedup cross-source duplicates only (IoU > 0.9)
    if len(boxes_data) > 1:
        boxes_data.sort(key=lambda b: b['confidence'], reverse=True)
        kept = []
        for b in boxes_data:
            dup = False
            for k in kept:
                if b['class_idx'] == k['class_idx'] and b['source'] != k['source']:
                    from .tracker import _box_iou
                    if _box_iou(b['xyxy'], k['xyxy']) > 0.9:
                        dup = True; break
            if not dup:
                kept.append(b)
        return [{k: v for k, v in b.items() if k != 'source'} for b in kept]
    return [{k: v for k, v in b.items() if k != 'source'} for b in boxes_data]


def _box_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
