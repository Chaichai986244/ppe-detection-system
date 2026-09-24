"""
Person-to-PPE Association — spatially map each PPE item to the person wearing it.
Returns confidence scores (not just booleans) so the temporal voter can use them.
"""
import config


def associate_ppe(person_boxes, ppe_boxes, margin=None):
    """
    For each person box, find all PPE whose center falls within the box.
    Returns max confidence per PPE type (0.0 if not present).

    Args:
        person_boxes: list of [x1, y1, x2, y2]
        ppe_boxes: list of (ppe_type, [x1, y1, x2, y2], confidence)
        margin: pixels to expand person box for association

    Returns:
        dict[int, dict[str, float]]  —  {person_index: {'helmet': 0.85, 'vest': 0.0, ...}}
    """
    if margin is None:
        margin = config.PPE_ASSOC_MARGIN

    ppe_types = ['Helmet', 'Safety Vest']

    results = {}
    for i, (px1, py1, px2, py2) in enumerate(person_boxes):
        # Expand person box by margin (clamp to 0)
        ex1 = max(0, px1 - margin)
        ey1 = max(0, py1 - margin)
        ex2 = px2 + margin
        ey2 = py2 + margin

        # Track max confidence per PPE type (multiple boxes of same type possible)
        ppe_confs = {k: 0.0 for k in ppe_types}

        for ppe_type, (bx1, by1, bx2, by2), conf in ppe_boxes:
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                if conf > ppe_confs[ppe_type]:
                    ppe_confs[ppe_type] = conf

        results[i] = ppe_confs

    return results


def compute_violations_per_person(ppe_confs, scene_key=None):
    """
    Given per-person PPE confidence dicts, compute violation candidates.
    Each violation now carries the confidence score for temporal voting.

    Args:
        ppe_confs: dict from associate_ppe() — {person_idx: {ppe_type: confidence}}
        scene_key: scene identifier for scene-aware rules

    Returns:
        list[list[tuple[str, int, float]]] — per-person list of (violation_type, level, confidence)
    """
    from alarm.rules import evaluate_person_with_confidence

    violations_per_person = []
    for i in sorted(ppe_confs.keys()):
        confs = ppe_confs[i]
        v = evaluate_person_with_confidence(confs, scene_key=scene_key)
        violations_per_person.append(v)

    return violations_per_person
