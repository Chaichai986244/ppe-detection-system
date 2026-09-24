"""
Alarm Rules — Multi-level, scene-aware violation classification.

Scene-driven logic:
  - mandatory PPE  → Level 2 (Red):   screenshot + CSV log
  - recommended PPE → Level 1 (Yellow): CSV log only
  - optional PPE   → No alarm, no log

Violations are INFERRED: if compliant PPE (Hardhat/Safety Vest) is not detected
near a person, the corresponding violation (NO-Hardhat/NO-Safety Vest) is triggered.
"""
import config


def evaluate_person(ppe_status, scene_key=None):
    """
    Binary evaluation: check which mandatory/recommended PPE is missing.
    """
    rules = config.get_scene_rules(scene_key)
    mandatory = rules.get('mandatory', [])
    recommended = rules.get('recommended', [])
    violations = []
    for item in mandatory:
        if not ppe_status.get(item, False):
            violations.append((f'NO-{item}', 2))
    for item in recommended:
        if not ppe_status.get(item, False):
            violations.append((f'NO-{item}', 1))
    return violations


def evaluate_person_with_confidence(ppe_confs, scene_key=None):
    """
    Evaluate a single person's PPE status, returning confidence scores.

    Args:
        ppe_confs: dict like {'Hardhat': 0.85, 'Safety Vest': 0.0, ...}
        scene_key: scene identifier

    Returns:
        list of (violation_type: str, level: int, confidence: float)
        confidence is the model's confidence for the PPE item (0.0 = not detected)
    """
    rules = config.get_scene_rules(scene_key)
    mandatory = rules.get('mandatory', [])
    recommended = rules.get('recommended', [])

    violations = []

    for item in mandatory:
        conf = ppe_confs.get(item, 0.0)
        violations.append((f'NO-{item}', 2, conf))

    for item in recommended:
        conf = ppe_confs.get(item, 0.0)
        violations.append((f'NO-{item}', 1, conf))

    return violations


def determine_frame_alarm(all_violations):
    """
    Determine the overall frame alarm level.
    Returns: 0 = clear, 1 = yellow warning, 2 = red alarm
    """
    if not all_violations:
        return 0
    levels = [level for _, level in all_violations]
    return max(levels)


def should_screenshot(violation_type, scene_key=None):
    """
    Check if a screenshot should be saved.
    Based on scene rules: mandatory items → screenshot.
    """
    rules = config.get_scene_rules(scene_key)
    ppe_name = violation_type.replace('NO-', '')
    return ppe_name in rules.get('mandatory', [])


def violation_level(violation_type, scene_key=None):
    """Get alarm level (1 or 2) for a violation type in a given scene."""
    rules = config.get_scene_rules(scene_key)
    ppe_name = violation_type.replace('NO-', '')
    if ppe_name in rules.get('mandatory', []):
        return 2
    if ppe_name in rules.get('recommended', []):
        return 1
    return 0


def is_critical(violation_type, scene_key=None):
    """Check if violation type is critical (Level 2) in this scene."""
    return violation_level(violation_type, scene_key) >= 2


def get_violation_display_name(violation_type):
    """Get Chinese display name for a violation type."""
    names = {
        'NO-Helmet': '未戴安全帽',
        'NO-Safety Vest': '未穿反光衣',
    }
    return names.get(violation_type, violation_type)
