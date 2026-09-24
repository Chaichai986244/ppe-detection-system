"""
Temporal Voter — Consecutive-frame debouncing with adaptive confidence.

State machine per (person, ppe_type):
    CLEAR ──[absent_streak >= alarm_consecutive]──→ VIOLATION
    VIOLATION ──[present_streak >= clear_consecutive]──→ CLEAR

Strategy:
  - Fast attack: alarm after ~8 consecutive absent frames (~0.8s at 10fps)
  - Slow release: clear after ~15 consecutive present frames (~1.5s)
  - Adaptive confidence: uses 50% of historical avg confidence per PPE type
  - Rate limits: 20s per-type cooldown, 3s global per-person
  - Re-arm delay: 5s block after clear
  - Escalation: violation >30s → escalated
"""
import time
import logging
from collections import defaultdict, deque

import config

logger = logging.getLogger(__name__)


class TemporalVoter:
    """
    Per-person, per-PPE-type debouncing with consecutive-frame counters
    and adaptive confidence thresholds.
    """

    def __init__(self,
                 window_size=20,
                 alarm_no_direct=3,
                 alarm_absent=8,
                 clear_present=10,
                 clear_no_absent=5,
                 conf_threshold=0.25,
                 adaptive_conf=True,
                 rate_limit_sec=20,
                 ema_alpha=0.30,
                 rearm_delay_sec=5,
                 global_rate_limit_sec=3,
                 escalation_sec=30):
        # Core — weighted thresholds
        self.window_size = window_size
        self.alarm_no_direct = alarm_no_direct       # NO-* label: fewer frames (high trust)
        self.alarm_absent = alarm_absent              # compliant absent: more frames (lower trust)
        self.clear_present = clear_present            # compliant present: more frames (slow release)
        self.clear_no_absent = clear_no_absent        # NO-* absent: fewer frames (high trust)
        self.conf_threshold = conf_threshold
        self.adaptive_conf = adaptive_conf
        self.rate_limit_sec = rate_limit_sec
        self.ema_alpha = ema_alpha
        self.rearm_delay_sec = rearm_delay_sec
        self.global_rate_limit_sec = global_rate_limit_sec
        self.escalation_sec = escalation_sec

        # Sliding window history {person_id: {ppe_type: deque of (timestamp, is_present)}}
        self._history = defaultdict(lambda: defaultdict(
            lambda: deque(maxlen=window_size)))

        # EMA values {person_id: {ppe_type: float}} — for display only
        self._ema = defaultdict(lambda: defaultdict(float))

        # Adaptive confidence tracking {ppe_type: (sum, count)}
        self._conf_sum = defaultdict(float)
        self._conf_count = defaultdict(int)

        # Consecutive streak counters {person_id: {ppe_type: int}}
        self._absent_streak = defaultdict(lambda: defaultdict(int))
        self._present_streak = defaultdict(lambda: defaultdict(int))

        # State {person_id: {ppe_type: 'CLEAR' | 'VIOLATION'}}
        self._state = defaultdict(dict)

        # Rate limit {person_id: {ppe_type: last_alarm_timestamp}}
        self._last_alarm = defaultdict(dict)

        # Per-person global rate limit {person_id: last_any_alarm_timestamp}
        self._last_any_alarm = {}

        # Re-arm block {person_id: {ppe_type: timestamp_until_blocked}}
        self._rearm_blocked_until = defaultdict(dict)

        # Violation start time {person_id: {ppe_type: timestamp}}
        self._violation_started = defaultdict(dict)

    # ── Adaptive confidence threshold ───────────────────────────────

    def _get_adaptive_threshold(self, ppe_type):
        """Return adaptive confidence threshold for a PPE type.
        Uses 50% of historical average confidence, clamped to [conf_threshold, 0.85]."""
        if not self.adaptive_conf:
            return self.conf_threshold
        if self._conf_count[ppe_type] < 10:
            return self.conf_threshold  # not enough data yet
        avg = self._conf_sum[ppe_type] / self._conf_count[ppe_type]
        adaptive = avg * 0.5
        return max(self.conf_threshold, min(0.85, adaptive))

    # ── Update ─────────────────────────────────────────────────────

    def update(self, person_id, ppe_confs):
        """
        Register a new frame's PPE confidence scores.
        Updates history, EMA, adaptive confidence tracking, and streak counters.
        """
        now = time.time()
        for ppe_type, conf in ppe_confs.items():
            # ── Adaptive confidence: track running average per PPE type ──
            if conf > 0.01:  # only track meaningful detections
                self._conf_sum[ppe_type] += conf
                self._conf_count[ppe_type] += 1

            # ── Determine if present using adaptive threshold ──
            threshold = self._get_adaptive_threshold(ppe_type)
            is_present = conf >= threshold

            # ── Sliding window history ──
            self._history[person_id][ppe_type].append((now, is_present))

            # ── EMA for display ──
            prev = self._ema[person_id][ppe_type]
            self._ema[person_id][ppe_type] = (
                self.ema_alpha * float(is_present) + (1 - self.ema_alpha) * prev
            )

            # ── Consecutive streak counters ──
            if is_present:
                self._present_streak[person_id][ppe_type] += 1
                self._absent_streak[person_id][ppe_type] = 0
            else:
                self._absent_streak[person_id][ppe_type] += 1
                self._present_streak[person_id][ppe_type] = 0

    # ── Presence rate (for logging / external queries) ─────────────

    def _presence_rate(self, person_id, ppe_type):
        """Simple ratio over window."""
        records = self._history.get(person_id, {}).get(ppe_type, deque())
        if not records:
            return 1.0
        present = sum(1 for _, is_p in records if is_p)
        return present / len(records)

    def _ema_rate(self, person_id, ppe_type):
        """EMA-weighted presence rate (for display)."""
        return self._ema.get(person_id, {}).get(ppe_type, 1.0)

    # ── Joint violation check (compliant + violation labels) ──────

    def check_violation_joint(self, person_id, compliant_type, violation_type):
        """
        Joint check using BOTH signals:
        - violation_type (NO-*) directly detected by model → alarm
        - compliant_type absent for N consecutive frames → alarm

        Either condition triggers the violation.
        """
        absent_compliant = self._absent_streak.get(person_id, {}).get(compliant_type, 0)
        present_violation = self._present_streak.get(person_id, {}).get(violation_type, 0)
        current = self._state.get(person_id, {}).get(compliant_type, 'CLEAR')

        if current != 'CLEAR':
            return False

        # Weighted: NO-* direct (high trust, fewer frames) OR compliant absent (lower trust, more frames)
        triggered = (present_violation >= self.alarm_no_direct or
                     absent_compliant >= self.alarm_absent)

        if not triggered:
            return False

        # ── Rate limit checks ──────────────────────────────────
        now = time.time()

        last = self._last_alarm.get(person_id, {}).get(compliant_type, 0)
        if now - last < self.rate_limit_sec:
            return False

        last_any = self._last_any_alarm.get(person_id, 0)
        if now - last_any < self.global_rate_limit_sec:
            return False

        block_until = self._rearm_blocked_until.get(person_id, {}).get(compliant_type, 0)
        if now < block_until:
            return False

        # ── Fire ────────────────────────────────────────────────
        self._state.setdefault(person_id, {})[compliant_type] = 'VIOLATION'
        self._violation_started.setdefault(person_id, {})[compliant_type] = now
        self._last_alarm.setdefault(person_id, {})[compliant_type] = now
        self._last_any_alarm[person_id] = now
        logger.debug(
            f"ALARM: person={person_id}, {compliant_type}, "
            f"NO_direct={present_violation}frames, absent={absent_compliant}frames"
        )
        return True

    def check_clear_joint(self, person_id, compliant_type, violation_type):
        """
        Joint clear: BOTH conditions must be met:
        - compliant_type present for N consecutive frames (person wearing PPE)
        - violation_type absent for N consecutive frames (NO-* not detected)
        """
        present_compliant = self._present_streak.get(person_id, {}).get(compliant_type, 0)
        absent_violation = self._absent_streak.get(person_id, {}).get(violation_type, 0)
        current = self._state.get(person_id, {}).get(compliant_type, 'CLEAR')

        if current != 'VIOLATION':
            return False

        # Weighted: compliant present (more frames, slow release) AND NO-* absent (fewer frames, high trust)
        cleared = (present_compliant >= self.clear_present and
                   absent_violation >= self.clear_no_absent)

        if not cleared:
            return False

        # ── Clear ───────────────────────────────────────────────
        now = time.time()
        self._state.setdefault(person_id, {})[compliant_type] = 'CLEAR'
        self._rearm_blocked_until.setdefault(person_id, {})[compliant_type] = \
            now + self.rearm_delay_sec

        started = self._violation_started.get(person_id, {}).pop(compliant_type, now)
        duration = now - started
        logger.debug(
            f"CLEAR: person={person_id}, {compliant_type}, "
            f"present={present_compliant}frames, NO_absent={absent_violation}frames, "
            f"duration={duration:.1f}s"
        )
        return True

    # ── Hysteresis: consecutive-frame checks ───────────────────────

    def check_violation(self, person_id, ppe_type):
        """
        Check if a NEW alarm should fire.
        Fires when absent_streak >= alarm_consecutive (fast attack).
        Respects re-arm delay and rate limits.
        """
        absent = self._absent_streak.get(person_id, {}).get(ppe_type, 0)
        current = self._state.get(person_id, {}).get(ppe_type, 'CLEAR')

        if current != 'CLEAR':
            return False

        if absent < self.alarm_absent:
            return False  # not enough consecutive absent frames yet

        # ── Rate limit checks ──────────────────────────────────
        now = time.time()

        # Per-PPE-type rate limit
        last = self._last_alarm.get(person_id, {}).get(ppe_type, 0)
        if now - last < self.rate_limit_sec:
            return False

        # Per-person global rate limit
        last_any = self._last_any_alarm.get(person_id, 0)
        if now - last_any < self.global_rate_limit_sec:
            return False

        # Re-arm delay (after a previous clear for same pair)
        block_until = self._rearm_blocked_until.get(person_id, {}).get(ppe_type, 0)
        if now < block_until:
            return False

        # ── Fire ────────────────────────────────────────────────
        self._state.setdefault(person_id, {})[ppe_type] = 'VIOLATION'
        self._violation_started.setdefault(person_id, {})[ppe_type] = now
        self._last_alarm.setdefault(person_id, {})[ppe_type] = now
        self._last_any_alarm[person_id] = now
        rate = self._ema_rate(person_id, ppe_type)
        threshold = self._get_adaptive_threshold(ppe_type)
        logger.debug(
            f"ALARM: person={person_id}, {ppe_type}, "
            f"absent_streak={absent}, ema={rate:.2f}, "
            f"adaptive_conf={threshold:.2f}"
        )
        return True

    def check_clear(self, person_id, ppe_type):
        """
        Check if a violation should clear.
        Clears when present_streak >= clear_consecutive (slow release).
        Sets re-arm delay on clear.
        """
        present = self._present_streak.get(person_id, {}).get(ppe_type, 0)
        current = self._state.get(person_id, {}).get(ppe_type, 'CLEAR')

        if current != 'VIOLATION':
            return False

        if present < self.clear_present:
            return False  # not enough consecutive present frames yet

        # ── Clear ───────────────────────────────────────────────
        now = time.time()
        self._state.setdefault(person_id, {})[ppe_type] = 'CLEAR'

        # Set re-arm block
        self._rearm_blocked_until.setdefault(person_id, {})[ppe_type] = \
            now + self.rearm_delay_sec

        started = self._violation_started.get(person_id, {}).pop(ppe_type, now)
        duration = now - started
        rate = self._ema_rate(person_id, ppe_type)
        logger.debug(
            f"CLEAR: person={person_id}, {ppe_type}, "
            f"present_streak={present}, rate={rate:.2f}, "
            f"duration={duration:.1f}s"
        )
        return True

    # ── Escalation ─────────────────────────────────────────────────

    def get_violation_duration(self, person_id, ppe_type):
        """How many seconds this violation has been active."""
        state = self._state.get(person_id, {}).get(ppe_type, 'CLEAR')
        if state != 'VIOLATION':
            return 0
        started = self._violation_started.get(person_id, {}).get(ppe_type, time.time())
        return time.time() - started

    def is_escalated(self, person_id, ppe_type):
        """Has this violation lasted long enough to escalate?"""
        return self.get_violation_duration(person_id, ppe_type) >= self.escalation_sec

    # ── Query ──────────────────────────────────────────────────────

    def get_state(self, person_id, ppe_type):
        return self._state.get(person_id, {}).get(ppe_type, 'CLEAR')

    def get_presence_rate(self, person_id, ppe_type):
        return self._presence_rate(person_id, ppe_type)

    def get_ema_rate(self, person_id, ppe_type):
        return self._ema_rate(person_id, ppe_type)

    def get_absent_streak(self, person_id, ppe_type):
        return self._absent_streak.get(person_id, {}).get(ppe_type, 0)

    def get_present_streak(self, person_id, ppe_type):
        return self._present_streak.get(person_id, {}).get(ppe_type, 0)

    def get_adaptive_threshold(self, ppe_type):
        return self._get_adaptive_threshold(ppe_type)

    def get_ppe_statuses(self, person_id, threshold=None):
        """Boolean PPE status for overlay display (uses EMA rate)."""
        if threshold is None:
            threshold = 0.30  # lower threshold for display to avoid flicker
        ppe_types = ['Helmet', 'Safety Vest']
        return {
            ppe_type: self._ema_rate(person_id, ppe_type) >= threshold
            for ppe_type in ppe_types
        }

    # ── Cleanup ────────────────────────────────────────────────────

    def prune(self, active_person_ids):
        active_set = set(active_person_ids)
        for pid in list(self._history.keys()):
            if pid not in active_set:
                del self._history[pid]
                self._ema.pop(pid, None)
                self._state.pop(pid, None)
                self._last_alarm.pop(pid, None)
                self._rearm_blocked_until.pop(pid, None)
                self._violation_started.pop(pid, None)
                self._absent_streak.pop(pid, None)
                self._present_streak.pop(pid, None)
        self._last_any_alarm = {
            k: v for k, v in self._last_any_alarm.items() if k in active_set
        }

    def reset(self):
        self._history.clear()
        self._ema.clear()
        self._state.clear()
        self._last_alarm.clear()
        self._last_any_alarm.clear()
        self._rearm_blocked_until.clear()
        self._violation_started.clear()
        self._absent_streak.clear()
        self._present_streak.clear()
        self._conf_sum.clear()
        self._conf_count.clear()

    @property
    def active_violations(self):
        """Number of unique persons with at least one active violation."""
        count = 0
        for pid, ppes in self._state.items():
            if any(s == 'VIOLATION' for s in ppes.values()):
                count += 1
        return count

    @property
    def active_violation_events(self):
        """Total active violation states (person × PPE type)."""
        count = 0
        for pid, ppes in self._state.items():
            for state in ppes.values():
                if state == 'VIOLATION':
                    count += 1
        return count
