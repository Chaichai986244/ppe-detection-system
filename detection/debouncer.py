"""
Temporal Debounce Filter - 3-frame sliding window per tracked person.
Only confirms a violation when ALL consecutive frames in the window agree.
This eliminates single-frame false positives.
"""
from collections import deque, defaultdict


class DebounceTracker:
    """
    Tracks per-person violation history over a sliding window of frames.
    A violation is only "confirmed" when all frames in the window show it.
    """

    def __init__(self, window_size=3):
        self.window_size = window_size
        # dict[person_id, deque[dict[violation_type, bool]]]
        self._history = defaultdict(lambda: deque(maxlen=window_size))

    def register(self, person_id, violation_dict):
        """
        Record current frame's violation state for a person.
        violation_dict: {'no_helmet': True, 'no_vest': False, 'no_goggles': True, ...}
        """
        self._history[person_id].append(dict(violation_dict))

    def is_confirmed(self, person_id, violation_key):
        """
        Returns True only if the last `window_size` consecutive frames
        ALL show this specific violation for this person.
        """
        records = self._history.get(person_id, deque())
        if len(records) < self.window_size:
            return False  # Not enough history yet
        return all(r.get(violation_key, False) for r in records)

    def get_confirmed_violations(self, person_id):
        """
        Get all confirmed violation types for a person.
        """
        records = self._history.get(person_id, deque())
        if len(records) < self.window_size:
            return []
        confirmed = []
        # Union of all possible keys across the window
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        for key in all_keys:
            if self.is_confirmed(person_id, key):
                confirmed.append(key)
        return confirmed

    def prune(self, active_person_ids):
        """
        Remove entries for persons no longer in frame.
        """
        stale = set(self._history.keys()) - set(active_person_ids)
        for pid in stale:
            del self._history[pid]

    def reset(self):
        """Clear all history."""
        self._history.clear()
