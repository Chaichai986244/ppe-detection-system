"""
Alarm Logger — Buffered CSV writer for violation records.
Log format: timestamp, violation_type, level, screenshot_path, persons_count, violations_count
"""
import os
import csv
import threading
import logging

import config

logger = logging.getLogger(__name__)


class AlarmLogger:
    """Thread-safe CSV logger for alarm events with buffered writes."""

    def __init__(self, csv_path=config.CSV_LOG_PATH, buffer_size=10):
        self.csv_path = csv_path
        self.buffer_size = buffer_size
        self._buffer = []
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self):
        """Create CSV with header if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(config.CSV_COLUMNS)

    def log(self, timestamp, violation_type, level, screenshot_path,
            persons_count, violations_count, scene='', location=''):
        """
        Log a violation event. Buffered: auto-flush when buffer is full.

        Args:
            timestamp: str, ISO-format time
            violation_type: str, e.g. 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'
            level: int, 1 or 2
            screenshot_path: str, relative or absolute path to screenshot ('' if none)
            persons_count: int, total persons detected in frame
            violations_count: int, total violations in frame
            scene: str, active scene key at time of violation
            location: str, camera preset location
        """
        row = [
            timestamp,
            violation_type,
            level,
            screenshot_path,
            persons_count,
            violations_count,
            scene,
            location,
        ]

        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= self.buffer_size:
                self._flush_unlocked()

    def flush(self):
        """Force flush all buffered entries to CSV."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self):
        """Write buffered rows to CSV. Must hold self._lock."""
        if not self._buffer:
            return
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self._buffer)
            logger.debug(f"Flushed {len(self._buffer)} alarm records to CSV")
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Failed to write alarm log: {e}")

    def read_all(self):
        """Read all records from CSV. Returns list of dicts."""
        self.flush()
        if not os.path.exists(self.csv_path):
            return []
        rows = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def count_today(self):
        """Count today's alarm records."""
        import time
        today = time.strftime('%Y-%m-%d')
        records = self.read_all()
        return sum(1 for r in records if r.get('timestamp', '').startswith(today))

    def get_stats(self):
        """Get aggregated stats from CSV."""
        self.flush()
        records = self.read_all()

        if not records:
            return {
                'total_detections': 0,
                'total_violations': 0,
                'compliance_rate': 100.0,
                'today_alarms': 0,
                'by_type': {},
                'by_hour': [],
            }

        # Count violation types
        by_type = {}
        total_detections = 0
        total_violations = 0

        for r in records:
            vtype = r.get('violation_type', 'unknown')
            by_type[vtype] = by_type.get(vtype, 0) + 1
            try:
                total_detections += int(r.get('persons_count', 0))
                total_violations += int(r.get('violations_count', 0))
            except (ValueError, TypeError):
                pass

        # Compliance rate
        total_persons = total_detections + total_violations
        if total_persons > 0:
            compliance_rate = round(total_detections / total_persons * 100, 1)
        else:
            compliance_rate = 100.0

        # Hourly breakdown
        by_hour_data = {}
        for r in records:
            ts = r.get('timestamp', '')
            if len(ts) >= 13:
                hour = ts[11:13]
                vtype = r.get('violation_type', 'unknown')
                if hour not in by_hour_data:
                    by_hour_data[hour] = {}
                by_hour_data[hour][vtype] = by_hour_data[hour].get(vtype, 0) + 1

        by_hour = []
        for h in sorted(by_hour_data.keys()):
            entry = {'hour': f'{h}:00'}
            entry.update(by_hour_data[h])
            by_hour.append(entry)

        # Today's count
        import time
        today = time.strftime('%Y-%m-%d')
        today_count = sum(1 for r in records if r.get('timestamp', '').startswith(today))

        return {
            'total_detections': total_detections,
            'total_violations': total_violations,
            'compliance_rate': compliance_rate,
            'today_alarms': today_count,
            'by_type': by_type,
            'by_hour': by_hour,
        }
