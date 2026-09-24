"""
DingTalk Robot Notification — Markdown message push via webhook.
Event-based reports: unique persons + unique violation events.
"""
import json, time, logging, urllib.request

logger = logging.getLogger(__name__)


class DingTalkNotifier:

    def __init__(self, webhook_url, enabled=True):
        self.webhook_url = webhook_url
        self.enabled = enabled

    def send_markdown(self, title, text):
        if not self.enabled:
            return False
        payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.webhook_url, data=data,
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("errcode") == 0:
                logger.info(f"DingTalk sent: {title}")
                return True
            logger.error(f"DingTalk error: {result}")
            return False
        except Exception as e:
            logger.error(f"DingTalk send failed: {e}")
            return False

    def _format_report(self, scene_name, location, labels_text, now_str,
                       period_str, total_events, violation_types):
        type_names = {'NO-Helmet': '未戴安全帽', 'NO-Safety Vest': '未穿反光衣'}
        lines = []
        for vtype, count in sorted(violation_types.items()):
            name = type_names.get(vtype, vtype)
            lines.append(f"{name}：{count}次")
        if lines:
            detail = '、'.join(lines)
            status = f"存在违规（{total_events}次）"
        else:
            detail = '-'
            status = '全部合规'

        return f"""## PPE 监控报告

> 场景：{scene_name}  |  机位：{location}  |  {now_str}  |  {period_str}

检测：{labels_text}

---
状态：{status}
{detail if lines else ''}"""

    def send_periodic_report(self, scene_name, location, active_labels,
                              period_seconds, total_events, violation_types):
        label_names = {'Helmet': '安全帽', 'Safety Vest': '反光衣'}
        labels_text = '、'.join(label_names.get(l, l) for l in active_labels if l != 'Person')
        now_str = time.strftime("%H:%M")
        period_str = f"近{period_seconds // 60}分钟"
        text = self._format_report(scene_name, location, labels_text, now_str,
                                   period_str, total_events, violation_types)
        title = "PPE告警" if violation_types else "PPE正常"
        return self.send_markdown(f"{title} - {scene_name}", text)

    def send_video_end_report(self, scene_name, location, active_labels,
                               total_events, violation_types, video_duration_sec):
        label_names = {'Helmet': '安全帽', 'Safety Vest': '反光衣'}
        labels_text = '、'.join(label_names.get(l, l) for l in active_labels if l != 'Person')
        now_str = time.strftime("%H:%M")
        mins, secs = int(video_duration_sec // 60), int(video_duration_sec % 60)
        period_str = f"视频 {mins}分{secs}秒"
        text = self._format_report(scene_name, location, labels_text, now_str,
                                   period_str, total_events, violation_types)
        title = "视频告警" if violation_types else "视频正常"
        return self.send_markdown(f"{title} - {scene_name}", text)
