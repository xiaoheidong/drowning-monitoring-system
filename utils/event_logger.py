"""事件日志：CSV + JSONL（供 Web 看板读取），路径一律写绝对路径。"""

import cv2
import csv
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from utils.paths import PROJECT_ROOT

SCHEMA_VERSION = 1


class EventLogger:
    """记录溺水检测事件、保存截图与录像；同步写入 events.jsonl 供 Web 使用。"""

    def __init__(self, output_dir="logs"):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (PROJECT_ROOT / self.output_dir).resolve()

        self.screenshots_dir = self.output_dir / "screenshots"
        self.clips_dir = self.output_dir / "clips"
        self.log_file = self.output_dir / "events.csv"
        self.jsonl_file = self.output_dir / "events.jsonl"

        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "时间", "事件类型", "检测人数", "溺水人数",
                    "置信度", "截图路径", "备注"
                ])

        self._clip_writer = None
        self._clip_path = None
        self._recording = False
        self._record_start_time = None
        self.max_clip_duration = 30

    def _append_jsonl(self, record: dict) -> None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "project_root": str(PROJECT_ROOT),
            **record,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(line)

    def log_event(self, event_type: str, person_count: int, drowning_count: int,
                  confidence: float, frame: np.ndarray = None, note: str = ""):
        timestamp = datetime.now()
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        time_iso = timestamp.astimezone().replace(microsecond=0).isoformat()

        screenshot_abs = ""
        if frame is not None:
            filename = f"event_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            screenshot_abs = str((self.screenshots_dir / filename).resolve())
            cv2.imwrite(screenshot_abs, frame)

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time_str, event_type, person_count, drowning_count,
                f"{confidence:.2%}", screenshot_abs, note
            ])

        self._append_jsonl({
            "kind": "drowning_alert",
            "time_local": time_str,
            "time_iso": time_iso,
            "event_type": event_type,
            "person_count": person_count,
            "drowning_count": drowning_count,
            "confidence": round(float(confidence), 6),
            "screenshot_path_abs": screenshot_abs or None,
            "clip_path_abs": None,
            "note": note,
        })

        if drowning_count > 0:
            try:
                from utils.dingtalk_notify import maybe_send_dingtalk_alert

                maybe_send_dingtalk_alert(
                    time_str=time_str,
                    person_count=person_count,
                    drowning_count=drowning_count,
                    confidence=float(confidence),
                    note=note,
                )
            except Exception as e:
                print(f"[钉钉] 通知异常: {e}")

        return screenshot_abs

    def notify_clip_saved(self, clip_path: str | None):
        """报警结束、录像文件关闭后调用，追加一条 jsonl 便于 Web 关联。"""
        if not clip_path:
            return
        p = Path(clip_path)
        if not p.is_file():
            return
        abs_path = str(p.resolve())
        now = datetime.now().astimezone().replace(microsecond=0).isoformat()
        self._append_jsonl({
            "kind": "clip_saved",
            "time_iso": now,
            "clip_path_abs": abs_path,
            "note": "报警时段录像结束",
        })

    def start_clip_recording(self, frame: np.ndarray, fps: float = 25.0):
        if self._recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._clip_path = str((self.clips_dir / f"alert_{timestamp}.avi").resolve())
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._clip_writer = cv2.VideoWriter(self._clip_path, fourcc, fps, (w, h))
        self._recording = True
        self._record_start_time = datetime.now()

    def write_clip_frame(self, frame: np.ndarray):
        if not self._recording or self._clip_writer is None:
            return

        self._clip_writer.write(frame)

        elapsed = (datetime.now() - self._record_start_time).total_seconds()
        if elapsed >= self.max_clip_duration:
            self.stop_clip_recording()

    def stop_clip_recording(self):
        if self._clip_writer:
            self._clip_writer.release()
            self._clip_writer = None
        self._recording = False
        path = self._clip_path
        self._clip_path = None
        return path

    @property
    def is_recording(self):
        return self._recording

    def get_recent_events(self, count=20) -> list[list[str]]:
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) <= 1:
            return []
        return rows[-count:][::-1]

    def release(self):
        self.stop_clip_recording()
