"""视频处理模块 - 集成检测、分类、追踪、报警、日志的完整推理流水线"""

import cv2
import time
import numpy as np
from models.detector import PersonDetector
from models.classifier import StateClassifier, CLASS_NAMES_CN, COLORS
from utils.drowning_tracker import DrowningTracker
from utils.alarm import AlarmManager
from utils.event_logger import EventLogger
from utils.settings import get_settings
from utils.cv2_chinese import put_text_cn, put_text_cn_with_bg, put_text_cn_center, text_size_cn


class VideoProcessor:
    def __init__(self, detector_model="yolov8n.pt", classifier_model=None,
                 det_conf=0.5, device="auto", confirm_frames=8, log_dir=None):
        self.detector = PersonDetector(
            model_path=detector_model, conf_threshold=det_conf, device=device
        )
        self.classifier = StateClassifier(model_path=classifier_model, device=device)
        self.tracker = DrowningTracker(confirm_frames=confirm_frames)
        self.alarm = AlarmManager(cooldown_seconds=5)
        log_dir = log_dir or (get_settings().get("logs") or {}).get("output_dir") or "logs"
        self.logger = EventLogger(output_dir=log_dir)

        self.fps = 0.0
        self.person_count = 0
        self.drowning_confirmed_count = 0
        self.det_conf = det_conf
        self.roi_polygon = None
        self._alarm_logged = False

    def set_det_confidence(self, value: float):
        self.det_conf = value
        self.detector.conf_threshold = value

    def set_confirm_frames(self, value: int):
        self.tracker.confirm_frames = value

    def set_alarm_cooldown(self, value: int):
        self.alarm.cooldown = value

    def set_roi(self, polygon: list[tuple[int, int]] | None):
        """设置 ROI 多边形区域，None 表示取消"""
        self.roi_polygon = polygon

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧：检测 → 分类 → 追踪 → 报警 → 日志 → 绘制"""
        start_time = time.time()

        detections = self.detector.detect(frame)

        if self.roi_polygon:
            detections = self._filter_by_roi(detections)

        crops = self.detector.crop_persons(frame, detections)
        predictions = self.classifier.predict_batch(crops) if crops else []

        tracked = self.tracker.update(detections, predictions)

        annotated = frame.copy()
        self.person_count = len(tracked)
        self.drowning_confirmed_count = 0

        if self.roi_polygon:
            self._draw_roi(annotated)

        for i, item in enumerate(tracked):
            x1, y1, x2, y2 = item["bbox"]
            track_id = item["track_id"]
            confirmed = item["confirmed_drowning"]
            drowning_cnt = item["drowning_count"]

            pred_class, pred_conf = predictions[i] if i < len(predictions) else ("unknown", 0)
            color = COLORS.get(pred_class, (255, 255, 255))
            cn_name = CLASS_NAMES_CN.get(pred_class, pred_class)

            if confirmed:
                self.drowning_confirmed_count += 1
                color = (0, 0, 255)
                thickness = 3
            else:
                thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"#{track_id} {cn_name} {pred_conf:.0%}"
            if pred_class == "drowning":
                label += f" [{drowning_cnt}/{self.tracker.confirm_frames}]"
            _fs = 18
            _, th = text_size_cn(label, _fs)
            ty = max(0, y1 - th - 8)
            put_text_cn_with_bg(annotated, label, (x1 + 2, ty), _fs, color, (255, 255, 255))

            if confirmed:
                put_text_cn(annotated, "溺水确认!", (x1, y2 + 22), 20, (0, 0, 255), anchor="ls")

        if self.drowning_confirmed_count > 0:
            self._draw_alarm_banner(annotated)
            self.alarm.trigger()

            if not self._alarm_logged:
                max_conf = max((p[1] for p in predictions if p[0] == "drowning"), default=0)
                self.logger.log_event(
                    "溺水报警", self.person_count, self.drowning_confirmed_count,
                    max_conf, frame, "连续帧确认触发"
                )
                if not self.logger.is_recording:
                    self.logger.start_clip_recording(frame, fps=25.0)
                self._alarm_logged = True

            if self.logger.is_recording:
                self.logger.write_clip_frame(annotated)
        else:
            if self._alarm_logged:
                clip_path = self.logger.stop_clip_recording()
                self.logger.notify_clip_saved(clip_path)
                self._alarm_logged = False

        elapsed = time.time() - start_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0

        self._draw_info_panel(annotated)

        return annotated

    def _filter_by_roi(self, detections: list[dict]) -> list[dict]:
        """过滤不在 ROI 区域内的检测框（用框底部中心点判断）"""
        if not self.roi_polygon:
            return detections

        pts = np.array(self.roi_polygon, dtype=np.int32)
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center_x = (x1 + x2) // 2
            bottom_y = y2
            if cv2.pointPolygonTest(pts, (float(center_x), float(bottom_y)), False) >= 0:
                filtered.append(det)
        return filtered

    def _draw_roi(self, frame: np.ndarray):
        """绘制 ROI 区域边框"""
        if not self.roi_polygon:
            return
        pts = np.array(self.roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts.reshape((-1, 1, 2))], (255, 200, 0, 40))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
        px, py = self.roi_polygon[0][0] + 5, self.roi_polygon[0][1] - 4
        put_text_cn(frame, "水域", (px, py), 16, (255, 200, 0), anchor="ls")

    def _draw_alarm_banner(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        blink = int(time.time() * 4) % 2 == 0
        if blink:
            warning = f"!! 溺水警报 — 共 {self.drowning_confirmed_count} 人 !!"
            put_text_cn_center(frame, warning, 40, 26, (255, 255, 255))

    def _draw_info_panel(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (248, 118), (20, 22, 28), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        fs = 17
        cyan = (180, 220, 255)
        put_text_cn(frame, f"帧率 {self.fps:.1f}", (14, 28), fs, cyan, anchor="ls")
        put_text_cn(frame, f"人数 {self.person_count}", (14, 54), fs, cyan, anchor="ls")

        status_color = (80, 80, 255) if self.drowning_confirmed_count > 0 else (100, 220, 120)
        if self.drowning_confirmed_count > 0:
            status_text = f"状态 溺水 ×{self.drowning_confirmed_count}"
        else:
            status_text = "状态 正常"
        put_text_cn(frame, status_text, (14, 80), fs, status_color, anchor="ls")

        if self.logger.is_recording:
            cv2.circle(frame, (w - 25, 25), 8, (0, 0, 255), -1)
            put_text_cn(frame, "录像", (w - 72, 30), 15, (0, 0, 255), anchor="ls")

    def reset(self):
        self.tracker.reset()
        self.logger.stop_clip_recording()
        self._alarm_logged = False

    def release(self):
        self.logger.release()
        self.alarm.stop()
