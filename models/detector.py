"""YOLOv8 人体检测模块 - 第一阶段：检测画面中的人体边界框"""

from ultralytics import YOLO
import numpy as np


class PersonDetector:
    PERSON_CLASS_ID = 0  # COCO 数据集中 person 类别的 ID

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5, device="auto"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        检测画面中的人体，返回边界框列表。
        每个元素: {"bbox": (x1, y1, x2, y2), "confidence": float}
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                })

        return detections

    def crop_persons(self, frame: np.ndarray, detections: list[dict]) -> list[np.ndarray]:
        """根据检测结果裁剪出人体区域图像"""
        crops = []
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
        return crops
