"""状态分类模型模块 - 第二阶段：对裁剪出的人体区域进行溺水状态分类"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

from models.classifier_arch import load_classifier_from_checkpoint

# 类别顺序必须与数据集 data.yaml 一致: 0=Drowning, 1=Person out of water, 2=Swimming
CLASS_NAMES = ["drowning", "out_of_water", "swimming"]
CLASS_NAMES_CN = {
    "drowning": "溺水",
    "out_of_water": "人员离水",
    "swimming": "游泳中",
}

COLORS = {
    "drowning": (0, 0, 255),       # 红色 (BGR)
    "swimming": (0, 255, 0),       # 绿色
    "out_of_water": (255, 165, 0), # 橙色
}


class StateClassifier:
    def __init__(self, model_path=None, num_classes=3, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.model = None

        if model_path and os.path.exists(model_path):
            self.model, meta = load_classifier_from_checkpoint(
                model_path, self.device, num_classes=num_classes
            )
            if meta:
                print(f"[分类器] 已加载模型: {model_path} ({meta.get('backbone', '?')})")
            else:
                print(f"[分类器] 已加载模型权重(旧版): {model_path}")
        else:
            from models.classifier_arch import build_classifier_model

            self.model = build_classifier_model("mobilenet_v3_small", num_classes, pretrained=True)
            print("[分类器] 未加载自定义权重，使用 ImageNet 预训练参数")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, crop: np.ndarray) -> tuple[str, float]:
        """对单个裁剪图像进行分类，返回 (类别名称, 置信度)"""
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        class_name = CLASS_NAMES[pred_idx.item()]
        return class_name, conf.item()

    def predict_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """批量预测"""
        if not crops:
            return []

        tensors = []
        for crop in crops:
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(image))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, pred_indices = torch.max(probs, dim=1)

        results = []
        for i in range(len(crops)):
            class_name = CLASS_NAMES[pred_indices[i].item()]
            results.append((class_name, confs[i].item()))

        return results
