"""
分类器结构：可选 ResNet / MobileNet + YOLOv8 风格 C2f 颈部，增强特征表达（不增加数据量）。

旧权重为纯 state_dict 时仅兼容 mobilenet_v3_small；新训练保存含 backbone 等字段的完整 checkpoint。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ConvBNSiLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BottleneckC2f(nn.Module):
    """C2f 分支内残差块（深度可分离卷积风格的小卷积塔）。"""

    def __init__(self, c: int):
        super().__init__()
        self.cv1 = ConvBNSiLU(c, c, 3, 1, 1)
        self.cv2 = ConvBNSiLU(c, c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class C2fNeck(nn.Module):
    """
    YOLOv8 风格 C2f：1x1 升维 → 沿通道一分为二 → 一半经 n 个 Bottleneck → concat → 1x1 压回 c_out。
    """

    def __init__(self, c_in: int, c_out: int, n: int = 2):
        super().__init__()
        hidden = max(c_in // 2, 64)
        self.cv1 = ConvBNSiLU(c_in, 2 * hidden, 1, 1, 0)
        self.blocks = nn.Sequential(*[BottleneckC2f(hidden) for _ in range(n)])
        self.cv2 = ConvBNSiLU(2 * hidden, c_out, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        a, b = torch.chunk(y, 2, dim=1)
        b = self.blocks(b)
        return self.cv2(torch.cat([a, b], dim=1))


class ResNetC2fClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        c2f_depth: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.resnet18(weights=w)
            self._feat_dim = 512
        elif backbone == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.resnet50(weights=w)
            self._feat_dim = 2048
        else:
            raise ValueError(f"不支持的 ResNet: {backbone}")

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.neck = C2fNeck(self._feat_dim, self._feat_dim, n=c2f_depth)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.neck(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


class MobileNetV3C2fClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        c2f_depth: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        if variant == "small":
            w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.mobilenet_v3_small(weights=w)
            self._dim = 576
        elif variant == "large":
            w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.mobilenet_v3_large(weights=w)
            self._dim = 960
        else:
            raise ValueError(variant)
        self.features = m.features
        self.neck = C2fNeck(self._dim, self._dim, n=c2f_depth)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.neck(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build_mobilenet_v3_small(num_classes: int, pretrained: bool = True) -> nn.Module:
    w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=w)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_mobilenet_v3_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=w)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def build_classifier_model(
    backbone: str,
    num_classes: int,
    *,
    c2f_depth: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    b = backbone.lower().strip()
    if b == "mobilenet_v3_small":
        return build_mobilenet_v3_small(num_classes, pretrained)
    if b == "mobilenet_v3_large":
        return build_mobilenet_v3_large(num_classes, pretrained)
    if b == "mobilenet_v3_small_c2f":
        return MobileNetV3C2fClassifier(num_classes, "small", c2f_depth, pretrained)
    if b == "mobilenet_v3_large_c2f":
        return MobileNetV3C2fClassifier(num_classes, "large", c2f_depth, pretrained)
    if b == "resnet18_c2f":
        return ResNetC2fClassifier(num_classes, "resnet18", c2f_depth, pretrained)
    if b == "resnet50_c2f":
        return ResNetC2fClassifier(num_classes, "resnet50", c2f_depth, pretrained)
    raise ValueError(
        f"未知 backbone: {backbone}。可选: mobilenet_v3_small, mobilenet_v3_large, "
        "mobilenet_v3_small_c2f, mobilenet_v3_large_c2f, resnet18_c2f, resnet50_c2f"
    )


def freeze_for_transfer(model: nn.Module, backbone: str) -> None:
    """前几层冻结，与 train_classifier 中 epoch 前若干轮策略配合。"""
    b = backbone.lower().strip()
    if b in ("mobilenet_v3_small", "mobilenet_v3_large"):
        n = 8 if "small" in b else 12
        for param in model.features[:n].parameters():
            param.requires_grad = False
    elif b in ("mobilenet_v3_small_c2f", "mobilenet_v3_large_c2f"):
        n = 8 if "small" in b else 12
        for param in model.features[:n].parameters():
            param.requires_grad = False
    elif b in ("resnet18_c2f", "resnet50_c2f"):
        for m in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3):
            for p in m.parameters():
                p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def load_classifier_from_checkpoint(
    path: str,
    map_location,
    num_classes: int = 3,
) -> tuple[nn.Module, dict | None]:
    """
    加载权重。返回 (model, meta_or_none)。
    meta 含 backbone / c2f_depth / num_classes；旧版 .pth 无 meta 时按 mobilenet_v3_small 加载。
    """
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = raw
        backbone = meta.get("backbone", "mobilenet_v3_small")
        nc = int(meta.get("num_classes", num_classes))
        depth = int(meta.get("c2f_depth", 2))
        model = build_classifier_model(backbone, nc, c2f_depth=depth, pretrained=False)
        model.load_state_dict(meta["state_dict"], strict=True)
        return model, meta
    model = build_classifier_model("mobilenet_v3_small", num_classes, pretrained=False)
    model.load_state_dict(raw, strict=True)
    return model, None
