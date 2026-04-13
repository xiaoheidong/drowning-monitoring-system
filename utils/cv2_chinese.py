"""OpenCV 的 putText 不支持中文，使用 Pillow 在 BGR 图像上绘制中文/英文。"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH: str | None = None
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _resolve_font_path() -> str | None:
    global _FONT_PATH
    if _FONT_PATH is not None:
        return _FONT_PATH
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        Path(windir) / "Fonts" / "msyh.ttc",
        Path(windir) / "Fonts" / "msyhbd.ttc",
        Path(windir) / "Fonts" / "simhei.ttf",
        Path(windir) / "Fonts" / "simsun.ttc",
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for p in candidates:
        if p.is_file():
            _FONT_PATH = str(p)
            return _FONT_PATH
    _FONT_PATH = ""
    return None


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _resolve_font_path()
    if not path:
        return ImageFont.load_default()
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def _bgr_to_rgb_fill(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return (bgr[2], bgr[1], bgr[0])


def put_text_cn(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int,
    color_bgr: tuple[int, int, int],
    *,
    anchor: str = "ls",
) -> None:
    """
    在 BGR 图像上绘制文字。org 为参考点坐标；anchor 与 Pillow 一致，默认 ls=左侧基线（对齐原 OpenCV 习惯）。
    """
    if not text:
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(font_size)
    fill = _bgr_to_rgb_fill(color_bgr)
    draw.text(org, text, font=font, fill=fill, anchor=anchor)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, out)


def put_text_cn_with_bg(
    img: np.ndarray,
    text: str,
    org_xy: tuple[int, int],
    font_size: int,
    bg_bgr: tuple[int, int, int],
    fg_bgr: tuple[int, int, int],
) -> None:
    """
    在文字背后画实心矩形条（用于检测框标签）。
    org_xy: 文字左上角 (x, y)，与旧代码中矩形条 + 白字布局一致。
    """
    if not text:
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(font_size)
    x, y = org_xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x, pad_y = 2, 4
    rx0 = bbox[0] - pad_x
    ry0 = bbox[1] - pad_y
    rx1 = bbox[2] + pad_x
    ry1 = bbox[3] + pad_y
    draw.rectangle([rx0, ry0, rx1, ry1], fill=_bgr_to_rgb_fill(bg_bgr))
    draw.text((x, y), text, font=font, fill=_bgr_to_rgb_fill(fg_bgr))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, out)


def text_size_cn(text: str, font_size: int) -> tuple[int, int]:
    """返回 (宽, 高) 近似，用于布局。"""
    path = _resolve_font_path()
    if not path:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        return (tw, th)
    font = get_font(font_size)
    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def put_text_cn_center(
    img: np.ndarray,
    text: str,
    y_baseline: int,
    font_size: int,
    color_bgr: tuple[int, int, int],
) -> None:
    """水平居中绘制一行文字（基线纵坐标为 y_baseline）。"""
    tw, _ = text_size_cn(text, font_size)
    h, w = img.shape[:2]
    x = max(8, (w - tw) // 2)
    put_text_cn(img, text, (x, y_baseline), font_size, color_bgr, anchor="ls")
