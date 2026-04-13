"""
数据集准备脚本
从 YOLO 目标检测格式的数据集中，裁剪人体区域，生成分类数据集。

你的数据集结构:
    data/
    ├── images/{train,val,test}/  *.jpg
    ├── labels/{train,val,test}/  *.txt  (YOLO格式: class cx cy w h)
    └── data.yaml

类别映射 (与 data.yaml 一致):
    0 = Drowning
    1 = Person out of water
    2 = Swimming

使用方式:
    python -m train.prepare_dataset
    python -m train.prepare_dataset --data_root data --output_dir data_cls
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter


CLASS_MAP = {0: "drowning", 1: "out_of_water", 2: "swimming"}


def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> list[dict]:
    results = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            class_name = CLASS_MAP.get(cls_id)
            if class_name and x2 > x1 and y2 > y1:
                results.append({"class_name": class_name, "bbox": (x1, y1, x2, y2)})
    return results


def prepare_split(images_dir: Path, labels_dir: Path, output_dir: Path, split_name: str):
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files:
        print(f"  [跳过] {split_name}: 未找到图片")
        return

    counter = Counter()
    crop_id = 0

    for img_path in tqdm(image_files, desc=f"  {split_name}"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        annotations = parse_yolo_label(str(label_path), w, h)

        for ann in annotations:
            cls = ann["class_name"]
            x1, y1, x2, y2 = ann["bbox"]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            out_dir = output_dir / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{cls}_{crop_id:06d}.jpg"
            cv2.imwrite(str(out_path), crop)
            crop_id += 1
            counter[cls] += 1

    print(f"  {split_name} 完成:")
    for cls in CLASS_MAP.values():
        print(f"    {cls}: {counter.get(cls, 0)} 张")
    return counter


def main():
    parser = argparse.ArgumentParser(description="从YOLO检测数据集生成分类数据集")
    parser.add_argument("--data_root", default="data", help="YOLO数据集根目录")
    parser.add_argument("--output_dir", default="data_cls", help="分类数据集输出目录")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    print("=" * 50)
    print("从 YOLO 检测标注生成分类数据集")
    print(f"  数据源: {data_root.absolute()}")
    print(f"  输出到: {output_dir.absolute()}")
    print("=" * 50)

    total = Counter()
    for split in ["train", "val", "test"]:
        images_dir = data_root / "images" / split
        labels_dir = data_root / "labels" / split
        if not images_dir.exists():
            print(f"  [跳过] {split}: 目录不存在")
            continue
        result = prepare_split(images_dir, labels_dir, output_dir, split)
        if result:
            total += result

    print("\n" + "=" * 50)
    print("汇总:")
    for cls in CLASS_MAP.values():
        print(f"  {cls}: {total.get(cls, 0)} 张")
    print(f"  总计: {sum(total.values())} 张裁剪图像")
    print("=" * 50)
    print(f"\n分类数据集已生成到: {output_dir.absolute()}")
    print("接下来可以运行训练:")
    print(f"  python -m train.train_classifier --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
