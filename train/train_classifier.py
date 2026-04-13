"""
分类模型训练脚本 - 适配 RTX 3050 (4GB VRAM)

硬件: RTX 3050 + i5-11260默认使用 GPU（CUDA）；训练结束会保存并自动打开训练曲线图。

使用方式:
    1. 先准备数据集: python -m train.prepare_dataset
    2. 再训练（默认轻量骨干）: python -m train.train_classifier
    3. 更强结构（C2f 颈部 + ResNet18，样本量不变）:
       python -m train.train_classifier --backbone resnet18_c2f --c2f_depth 2
       显存紧张可加: --batch_size 16
    4. 仅 CPU: python -m train.train_classifier --device cpu
    5. 不自动弹图: python -m train.train_classifier --no-show-chart
"""

import os
import sys
import subprocess
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from models.classifier_arch import (
    build_classifier_model,
    freeze_for_transfer,
    unfreeze_all,
)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_weighted_sampler(dataset):
    """计算类别权重，解决样本不平衡问题 (class1 只有584个)"""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total = sum(class_counts.values())

    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  训练", ncols=90)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return running_loss / total, correct / total


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  验证", ncols=90):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def _open_file(path: str) -> None:
    """用系统默认程序打开文件（图表 PNG）"""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except OSError as e:
        print(f"[提示] 无法自动打开图表: {e}，请手动打开: {path}")


def plot_history(history, class_names, save_dir="weights", show_chart: bool = True):
    """
    绘制并保存训练过程曲线；show_chart=True 时在完成后用系统默认看图软件打开。
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "training_history.png")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "分类模型训练过程 · Training History",
        fontsize=15,
        fontweight="600",
        color="#1d1d1f",
        y=1.02,
    )
    cls_line = " / ".join(class_names)
    fig.text(0.5, 0.97, cls_line, ha="center", fontsize=10, color="#86868b")

    colors = {"train": "#0071e3", "val": "#34c759"}

    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="-", alpha=0.25, color="#c7c7cc")
        ax.set_xlabel("Epoch", fontsize=11, color="#424245")
        for spine in ax.spines.values():
            spine.set_color("#d2d2d7")

    axes[0].plot(epochs, history["train_loss"], label="Train loss", color=colors["train"], linewidth=2.2)
    axes[0].plot(epochs, history["val_loss"], label="Val loss", color=colors["val"], linewidth=2.2)
    axes[0].set_title("Loss", fontsize=13, fontweight="600", color="#1d1d1f", pad=8)
    axes[0].legend(frameon=False, loc="upper right", fontsize=10)

    axes[1].plot(epochs, history["train_acc"], label="Train acc", color=colors["train"], linewidth=2.2)
    axes[1].plot(epochs, history["val_acc"], label="Val acc", color=colors["val"], linewidth=2.2)
    axes[1].set_title("Accuracy", fontsize=13, fontweight="600", color="#1d1d1f", pad=8)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, loc="lower right", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"\n训练曲线已保存: {os.path.abspath(out_path)}")
    if show_chart:
        print("正在打开图表…")
        _open_file(out_path)

    return out_path


def print_per_class_accuracy(preds, labels, class_names):
    from sklearn.metrics import classification_report
    print("\n" + "=" * 55)
    print("各类别详细指标:")
    print("=" * 55)
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def main():
    parser = argparse.ArgumentParser(description="训练溺水状态分类模型 (RTX 3050 优化)")
    parser.add_argument("--data_dir", default="data_cls", help="分类数据集目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="weights/classifier_best.pth")
    parser.add_argument(
        "--backbone",
        default="mobilenet_v3_small",
        choices=[
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "mobilenet_v3_small_c2f",
            "mobilenet_v3_large_c2f",
            "resnet18_c2f",
            "resnet50_c2f",
        ],
        help="骨干网络；带 _c2f 的在特征图后接 YOLO 风格 C2f 颈部，表达能力更强、显存更大",
    )
    parser.add_argument("--c2f_depth", type=int, default=2, help="C2f 颈部内 Bottleneck 重复次数（仅 *_c2f 有效）")
    parser.add_argument("--workers", type=int, default=-1,
                        help="DataLoader workers，-1 表示 Windows 用 0、其他系统用 4")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="训练设备，默认 cuda（需要 NVIDIA GPU + 正确驱动）",
    )
    parser.add_argument(
        "--no-show-chart",
        action="store_true",
        help="训练结束后不自动打开曲线图（仍会保存 PNG）",
    )
    args = parser.parse_args()

    if args.workers < 0:
        args.workers = 0 if os.name == "nt" else 4

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[错误] 已指定 --device cuda，但当前环境不可用 CUDA。")
            print("请确认: NVIDIA 驱动已安装，且已安装支持 CUDA 的 PyTorch。")
            print("或改用: torch 官网按 CUDA 版本重装; 临时 CPU 训练请加参数 --device cpu")
            sys.exit(1)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    use_amp = device.type == "cuda"
    print(f"{'='*55}")
    print(f"  防溺水监测 - 分类模型训练")
    print(f"  设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存: {vram:.1f} GB")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  AMP: {'开' if use_amp else '关（CPU）'}")
    print(f"  Backbone: {args.backbone}" + (f"  C2f_depth={args.c2f_depth}" if "c2f" in args.backbone else ""))
    print(f"{'='*55}\n")

    train_transform, val_transform = get_transforms()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir):
        print(f"[错误] 训练集目录不存在: {train_dir}")
        print("请先运行: python -m train.prepare_dataset")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if len(val_dataset) == 0:
        print(f"[错误] 验证集为空: {val_dir}")
        print("请先运行: python -m train.prepare_dataset")
        return

    class_names = train_dataset.classes
    val_names = val_dataset.classes
    if val_names != class_names:
        print("[错误] 训练集与验证集类别不一致:")
        print(f"  train: {class_names}")
        print(f"  val:   {val_names}")
        return
    print(f"类别映射: {train_dataset.class_to_idx}")
    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")

    targets = [s[1] for s in train_dataset.samples]
    class_counts = Counter(targets)
    print("训练集各类别数量:")
    for cls_idx, name in enumerate(class_names):
        print(f"  {name}: {class_counts.get(cls_idx, 0)}")

    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    num_classes = len(class_names)
    model = build_classifier_model(
        args.backbone,
        num_classes,
        c2f_depth=args.c2f_depth,
        pretrained=True,
    ).to(device)
    freeze_for_transfer(model, args.backbone)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {total_params:,} (可训练: {trainable_params:,})")

    class_weight_tensor = torch.tensor(
        [sum(class_counts.values()) / class_counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    class_weight_tensor = class_weight_tensor / class_weight_tensor.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    patience_counter = 0
    early_stop_patience = 12
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 第5个epoch解冻全部参数
    unfreeze_epoch = 5

    for epoch in range(args.epochs):
        if epoch == unfreeze_epoch:
            print(f"\n>> Epoch {epoch+1}: 解冻全部参数进行微调")
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - unfreeze_epoch, eta_min=1e-6
            )

        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{args.epochs}  (lr={lr:.6f})")
        print("-" * 45)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "c2f_depth": args.c2f_depth,
                    "num_classes": num_classes,
                },
                args.output,
            )
            print(f"  >> 保存最优模型 (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n连续 {early_stop_patience} 轮无提升，提前停止训练")
                break

    # 训练结束，加载最佳模型做最终评估
    if not os.path.isfile(args.output):
        print(f"[警告] 未生成权重文件，保存当前模型到 {args.output}")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "backbone": args.backbone,
                "c2f_depth": args.c2f_depth,
                "num_classes": num_classes,
            },
            args.output,
        )
    _ckpt = torch.load(args.output, map_location=device, weights_only=False)
    if isinstance(_ckpt, dict) and "state_dict" in _ckpt:
        model.load_state_dict(_ckpt["state_dict"])
    else:
        model.load_state_dict(_ckpt)
    _, final_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, device, use_amp
    )

    print_per_class_accuracy(final_preds, final_labels, class_names)
    plot_history(history, class_names)

    print(f"\n{'='*55}")
    print(f"  训练完成!")
    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    print(f"  模型保存于: {os.path.abspath(args.output)}")
    print(f"  训练曲线: weights/training_history.png")
    print(f"{'='*55}")
    print(f"\n启动监测系统:")
    print(f"  python main.py --classifier {args.output}")


if __name__ == "__main__":
    main()
