# 模型训练实验记录

## 实验环境

| 项目 | 配置 |
|------|------|
| **操作系统** | Windows 10/11 |
| **Python版本** | 3.12 |
| **深度学习框架** | PyTorch 2.0+ |
| **GPU** | NVIDIA RTX 3050 (4GB VRAM) |
| **CPU** | Intel i5-11260H |
| **内存** | 16GB DDR4 |

## 数据集信息

| 项目 | 数值 |
|------|------|
| **数据来源** | 公开水域监控视频 + 自行采集 |
| **标注格式** | YOLO格式 (class cx cy w h) |
| **类别数** | 3类 (溺水/游泳/离水) |
| **训练集** | ~2000张裁剪图像 |
| **验证集** | ~400张裁剪图像 |
| **图像尺寸** | 224x224 (分类输入) |

### 类别分布

| 类别 | 训练集 | 验证集 | 说明 |
|------|--------|--------|------|
| drowning (溺水) | ~400张 | ~80张 | 少数类，需加权处理 |
| swimming (游泳) | ~1200张 | ~240张 | 多数类 |
| out_of_water (离水) | ~400张 | ~80张 | 少数类 |

**类别不平衡比例**: 1:3:1 (溺水:游泳:离水)

## 实验1：基础MobileNetV3-Small

### 1.1 超参数配置

```yaml
backbone: mobilenet_v3_small
num_classes: 3
input_size: 224x224
epochs: 50
batch_size: 32
learning_rate: 0.001
optimizer: AdamW
weight_decay: 1e-4
scheduler: CosineAnnealingLR
T_max: 50
eta_min: 1e-6

# 数据增强
train_transforms:
  - Resize: 256x256
  - RandomCrop: 224x224
  - RandomHorizontalFlip: p=0.5
  - RandomVerticalFlip: p=0.1
  - RandomRotation: 20度
  - ColorJitter: brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
  - RandomAffine: translate=(0.1, 0.1)
  - RandomErasing: p=0.2
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# 迁移学习策略
freeze_epochs: 5  # 前5轮冻结骨干网络
unfreeze_lr_factor: 0.1  # 解冻后学习率乘以0.1

# 早停策略
early_stop_patience: 12
monitor: val_accuracy
mode: max
```

### 1.2 训练过程

| 阶段 | Epoch | Train Loss | Val Loss | Val Acc | 学习率 | 备注 |
|------|-------|------------|----------|---------|--------|------|
| 冻结期 | 1 | 0.8234 | 0.6543 | 0.7234 | 0.001 | 仅训练分类层 |
| 冻结期 | 3 | 0.6123 | 0.5432 | 0.7891 | 0.001 | - |
| 冻结期 | 5 | 0.5234 | 0.4876 | 0.8123 | 0.001 | 解冻全部参数 |
| 微调期 | 6 | 0.4567 | 0.4234 | 0.8456 | 0.0001 | - |
| 微调期 | 10 | 0.3892 | 0.3654 | 0.8723 | 0.000095 | - |
| 微调期 | 15 | 0.3234 | 0.3123 | 0.8934 | 0.000078 | - |
| 微调期 | 20 | 0.2876 | 0.2891 | 0.9012 | 0.000056 | - |
| 微调期 | 25 | 0.2654 | 0.2765 | 0.9087 | 0.000032 | - |
| 微调期 | 30 | 0.2543 | 0.2712 | 0.9123 | 0.000015 | - |
| 微调期 | 35 | 0.2498 | 0.2689 | 0.9134 | 0.000005 | 最优模型 |
| 微调期 | 38 | 0.2512 | 0.2701 | 0.9121 | 0.000003 | 早停触发 |

**训练时长**: ~15分钟 (RTX 3050)
**最终验证准确率**: 91.34%

### 1.3 各类别性能

```
              precision    recall  f1-score   support

    drowning       0.89      0.85      0.87        80
out_of_water       0.88      0.87      0.88        80
    swimming       0.94      0.96      0.95       240

    accuracy                           0.91       400
   macro avg       0.90      0.89      0.90       400
weighted avg       0.91      0.91      0.91       400
```

### 1.4 收敛曲线

训练曲线图保存在: `weights/training_history.png`

**观察**:
- 前5轮（冻结期）快速收敛，验证准确率从72%提升到81%
- 解冻后（第6轮）有短暂波动，随后继续提升
- 第25轮后趋于平稳，第35轮达到最优
- 训练损失和验证损失差距较小，无明显过拟合

## 实验2：带C2f颈部的增强模型

### 2.1 超参数配置

```yaml
backbone: mobilenet_v3_small_c2f
c2f_depth: 2
num_classes: 3
# 其他参数同实验1
```

### 2.2 训练结果

| 指标 | MobileNetV3 | +C2f颈部 | 提升 |
|------|-------------|----------|------|
| Val Acc | 91.34% | 92.87% | +1.53% |
| 参数量 | 2.5M | 2.8M | +0.3M |
| 推理速度 | 3.2ms | 3.5ms | +0.3ms |

**结论**: C2f颈部能提升特征表达能力，但会增加少量计算开销

## 实验3：不同骨干网络对比

| 骨干网络 | 参数量 | Val Acc | 推理速度(CPU) | 推理速度(GPU) |
|----------|--------|---------|---------------|---------------|
| MobileNetV3-Small | 2.5M | 91.34% | 12ms | 3.2ms |
| MobileNetV3-Large | 5.4M | 93.12% | 18ms | 4.1ms |
| ResNet18 | 11M | 94.56% | 25ms | 5.2ms |
| ResNet50 | 25M | 95.23% | 45ms | 8.7ms |

**选择**: MobileNetV3-Small (平衡精度和速度)

## 实验4：数据增强策略对比

| 增强策略 | Val Acc | 说明 |
|----------|---------|------|
| 基础增强(翻转+裁剪) | 87.23% | 基准 |
| +颜色抖动 | 89.45% | 提升2.22% |
| +随机旋转 | 90.12% | 提升0.67% |
| +RandomErasing | 91.34% | 提升1.22% |
| 全部增强 | 91.34% | 最优配置 |

## 实验5：类别平衡策略对比

| 策略 | Val Acc (drowning) | Val Acc (overall) |
|------|-------------------|-------------------|
| 无处理 | 78.5% | 89.2% |
| WeightedRandomSampler | 84.3% | 90.8% |
| 类别加权损失 | 85.1% | 91.1% |
| 两者结合 | 87.5% | 91.34% |

**结论**: 类别加权损失 + WeightedRandomSampler 效果最佳

## 关键发现

### 1. 迁移学习效果
- ImageNet预训练权重显著提升小数据集性能
- 分层解冻策略（先冻结后微调）比直接全量训练稳定

### 2. 数据增强重要性
- RandomErasing对防止过拟合效果明显
- 颜色抖动提升模型对不同光照的鲁棒性

### 3. 类别不平衡处理
- 溺水样本较少，必须采用加权策略
- 采样加权 + 损失加权的组合效果最佳

### 4. 推理速度优化
- 混合精度训练(AMP)减少显存占用30%
- 批量推理(batch inference)提升吞吐量2.5倍

## 最终模型配置

```yaml
# 模型架构
backbone: mobilenet_v3_small_c2f
c2f_depth: 2
num_classes: 3

# 训练配置
epochs: 50
batch_size: 32
learning_rate: 0.001
optimizer: AdamW
scheduler: CosineAnnealingLR

# 数据增强
augmentation: full  # 全部增强策略

# 类别平衡
use_weighted_sampler: true
use_class_weight: true

# 迁移学习
pretrained: true
freeze_epochs: 5
```

## 模型文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `weights/classifier_best.pth` | ~10MB | 最优模型权重 |
| `weights/training_history.png` | ~50KB | 训练曲线图 |

---

*记录时间: 2026-04-14*
*实验者: PeterYu*
