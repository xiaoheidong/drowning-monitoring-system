# 🌊 防溺水智能监测系统 v2.0

基于深度学习的实时溺水检测系统，采用YOLOv8目标检测 + MobileNetV3状态分类的两阶段架构，集成AI智能分析和大数据处理能力。

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 功能特性

### 核心功能
- ✅ **实时检测**: 支持摄像头和视频文件输入，GPU加速推理
- ✅ **两阶段检测**: YOLOv8人体检测 + MobileNetV3状态分类
- ✅ **时序追踪**: IOU匹配 + 连续帧确认机制，有效降低误报
- ✅ **智能报警**: 声音报警 + 截图保存 + 自动录像（最长30秒）
- ✅ **ROI设置**: 自定义监测区域，减少无效检测
- ✅ **Web看板**: FastAPI提供历史事件查询、统计分析和录像回放
- ✅ **AI简报**: DeepSeek大模型智能分析事件趋势

### 大数据功能（可选）
- 📊 **数据分析**: 日报/周报自动生成，溺水高发时段分析
- 🔍 **Elasticsearch**: 事件存储与全文检索（预留接口）
- 📈 **可视化**: 趋势图、分布图、热力图自动生成
- 📑 **报告导出**: HTML格式报告，含图表和统计数据

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Windows 10/11 或 Linux
- **推荐**: NVIDIA GPU + CUDA 11.8+ (支持RTX 3050及以上)
- **最低**: CPU模式（推理速度较慢）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/xiaoheidong/drowning-monitoring-system.git
cd drowning-monitoring-system
```

2. **创建虚拟环境**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置系统**
```bash
# 复制配置文件模板
copy config\settings.example.json config\settings.json

# 编辑 settings.json，配置以下参数：
# - deepseek.api_key: DeepSeek API密钥（可选，用于AI简报）
# - dingtalk: 钉钉机器人配置（可选，用于报警通知）
```

5. **运行系统**
```bash
# 启动桌面应用程序（GPU模式）
python main.py

# 或指定设备
python main.py --device cpu  # CPU模式
python main.py --device 0    # 指定GPU

# 启动Web看板（可选，另一个终端）
python -m web
```

## 📁 项目结构

```
drowning-monitoring-system/
├── main.py                      # 主入口
├── requirements.txt             # 依赖列表
├── config/
│   ├── settings.example.json   # 配置模板
│   └── settings.json           # 本地配置（gitignore保护）
├── models/                      # 模型定义
│   ├── detector.py             # YOLOv8检测器
│   ├── classifier.py           # MobileNetV3分类器
│   └── classifier_arch.py      # 网络架构（含C2f颈部）
├── utils/                       # 工具模块
│   ├── video_processor.py      # 视频处理流水线
│   ├── drowning_tracker.py     # 时序追踪器
│   ├── event_logger.py         # 事件日志（CSV/JSONL双格式）
│   ├── alarm.py                # 报警管理
│   └── deepseek_client.py      # DeepSeek AI客户端
├── ui/                          # 用户界面
│   └── main_window.py          # PyQt5主窗口（深色主题）
├── web/                         # Web服务
│   ├── server.py               # FastAPI服务
│   ├── dashboard.py            # 数据看板
│   └── static/                 # 静态文件
├── train/                       # 训练模块
│   ├── prepare_dataset.py      # 数据准备
│   └── train_classifier.py     # 模型训练
├── analytics/                   # 数据分析（新增）
│   ├── event_analyzer.py       # 分析核心
│   ├── daily_report.py         # 日报生成
│   └── weekly_report.py        # 周报生成
├── bigdata/                     # 大数据组件（新增）
│   └── elasticsearch_client.py # ES客户端（可选）
├── docs/                        # 技术文档（新增）
│   └── architecture.md         # 架构图和数据流图
├── experiments/                 # 实验记录（新增）
│   └── experiment_log.md       # 详细训练记录
├── logs/                        # 日志目录（gitignore）
│   ├── events.jsonl            # 结构化事件日志
│   ├── events.csv              # 表格格式日志
│   ├── screenshots/            # 报警截图
│   └── clips/                  # 报警录像
└── weights/                     # 模型权重（gitignore）
    ├── classifier_best.pth     # 最优分类模型
    └── training_history.png    # 训练曲线
```

## 🎯 使用指南

### 1. 实时监测
```bash
# 基本使用
python main.py

# 使用指定模型
python main.py --classifier weights/classifier_best.pth

# 调整检测阈值和确认帧数
python main.py --conf 0.6 --confirm 10

# CPU模式（无GPU时）
python main.py --device cpu
```

**界面操作**:
- 点击"开始监测"启动摄像头检测
- 点击"选择视频"检测本地视频文件
- 点击"设置ROI"自定义监测区域
- 点击"生成事件简报"使用AI分析近期事件

### 2. 模型训练
```bash
# 准备数据集（从YOLO格式裁剪）
python -m train.prepare_dataset

# 训练分类模型
python -m train.train_classifier

# 使用更强的骨干网络
python -m train.train_classifier --backbone resnet18_c2f

# 查看训练曲线
# weights/training_history.png
```

### 3. 数据分析与报告
```bash
# 生成日报
python -m analytics.daily_report

# 生成周报
python -m analytics.weekly_report

# 查看输出报告
# analytics/output/daily_report_YYYYMMDD.html
# analytics/output/weekly_report_YYYYMMDD.html
```

**报告内容**:
- 基本统计（总事件数、溺水报警、检测人数）
- 高发时段分析（小时级分布图）
- 置信度统计
- 趋势分析（日报/周报）

### 4. Web看板
```bash
# 启动Web服务
python -m web

# 访问 http://localhost:8080
```

**API接口**:
- `GET /api/events` - 获取事件列表
- `GET /api/stats` - 获取统计数据
- `GET /api/stats/hourly` - 获取小时级统计
- `GET /files/screenshots/{filename}` - 获取截图

### 5. 大数据组件（可选）

#### Elasticsearch集成
```bash
# 1. 安装Elasticsearch（Docker方式）
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.0.0

# 2. 安装Python客户端
pip install elasticsearch>=8.0.0

# 3. 启用配置（settings.json）
{
  "elasticsearch": {
    "enabled": true,
    "host": "localhost",
    "port": 9200
  }
}
```

## 🔧 配置说明

编辑 `config/settings.json`:

```json
{
  "classifier": {
    "path": "weights/classifier_best.pth"
  },
  "deepseek": {
    "api_key": "your-deepseek-api-key",
    "api_base": "https://api.deepseek.com",
    "model": "deepseek-chat"
  },
  "dingtalk": {
    "enabled": false,
    "webhook": "https://oapi.dingtalk.com/robot/send?access_token=xxx",
    "secret": "SECxxx"
  },
  "logs": {
    "output_dir": "logs"
  },
  "elasticsearch": {
    "enabled": false,
    "host": "localhost",
    "port": 9200,
    "index": "drowning-events"
  }
}
```

**配置项说明**:
- `classifier.path`: 分类模型路径
- `deepseek.api_key`: DeepSeek API密钥（可选）
- `dingtalk`: 钉钉机器人配置（可选）
- `elasticsearch`: ES配置（可选，默认禁用）

## 📊 性能指标

| 指标 | 数值 | 测试环境 |
|------|------|----------|
| 检测准确率 | 91.34% | 验证集400张 |
| 推理速度 (GPU) | ~25 FPS | RTX 3050 |
| 推理速度 (CPU) | ~8 FPS | i5-11260H |
| 模型大小 | ~10 MB | MobileNetV3-Small |
| 训练时长 | ~15分钟 | RTX 3050 |

### 各类别性能
```
              precision    recall  f1-score   support
    drowning       0.89      0.85      0.87        80
out_of_water       0.88      0.87      0.88        80
    swimming       0.94      0.96      0.95       240
    accuracy                           0.91       400
```

## 🏗️ 系统架构

### 核心流程
```
视频输入 → YOLOv8检测 → MobileNetV3分类 → 时序追踪 → 报警/日志
                ↓              ↓              ↓
           人体边界框      状态类别      连续帧确认
                                            ↓
                                    溺水确认 → 报警 + 截图 + 录像
                                                    ↓
                                            CSV/JSONL/ES存储
                                                    ↓
                                            Web看板/数据分析
```

### 架构特点
- **模块化设计**: 检测、分类、追踪、报警独立模块
- **可选扩展**: ES、DeepSeek等组件可插拔
- **双格式日志**: CSV便于查看，JSONL便于程序处理
- **前后端分离**: PyQt5桌面端 + FastAPI Web服务

详细架构文档: [docs/architecture.md](docs/architecture.md)

## 📝 实验记录

训练实验记录: [experiments/experiment_log.md](experiments/experiment_log.md)

### 关键实验
1. **骨干网络对比**: MobileNetV3-Small vs ResNet18/50
2. **C2f颈部增强**: 准确率提升1.53%
3. **数据增强策略**: RandomErasing效果最佳
4. **类别平衡**: WeightedSampler + 加权损失

### 最优配置
- 骨干网络: MobileNetV3-Small + C2f颈部
- 输入尺寸: 224x224
- 优化器: AdamW (lr=0.001)
- 学习率调度: CosineAnnealingLR
- 数据增强: 翻转 + 旋转 + 颜色抖动 + RandomErasing

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习 | PyTorch + YOLOv8 + MobileNetV3 | 目标检测与分类 |
| GUI | PyQt5 | 桌面应用程序 |
| Web服务 | FastAPI + Uvicorn | RESTful API |
| 数据处理 | Pandas + NumPy | 数据分析 |
| 可视化 | Matplotlib + Seaborn | 图表生成 |
| AI分析 | DeepSeek API | 事件智能总结 |
| 大数据 | Elasticsearch (可选) | 事件存储与检索 |
| 视频处理 | OpenCV | 视频流处理 |

## 🤝 贡献指南

欢迎提交Issue和Pull Request!

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

MIT License

Copyright (c) 2026 PeterYu

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 视频处理
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
- [DeepSeek](https://deepseek.com/) - AI大模型

## 📞 联系方式

- 项目地址: https://github.com/xiaoheidong/drowning-monitoring-system
- 问题反馈: https://github.com/xiaoheidong/drowning-monitoring-system/issues

---

**毕业设计项目** | 数据科学与大数据技术 | 湖南科技学院 | 2026

**指导教师**: [教师姓名]  
**学生**: PeterYu  
**完成时间**: 2026年4月
