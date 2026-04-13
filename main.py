"""
防溺水智能监测系统 v2.0 - 主入口

功能:
    - YOLOv8 人体检测 + MobileNetV3 状态分类（两阶段）
    - 摄像头实时监测 / 视频文件检测
    - 连续帧时序判定，降低误报
    - 声音报警 + 系统托盘通知
    - 自动截图 + 视频片段录制
    - ROI 水域区域设定
    - 事件日志管理

配置:
    复制 config/settings.example.json 为 config/settings.json，填写 deepseek.api_key、确认 classifier.path。

启动方式:
    python main.py
    python main.py --classifier weights/classifier_best.pth
    python main.py --detector yolov8s.pt --conf 0.6 --confirm 10

Web 看板（可选，独立进程，不影响桌面）:
    python -m web
    或双击 run_web_dashboard.bat
"""

import sys
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from utils.video_processor import VideoProcessor
from utils.settings import resolve_classifier_path
from ui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="防溺水智能监测系统 v2.0")
    parser.add_argument("--detector", default="yolov8n.pt",
                        help="YOLOv8 检测模型路径 (默认 yolov8n.pt)")
    parser.add_argument("--classifier", default=None,
                        help="分类模型权重；省略则使用 config/settings.json 中的 classifier.path")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="检测置信度阈值 (默认 0.5)")
    parser.add_argument("--confirm", type=int, default=8,
                        help="溺水确认所需连续帧数 (默认 8)")
    parser.add_argument("--device", default="auto",
                        help="推理设备 (auto/cpu/cuda)")
    args = parser.parse_args()

    classifier_path = resolve_classifier_path(args.classifier)

    processor = VideoProcessor(
        detector_model=args.detector,
        classifier_model=classifier_path,
        det_conf=args.conf,
        device=args.device,
        confirm_frames=args.confirm,
    )

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    app.setQuitOnLastWindowClosed(True)

    window = MainWindow(processor)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
