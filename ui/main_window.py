"""
防溺水智能监测系统 - 主界面
设计语言: Apple-inspired dark minimal · 毛玻璃卡片 · 大留白 · 克制用色
"""

from __future__ import annotations

import cv2
import os
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QSlider, QSpinBox,
    QMessageBox, QFrame, QTabWidget, QTableWidget, QTableWidgetItem,
    QCheckBox, QHeaderView, QSystemTrayIcon, QMenu, QAction, QApplication,
    QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem,
    QDialog, QTextEdit, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint, QSize, QUrl
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen,
    QLinearGradient, QBrush, QPalette, QFontDatabase, QDesktopServices,
)
from utils.video_processor import VideoProcessor


# ======================== 线程 ========================

class InferenceThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, processor: VideoProcessor):
        super().__init__()
        self.processor = processor
        self._frame = None
        self._has_frame = False
        self._running = True

    def set_frame(self, frame: np.ndarray):
        self._frame = frame.copy()
        self._has_frame = True

    def run(self):
        while self._running:
            if self._has_frame and self._frame is not None:
                result = self.processor.process_frame(self._frame)
                self.frame_ready.emit(result)
                self._has_frame = False
            else:
                self.msleep(5)

    def stop(self):
        self._running = False
        self.wait(3000)


class BriefWorker(QThread):
    """后台调用 DeepSeek，避免卡住界面"""
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, processor: VideoProcessor):
        super().__init__()
        self.processor = processor

    def run(self):
        try:
            from utils.deepseek_client import (
                summarize_events_text,
                build_payload_from_recent_events,
                is_configured,
            )
            if not is_configured():
                self.finished_err.emit("请先在 config/settings.json 中填写 deepseek.api_key（可复制 config/settings.example.json 为 settings.json）")
                return
            rows = self.processor.logger.get_recent_events(50)
            payload = build_payload_from_recent_events(rows)
            text = summarize_events_text(payload)
            self.finished_ok.emit(text)
        except Exception as e:
            self.finished_err.emit(str(e))


# ======================== ROI 视频标签 ========================

class ROIVideoLabel(QLabel):
    roi_updated = pyqtSignal(list)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._drawing = False
        self._roi_points = []
        self._frame_size = (0, 0)
        self._current_pixmap_rect = None

    def set_drawing_mode(self, enabled: bool):
        self._drawing = enabled
        if enabled:
            self._roi_points = []
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _map_to_frame(self, pos):
        if self._current_pixmap_rect is None or self._frame_size == (0, 0):
            return None
        r = self._current_pixmap_rect
        fw, fh = self._frame_size
        fx = max(0, min(fw, (pos.x() - r.x()) / r.width() * fw))
        fy = max(0, min(fh, (pos.y() - r.y()) / r.height() * fh))
        return (int(fx), int(fy))

    def mousePressEvent(self, event):
        if not self._drawing or event.button() != Qt.LeftButton:
            return
        pt = self._map_to_frame(event.pos())
        if pt:
            self._roi_points.append(pt)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if not self._drawing:
            return
        if len(self._roi_points) >= 3:
            self.roi_updated.emit(self._roi_points.copy())
        self._drawing = False
        self.setCursor(Qt.ArrowCursor)

    def set_frame_size(self, w, h):
        self._frame_size = (w, h)

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        if pixmap:
            from PyQt5.QtCore import QRect
            lw, lh = self.width(), self.height()
            pw, ph = pixmap.width(), pixmap.height()
            self._current_pixmap_rect = QRect((lw - pw) // 2, (lh - ph) // 2, pw, ph)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._roi_points or self._current_pixmap_rect is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 204, 0, 200), 2, Qt.DashLine))
        r = self._current_pixmap_rect
        fw, fh = self._frame_size if self._frame_size != (0, 0) else (1, 1)
        mapped = [QPoint(int(r.x() + fx / fw * r.width()), int(r.y() + fy / fh * r.height()))
                  for fx, fy in self._roi_points]
        for i, pt in enumerate(mapped):
            painter.setBrush(QColor(255, 204, 0, 120))
            painter.drawEllipse(pt, 5, 5)
            if i > 0:
                painter.drawLine(mapped[i - 1], pt)
        if len(mapped) >= 3 and not self._drawing:
            painter.drawLine(mapped[-1], mapped[0])
        painter.end()


# ======================== 卡片容器 ========================

class Card(QFrame):
    """毛玻璃风格圆角卡片"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

    @staticmethod
    def create(parent=None):
        card = Card(parent)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))
        card.setGraphicsEffect(shadow)
        return card


# ======================== 主窗口 ========================

class MainWindow(QMainWindow):
    def __init__(self, processor: VideoProcessor):
        super().__init__()
        self.processor = processor
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._read_frame)
        self.is_camera = False
        self.is_playing = False
        self.inference_thread = None
        self.tray_icon = None
        self._last_tray_alert = 0

        self._apply_styles()
        self._init_ui()
        self._init_tray_icon()

    # ====================== UI 构建 ======================

    def _init_ui(self):
        self.setWindowTitle("Drowning Monitor")
        self.setMinimumSize(1340, 820)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(0)

        # ---- 顶栏 ----
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 16)

        brand = QLabel("Drowning Monitor")
        brand.setObjectName("brand")
        top_bar.addWidget(brand)

        top_bar.addStretch()

        self.status_pill = QLabel("  就绪")
        self.status_pill.setObjectName("statusPill")
        top_bar.addWidget(self.status_pill)

        top_bar.addSpacerItem(QSpacerItem(16, 0))

        self.label_time = QLabel()
        self.label_time.setObjectName("timeLabel")
        top_bar.addWidget(self.label_time)

        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)
        self._update_time()

        root.addLayout(top_bar)

        # ---- 主体 ----
        body = QHBoxLayout()
        body.setSpacing(20)

        # == 左侧：视频区 ==
        left = QVBoxLayout()
        left.setSpacing(12)

        video_card = Card.create()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(2, 2, 2, 2)

        self.video_label = ROIVideoLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 520)
        self.video_label.setObjectName("videoLabel")
        self.video_label.setText("选择视频源以开始监测")
        self.video_label.roi_updated.connect(self._on_roi_set)
        video_layout.addWidget(self.video_label)

        left.addWidget(video_card)

        # 进度条 + 底部信息
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self._seek_video)
        self.progress_slider.setFixedHeight(20)
        bottom_bar.addWidget(self.progress_slider, stretch=1)

        self.label_fps = QLabel("— FPS")
        self.label_fps.setObjectName("metricLabel")
        bottom_bar.addWidget(self.label_fps)

        self.label_persons = QLabel("— 人")
        self.label_persons.setObjectName("metricLabel")
        bottom_bar.addWidget(self.label_persons)

        left.addLayout(bottom_bar)
        body.addLayout(left, stretch=7)

        # == 右侧：控制面板 ==
        right = QVBoxLayout()
        right.setSpacing(16)

        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("sidebarTabs")
        self.tab_widget.setFixedWidth(310)

        self._build_control_tab()
        self._build_settings_tab()
        self._build_log_tab()

        right.addWidget(self.tab_widget)
        body.addLayout(right, stretch=0)

        root.addLayout(body)

    # ---- 控制 ----
    def _build_control_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(16, 20, 16, 16)
        lay.setSpacing(0)

        lay.addWidget(self._section("视频源"))
        lay.addSpacing(8)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2"])
        lay.addWidget(self.camera_combo)
        lay.addSpacing(10)

        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.setObjectName("accentBtn")
        self.btn_camera.clicked.connect(self._toggle_camera)
        lay.addWidget(self.btn_camera)
        lay.addSpacing(6)

        self.btn_upload = QPushButton("导入视频")
        self.btn_upload.setObjectName("ghostBtn")
        self.btn_upload.clicked.connect(self._open_video_file)
        lay.addWidget(self.btn_upload)
        lay.addSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_play_pause = QPushButton("暂停")
        self.btn_play_pause.setObjectName("ghostBtn")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        row.addWidget(self.btn_play_pause)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_all)
        row.addWidget(self.btn_stop)
        lay.addLayout(row)

        lay.addSpacing(16)
        lay.addWidget(self._divider())
        lay.addSpacing(16)

        lay.addWidget(self._section("ROI 监控区域"))
        lay.addSpacing(8)

        roi_row = QHBoxLayout()
        roi_row.setSpacing(8)
        self.btn_set_roi = QPushButton("绘制")
        self.btn_set_roi.setObjectName("ghostBtn")
        self.btn_set_roi.clicked.connect(self._start_roi_drawing)
        roi_row.addWidget(self.btn_set_roi)
        self.btn_clear_roi = QPushButton("清除")
        self.btn_clear_roi.setObjectName("ghostBtn")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        roi_row.addWidget(self.btn_clear_roi)
        lay.addLayout(roi_row)

        self.label_roi_status = QLabel("未设置 · 检测全画面")
        self.label_roi_status.setObjectName("hint")
        lay.addWidget(self.label_roi_status)

        lay.addSpacing(20)
        lay.addWidget(self._divider())
        lay.addSpacing(20)

        # 状态卡片
        lay.addWidget(self._section("实时状态"))
        lay.addSpacing(10)

        self.label_drowning = QLabel("0")
        self.label_drowning.setObjectName("bigMetric")
        self.label_drowning_hint = QLabel("溺水确认人数")
        self.label_drowning_hint.setObjectName("hint")

        metric_card = Card.create()
        metric_card.setObjectName("metricCard")
        mc_lay = QVBoxLayout(metric_card)
        mc_lay.setContentsMargins(16, 14, 16, 14)
        mc_lay.setSpacing(2)
        mc_lay.addWidget(self.label_drowning)
        mc_lay.addWidget(self.label_drowning_hint)
        lay.addWidget(metric_card)

        lay.addStretch()

        # 图例
        lay.addSpacing(8)
        legend_row = QHBoxLayout()
        legend_row.setSpacing(16)
        for text, color in [("溺水", "#FF453A"), ("游泳", "#30D158"), ("离水", "#FF9F0A")]:
            dot = QLabel(f'<span style="color:{color}; font-size:18px;">●</span>  {text}')
            dot.setObjectName("legend")
            legend_row.addWidget(dot)
        legend_row.addStretch()
        lay.addLayout(legend_row)

        self.tab_widget.addTab(tab, "控制")

    # ---- 设置 ----
    def _build_settings_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(16, 20, 16, 16)
        lay.setSpacing(0)

        lay.addWidget(self._section("检测置信度"))
        lay.addSpacing(8)
        conf_row = QHBoxLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.slider_conf)
        self.label_conf_val = QLabel("0.50")
        self.label_conf_val.setObjectName("sliderVal")
        self.label_conf_val.setMinimumWidth(36)
        conf_row.addWidget(self.label_conf_val)
        lay.addLayout(conf_row)

        lay.addSpacing(20)
        lay.addWidget(self._section("溺水确认帧数"))
        lay.addSpacing(8)
        self.spin_confirm = QSpinBox()
        self.spin_confirm.setRange(1, 60)
        self.spin_confirm.setValue(8)
        self.spin_confirm.setSuffix("  帧")
        self.spin_confirm.valueChanged.connect(self._on_confirm_frames_changed)
        lay.addWidget(self.spin_confirm)
        hint = QLabel("连续多帧判定溺水才触发报警，越大越不易误报")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addSpacing(20)
        lay.addWidget(self._section("报警冷却"))
        lay.addSpacing(8)
        self.spin_cooldown = QSpinBox()
        self.spin_cooldown.setRange(1, 60)
        self.spin_cooldown.setValue(5)
        self.spin_cooldown.setSuffix("  秒")
        self.spin_cooldown.valueChanged.connect(self._on_cooldown_changed)
        lay.addWidget(self.spin_cooldown)

        lay.addSpacing(24)
        lay.addWidget(self._divider())
        lay.addSpacing(20)

        lay.addWidget(self._section("功能开关"))
        lay.addSpacing(10)

        self.chk_alarm = QCheckBox("声音报警")
        self.chk_alarm.setChecked(True)
        self.chk_alarm.toggled.connect(lambda v: setattr(self.processor.alarm, 'enabled', v))
        lay.addWidget(self.chk_alarm)
        lay.addSpacing(6)

        self.chk_auto_record = QCheckBox("自动录像")
        self.chk_auto_record.setChecked(True)
        lay.addWidget(self.chk_auto_record)
        lay.addSpacing(6)

        self.chk_tray_notify = QCheckBox("系统通知")
        self.chk_tray_notify.setChecked(True)
        lay.addWidget(self.chk_tray_notify)

        lay.addSpacing(20)
        lay.addWidget(self._section("最大录像时长"))
        lay.addSpacing(8)
        self.spin_clip_dur = QSpinBox()
        self.spin_clip_dur.setRange(5, 120)
        self.spin_clip_dur.setValue(30)
        self.spin_clip_dur.setSuffix("  秒")
        self.spin_clip_dur.valueChanged.connect(
            lambda v: setattr(self.processor.logger, 'max_clip_duration', v)
        )
        lay.addWidget(self.spin_clip_dur)

        lay.addSpacing(24)
        lay.addWidget(self._divider())
        lay.addSpacing(16)

        lay.addWidget(self._section("AI 简报 · DeepSeek"))
        lay.addSpacing(6)
        ds_hint = QLabel("在 config/settings.json 填写 deepseek.api_key（可复制 settings.example.json）")
        ds_hint.setObjectName("hint")
        ds_hint.setWordWrap(True)
        lay.addWidget(ds_hint)
        lay.addSpacing(8)
        self.btn_ai_brief = QPushButton("生成最近事件简报")
        self.btn_ai_brief.setObjectName("ghostBtn")
        self.btn_ai_brief.clicked.connect(self._generate_ai_brief)
        lay.addWidget(self.btn_ai_brief)

        lay.addWidget(self._section("Web 看板（可选）"))
        web_hint = QLabel(
            "与桌面监测独立进程：不启动看板不影响本地监测；看板仅读取 logs 目录。"
            " 启动方式：项目根目录运行 python -m web 或双击 run_web_dashboard.bat"
        )
        web_hint.setObjectName("hint")
        web_hint.setWordWrap(True)
        lay.addWidget(web_hint)
        lay.addSpacing(8)
        web_row = QHBoxLayout()
        web_row.setSpacing(8)
        self.btn_open_dashboard = QPushButton("在浏览器打开看板")
        self.btn_open_dashboard.setObjectName("ghostBtn")
        self.btn_open_dashboard.setToolTip("打开 http://127.0.0.1:8080（需已先启动看板服务）")
        self.btn_open_dashboard.clicked.connect(self._open_dashboard_browser)
        web_row.addWidget(self.btn_open_dashboard)
        self.btn_open_readme = QPushButton("打开说明 web/README.md")
        self.btn_open_readme.setObjectName("ghostBtn")
        self.btn_open_readme.clicked.connect(self._open_web_readme)
        web_row.addWidget(self.btn_open_readme)
        lay.addLayout(web_row)

        lay.addStretch()

        self.btn_open_logs = QPushButton("打开日志目录")
        self.btn_open_logs.setObjectName("ghostBtn")
        self.btn_open_logs.clicked.connect(self._open_logs_folder)
        lay.addWidget(self.btn_open_logs)

        self.tab_widget.addTab(tab, "设置")

    # ---- 日志 ----
    def _build_log_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(16, 20, 16, 16)
        lay.setSpacing(0)

        lay.addWidget(self._section("报警记录"))
        lay.addSpacing(10)

        self.event_table = QTableWidget()
        self.event_table.setColumnCount(4)
        self.event_table.setHorizontalHeaderLabels(["时间", "类型", "人数", "置信度"])
        hdr = self.event_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setMinimumSectionSize(72)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setShowGrid(False)
        self.event_table.setObjectName("logTable")
        lay.addWidget(self.event_table)

        lay.addSpacing(10)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_refresh_log = QPushButton("刷新")
        self.btn_refresh_log.setObjectName("ghostBtn")
        self.btn_refresh_log.clicked.connect(self._refresh_event_log)
        row.addWidget(self.btn_refresh_log)
        self.btn_screenshots = QPushButton("截图")
        self.btn_screenshots.setObjectName("ghostBtn")
        self.btn_screenshots.clicked.connect(self._open_screenshots_folder)
        row.addWidget(self.btn_screenshots)
        lay.addLayout(row)

        lay.addSpacing(6)
        self.label_event_count = QLabel("共 0 条")
        self.label_event_count.setObjectName("hint")
        lay.addWidget(self.label_event_count)

        self.tab_widget.addTab(tab, "日志")

        self.log_refresh_timer = QTimer()
        self.log_refresh_timer.timeout.connect(self._refresh_event_log)
        self.log_refresh_timer.start(5000)

    # ====================== 辅助组件 ======================

    @staticmethod
    def _section(text):
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    @staticmethod
    def _divider():
        d = QFrame()
        d.setFixedHeight(1)
        d.setObjectName("divider")
        return d

    # ====================== 系统托盘 ======================

    def _init_tray_icon(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip("Drowning Monitor")

        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(0, 122, 255))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, 32, 32, 8, 8)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Arial", 17, QFont.Bold))
        p.drawText(pm.rect(), Qt.AlignCenter, "D")
        p.end()

        icon = QIcon(pm)
        self.tray_icon.setIcon(icon)
        self.setWindowIcon(icon)

        menu = QMenu()
        menu.addAction("显示窗口", self.showNormal)
        menu.addSeparator()
        menu.addAction("退出", QApplication.quit)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(
            lambda r: (self.showNormal(), self.activateWindow()) if r == QSystemTrayIcon.DoubleClick else None
        )
        self.tray_icon.show()

    def _show_tray_notification(self, title, message):
        if self.tray_icon and self.chk_tray_notify.isChecked():
            import time
            now = time.time()
            if now - self._last_tray_alert < 10:
                return
            self._last_tray_alert = now
            self.tray_icon.showMessage(title, message, QSystemTrayIcon.Critical, 5000)

    # ====================== 参数回调 ======================

    def _on_conf_changed(self, v):
        conf = v / 100.0
        self.label_conf_val.setText(f"{conf:.2f}")
        self.processor.set_det_confidence(conf)

    def _on_confirm_frames_changed(self, v):
        self.processor.set_confirm_frames(v)

    def _on_cooldown_changed(self, v):
        self.processor.set_alarm_cooldown(v)

    # ====================== ROI ======================

    def _start_roi_drawing(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.information(self, "提示", "请先打开视频源")
            return
        self.video_label.set_drawing_mode(True)
        self.label_roi_status.setText("绘制中 · 单击添加顶点 · 双击完成")
        self.label_roi_status.setStyleSheet("color: #FF9F0A;")

    def _on_roi_set(self, points):
        if len(points) >= 3:
            self.processor.set_roi(points)
            self.label_roi_status.setText(f"已设置 · {len(points)} 个顶点")
            self.label_roi_status.setStyleSheet("color: #30D158;")

    def _clear_roi(self):
        self.processor.set_roi(None)
        self.video_label._roi_points = []
        self.video_label.set_drawing_mode(False)
        self.video_label.update()
        self.label_roi_status.setText("未设置 · 检测全画面")
        self.label_roi_status.setStyleSheet("")

    # ====================== 摄像头 ======================

    def _toggle_camera(self):
        if self.is_camera:
            self._stop_all()
            return
        self._stop_all()
        cam_idx = self.camera_combo.currentIndex()
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开摄像头 {cam_idx}")
            return
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_label.set_frame_size(w, h)
        self.is_camera = True
        self.is_playing = True
        self._update_ui_playing(True)
        self._start_inference_thread()
        self.timer.start(30)

    # ====================== 视频文件 ======================

    def _open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;全部 (*)"
        )
        if not path:
            return
        self._stop_all()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开:\n{path}")
            return
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_label.set_frame_size(w, h)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_slider.setRange(0, max(total, 1))
        self.progress_slider.setEnabled(True)
        self.is_camera = False
        self.is_playing = True
        self._update_ui_playing(False)
        self._start_inference_thread()
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.timer.start(int(1000 / fps))

    def _toggle_play_pause(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.btn_play_pause.setText("继续")
            self._set_pill("已暂停", "pause")
        else:
            self.is_playing = True
            fps = (self.cap.get(cv2.CAP_PROP_FPS) or 30) if self.cap else 30
            self.timer.start(int(1000 / fps))
            self.btn_play_pause.setText("暂停")
            self._set_pill("监测中", "active")

    def _seek_video(self, pos):
        if self.cap and not self.is_camera:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    # ====================== 帧处理 ======================

    def _read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            if not self.is_camera:
                self.timer.stop()
                self._set_pill("已结束", "pause")
            return
        if not self.is_camera:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_slider.blockSignals(True)
            self.progress_slider.setValue(pos)
            self.progress_slider.blockSignals(False)
        if self.inference_thread:
            self.inference_thread.set_frame(frame)

    def _on_frame_processed(self, annotated: np.ndarray):
        self._display_frame(annotated)
        self.label_fps.setText(f"{self.processor.fps:.0f} FPS")
        self.label_persons.setText(f"{self.processor.person_count} 人")

        dc = self.processor.drowning_confirmed_count
        self.label_drowning.setText(str(dc))
        if dc > 0:
            self.label_drowning.setStyleSheet("color: #FF453A;")
            self.label_drowning_hint.setStyleSheet("color: #FF453A;")
            self._set_pill(f"溺水报警 · {dc}人", "alert")
            self._show_tray_notification("溺水报警", f"检测到 {dc} 人疑似溺水！")
        else:
            self.label_drowning.setStyleSheet("")
            self.label_drowning_hint.setStyleSheet("")
            if self.is_playing:
                self._set_pill("监测中", "active")

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(qi).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    # ====================== 推理线程 ======================

    def _start_inference_thread(self):
        self._stop_inference_thread()
        self.processor.reset()
        self.inference_thread = InferenceThread(self.processor)
        self.inference_thread.frame_ready.connect(self._on_frame_processed)
        self.inference_thread.start()

    def _stop_inference_thread(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None

    # ====================== 停止 ======================

    def _stop_all(self):
        self.timer.stop()
        self._stop_inference_thread()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.processor.release()
        self.processor.reset()
        self.is_camera = False
        self.is_playing = False

        self.btn_camera.setText("打开摄像头")
        self.btn_play_pause.setText("暂停")
        self.btn_play_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.progress_slider.setEnabled(False)
        self.progress_slider.setValue(0)
        self.video_label.clear()
        self.video_label.setText("选择视频源以开始监测")
        self.label_fps.setText("— FPS")
        self.label_persons.setText("— 人")
        self.label_drowning.setText("0")
        self.label_drowning.setStyleSheet("")
        self.label_drowning_hint.setStyleSheet("")
        self._set_pill("就绪", "idle")

    # ====================== UI 状态 ======================

    def _update_ui_playing(self, is_camera):
        self.btn_camera.setText("关闭摄像头" if is_camera else "打开摄像头")
        self.btn_stop.setEnabled(True)
        self.btn_play_pause.setEnabled(not is_camera)
        self.btn_play_pause.setText("暂停")
        self._set_pill("摄像头监测中" if is_camera else "视频监测中", "active")

    def _set_pill(self, text, state="idle"):
        colors = {
            "idle":   ("#8E8E93", "#1C1C1E"),
            "active": ("#30D158", "#0A2E14"),
            "alert":  ("#FF453A", "#3A1210"),
            "pause":  ("#FF9F0A", "#2E1E05"),
        }
        fg, bg = colors.get(state, colors["idle"])
        self.status_pill.setText(f"  {text}")
        self.status_pill.setStyleSheet(
            f"background-color: {bg}; color: {fg}; "
            f"border: 1px solid {fg}; border-radius: 10px; "
            f"padding: 3px 12px; font-size: 11px; font-weight: 600;"
        )

    def _update_time(self):
        self.label_time.setText(datetime.now().strftime("%H:%M:%S"))

    # ====================== 日志 ======================

    def _refresh_event_log(self):
        events = self.processor.logger.get_recent_events(50)
        self.event_table.setRowCount(len(events))
        for i, row in enumerate(events):
            for j, col_idx in enumerate([0, 1, 2, 4]):
                if col_idx < len(row):
                    item = QTableWidgetItem(row[col_idx])
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 1 and "溺水" in row[col_idx]:
                        item.setForeground(QColor("#FF453A"))
                    self.event_table.setItem(i, j, item)
        self.label_event_count.setText(f"共 {len(events)} 条")

    def _open_logs_folder(self):
        d = str(self.processor.logger.output_dir)
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def _open_dashboard_browser(self):
        QDesktopServices.openUrl(QUrl("http://127.0.0.1:8080"))

    def _open_web_readme(self):
        from utils.paths import PROJECT_ROOT
        p = PROJECT_ROOT / "web" / "README.md"
        if p.is_file():
            os.startfile(str(p))
        else:
            QMessageBox.information(self, "提示", "未找到 web/README.md")

    def _generate_ai_brief(self):
        self.btn_ai_brief.setEnabled(False)
        self._brief_worker = BriefWorker(self.processor)
        self._brief_worker.finished_ok.connect(self._on_brief_ok)
        self._brief_worker.finished_err.connect(self._on_brief_err)
        self._brief_worker.finished.connect(lambda: self.btn_ai_brief.setEnabled(True))
        self._brief_worker.start()

    def _on_brief_ok(self, text: str):
        dlg = QDialog(self)
        dlg.setWindowTitle("AI 简报")
        dlg.setMinimumSize(520, 420)
        # 设置对话框样式，确保文字可见
        dlg.setStyleSheet("""
            QDialog {
                background-color: #1C1C1E;
            }
            QTextEdit {
                background-color: #2C2C2E;
                color: #F5F5F7;
                border: 1px solid #3A3A3C;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.6;
            }
            QPushButton {
                background-color: #0A84FF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0070E0;
            }
        """)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        lay.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec_()

    def _on_brief_err(self, msg: str):
        QMessageBox.warning(self, "DeepSeek", msg)

    def _open_screenshots_folder(self):
        d = os.path.abspath("logs/screenshots")
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    # ====================== 样式 ======================

    def _apply_styles(self):
        self.setStyleSheet("""
            /* ===== 全局 ===== */
            QMainWindow {
                background-color: #000000;
            }
            QWidget {
                color: #F5F5F7;
                font-family: "Microsoft YaHei", "SF Pro Display", "Segoe UI", sans-serif;
            }

            /* ===== 顶栏 ===== */
            #brand {
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.5px;
                color: #F5F5F7;
            }
            #timeLabel {
                font-size: 13px;
                color: #8E8E93;
                font-family: "Consolas", "SF Mono", monospace;
            }
            #statusPill {
                font-size: 11px;
                font-weight: 600;
            }

            /* ===== 卡片 ===== */
            #card {
                background-color: #1C1C1E;
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.06);
            }
            #metricCard {
                background-color: #1C1C1E;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.06);
            }

            /* ===== 视频区 ===== */
            #videoLabel {
                background-color: #000000;
                border-radius: 12px;
                color: #3A3A3C;
                font-size: 14px;
                font-weight: 500;
            }

            /* ===== 选项卡 ===== */
            #sidebarTabs {
                background-color: transparent;
            }
            QTabWidget::pane {
                background-color: #1C1C1E;
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 14px;
                border-top-left-radius: 0px;
            }
            QTabBar::tab {
                background-color: transparent;
                color: #8E8E93;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: 600;
                border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #F5F5F7;
                border-bottom: 2px solid #0A84FF;
            }
            QTabBar::tab:hover {
                color: #F5F5F7;
            }

            /* ===== 文字层级 ===== */
            #sectionTitle {
                font-size: 11px;
                font-weight: 700;
                color: #8E8E93;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            #hint {
                font-size: 11px;
                color: #636366;
                margin-top: 4px;
            }
            #bigMetric {
                font-size: 36px;
                font-weight: 700;
                color: #F5F5F7;
                font-family: "Consolas", "SF Mono", monospace;
            }
            #metricLabel {
                font-size: 12px;
                color: #8E8E93;
                font-family: "Consolas", "SF Mono", monospace;
                padding: 0 4px;
            }
            #legend {
                font-size: 12px;
                color: #8E8E93;
            }
            #sliderVal {
                font-size: 12px;
                color: #F5F5F7;
                font-family: "Consolas", monospace;
                font-weight: 600;
            }

            /* ===== 分割线 ===== */
            #divider {
                background-color: rgba(255,255,255,0.06);
            }

            /* ===== 按钮 ===== */
            QPushButton {
                padding: 10px 16px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                border: none;
            }
            #accentBtn {
                background-color: #0A84FF;
                color: #FFFFFF;
            }
            #accentBtn:hover {
                background-color: #409CFF;
            }
            #accentBtn:pressed {
                background-color: #0071E3;
            }
            #ghostBtn {
                background-color: rgba(255,255,255,0.06);
                color: #F5F5F7;
                border: 1px solid rgba(255,255,255,0.1);
            }
            #ghostBtn:hover {
                background-color: rgba(255,255,255,0.1);
            }
            #ghostBtn:pressed {
                background-color: rgba(255,255,255,0.04);
            }
            #ghostBtn:disabled {
                color: #3A3A3C;
                border-color: rgba(255,255,255,0.04);
            }
            #dangerBtn {
                background-color: rgba(255,69,58,0.15);
                color: #FF453A;
                border: 1px solid rgba(255,69,58,0.3);
            }
            #dangerBtn:hover {
                background-color: rgba(255,69,58,0.25);
            }
            #dangerBtn:disabled {
                color: #3A3A3C;
                background-color: transparent;
                border-color: rgba(255,255,255,0.04);
            }

            /* ===== 输入控件 ===== */
            QComboBox {
                padding: 8px 12px;
                border-radius: 10px;
                background-color: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.1);
                color: #F5F5F7;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow { image: none; }
            QComboBox QAbstractItemView {
                background-color: #2C2C2E;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                color: #F5F5F7;
                selection-background-color: #0A84FF;
            }
            QSpinBox {
                padding: 8px 12px;
                border-radius: 10px;
                background-color: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.1);
                color: #F5F5F7;
                font-size: 12px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                border: none;
                background: transparent;
            }

            /* ===== 复选框 ===== */
            QCheckBox {
                spacing: 10px;
                font-size: 13px;
                color: #F5F5F7;
            }
            QCheckBox::indicator {
                width: 18px; height: 18px;
                border-radius: 5px;
                border: 2px solid #48484A;
                background-color: transparent;
            }
            QCheckBox::indicator:checked {
                background-color: #0A84FF;
                border-color: #0A84FF;
            }

            /* ===== 滑块 ===== */
            QSlider::groove:horizontal {
                height: 4px;
                background: #38383A;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #F5F5F7;
                width: 16px; height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #0A84FF;
                border-radius: 2px;
            }
            QSlider:disabled::groove:horizontal { background: #1C1C1E; }
            QSlider:disabled::handle:horizontal { background: #3A3A3C; }

            /* ===== 表格（暗色统一底，避免白底灰字看不清） ===== */
            QTableWidget#logTable {
                background-color: #1C1C1E;
                alternate-background-color: #252528;
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 10px;
                font-size: 12px;
                color: #E8E8ED;
                gridline-color: rgba(255,255,255,0.06);
            }
            #logTable::item {
                padding: 8px 6px;
                border-bottom: 1px solid rgba(255,255,255,0.06);
                background-color: #1C1C1E;
                color: #E8E8ED;
            }
            #logTable::item:alternate {
                background-color: #252528;
                color: #E8E8ED;
            }
            #logTable::item:selected {
                background-color: #0A84FF;
                color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #2C2C2E;
                color: #AEAEB2;
                padding: 8px 6px;
                border: none;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                font-size: 11px;
                font-weight: 600;
            }

            /* ===== 滚动条 ===== */
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,0.15);
                border-radius: 3px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }

            /* ===== 状态栏 ===== */
            QStatusBar { background: transparent; color: #3A3A3C; font-size: 0px; }
        """)

    # ====================== 关闭 ======================

    def closeEvent(self, event):
        self._stop_all()
        if self.tray_icon:
            self.tray_icon.hide()
        event.accept()
