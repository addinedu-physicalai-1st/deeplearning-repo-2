import sys
import cv2
import time
import logging
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFrame, QGridLayout, QStackedWidget, QMessageBox, QApplication,
                             QProgressBar)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont
import pyqtgraph as pg
from datetime import datetime

from client.core.camera import Camera
from client.core.network import NetworkClient
from client.core.posture import PostureMonitor

# Logging setup
logger = logging.getLogger(__name__)

# --- 공통 스타일 정의 ---
STYLE_SHEET = """
    QMainWindow { background-color: #121212; }
    QWidget { color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
    QFrame#Card { background-color: #1E1E1E; border-radius: 12px; border: 1px solid #333; }
    QLabel#Title { font-size: 32px; font-weight: bold; color: #BB86FC; margin-bottom: 20px; }
    QLabel#Header { font-size: 18px; color: #9E9E9E; }
    QLabel#StatValue { font-size: 36px; font-weight: bold; color: #03DAC6; }
    QLabel#StatLabel { font-size: 14px; color: #9E9E9E; text-transform: uppercase; }
    QPushButton#PrimaryBtn { 
        background-color: #BB86FC; color: #000; font-weight: bold; 
        border-radius: 8px; padding: 15px 30px; font-size: 18px;
    }
    QPushButton#PrimaryBtn:hover { background-color: #D7B7FD; }
    QPushButton#SecondaryBtn { 
        background-color: transparent; color: #BB86FC; font-weight: bold; 
        border: 2px solid #BB86FC; border-radius: 8px; padding: 10px 20px;
    }
    QPushButton#StopBtn { 
        background-color: #CF6679; color: #000; font-weight: bold; 
        border-radius: 8px; padding: 12px; font-size: 16px;
    }
"""

class InferenceThread(QThread):
    result_ready = pyqtSignal(object)
    def __init__(self, network_client, image_base64, session_id=None):
        super().__init__()
        self.network_client = network_client
        self.image_base64 = image_base64
        self.session_id = session_id
    def run(self):
        result = self.network_client.send_inference_request(self.image_base64, self.session_id)
        self.result_ready.emit(result)

class MainPage(QWidget):
    """프로그램 시작 메인 화면"""
    start_requested = pyqtSignal()
    history_requested = pyqtSignal()
    calibration_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Welcome Card
        card = QFrame()
        card.setObjectName("Card")
        card.setFixedSize(500, 450)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(20)

        title = QLabel("READY TO FOCUS?")
        title.setObjectName("Title")
        card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        desc = QLabel("Start your session to monitor productivity\nand get AI-powered feedback.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 16px; color: #9E9E9E;")
        card_layout.addWidget(desc)

        # Connection Status
        self.status_label = QLabel("Checking server connection...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #FFB74D; font-weight: bold;")  # Orange for waiting
        card_layout.addWidget(self.status_label)

        self.calibration_btn = QPushButton("거리 캘리브레이션")
        self.calibration_btn.setObjectName("SecondaryBtn")
        self.calibration_btn.clicked.connect(self.calibration_requested.emit)
        card_layout.addWidget(self.calibration_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.start_btn = QPushButton("START NEW SESSION")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setEnabled(False) # Disabled by default
        self.start_btn.setStyleSheet("QPushButton:disabled { background-color: #333; color: #666; }")
        self.start_btn.clicked.connect(self.start_requested.emit)
        card_layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.history_btn = QPushButton("VIEW PAST HISTORY")
        self.history_btn.setObjectName("SecondaryBtn")
        self.history_btn.clicked.connect(self.history_requested.emit)
        card_layout.addWidget(self.history_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(card)

    def set_connection_status(self, connected: bool):
        if connected:
            self.status_label.setText("Server Connected")
            self.status_label.setStyleSheet("color: #03DAC6; font-weight: bold;")
            self.start_btn.setEnabled(True)
        else:
            self.status_label.setText("Waiting for Orchestrator...")
            self.status_label.setStyleSheet("color: #CF6679; font-weight: bold;")
            self.start_btn.setEnabled(False)


class DistanceCalibrationPage(QWidget):
    """노트북-사람 거리 캘리브레이션 화면 (posture.py 로직 + ai_body set_baseline)."""
    done_requested = pyqtSignal()

    def __init__(self, camera: Camera, network_client: NetworkClient):
        super().__init__()
        self.camera = camera
        self.network_client = network_client
        self.monitor = PostureMonitor()
        self.current_distance_cm = None
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # Video + labels left
        left = QVBoxLayout()
        self.video_label = QLabel("Camera")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setFixedSize(640, 480)
        left.addWidget(self.video_label)

        self.status_label = QLabel("상태: 대기중")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFB74D;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.status_label)
        layout.addLayout(left)

        # Control panel right
        panel = QFrame()
        panel.setObjectName("Card")
        panel.setStyleSheet("QFrame#Card { background-color: #1E1E1E; border-radius: 12px; border: 1px solid #333; }")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(12)

        title = QLabel("거리 캘리브레이션")
        title.setObjectName("Title")
        title.setStyleSheet("font-size: 22px; color: #BB86FC;")
        panel_layout.addWidget(title)

        self.shoulder_label = QLabel("어깨 각도: --°")
        self.shoulder_label.setStyleSheet("font-size: 16px; color: #9E9E9E;")
        panel_layout.addWidget(self.shoulder_label)

        self.distance_label = QLabel("거리: -- cm")
        self.distance_label.setStyleSheet("font-size: 16px; color: #03DAC6;")
        panel_layout.addWidget(self.distance_label)

        self.posture_label = QLabel("거북목: -- %")
        self.posture_label.setStyleSheet("font-size: 16px; color: #64B5F6;")
        panel_layout.addWidget(self.posture_label)

        self.posture_progress = QProgressBar()
        self.posture_progress.setMinimum(0)
        self.posture_progress.setMaximum(100)
        self.posture_progress.setValue(0)
        self.posture_progress.setStyleSheet("""
            QProgressBar { border: 2px solid #64B5F6; border-radius: 5px; background-color: #333; }
            QProgressBar::chunk { background-color: #64B5F6; }
        """)
        panel_layout.addWidget(self.posture_progress)

        panel_layout.addStretch(1)

        self.set_baseline_btn = QPushButton("정자세 설정")
        self.set_baseline_btn.setObjectName("PrimaryBtn")
        self.set_baseline_btn.clicked.connect(self._on_set_baseline)
        panel_layout.addWidget(self.set_baseline_btn)

        self.done_btn = QPushButton("완료")
        self.done_btn.setObjectName("SecondaryBtn")
        self.done_btn.clicked.connect(self.done_requested.emit)
        panel_layout.addWidget(self.done_btn)

        layout.addWidget(panel)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start(33)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def _update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return
        self.current_frame = frame
        image_rgb, shoulder_angle, distance_cm, posture_percentage, distance_offset_cm = self.monitor.process_frame(frame)
        self.current_distance_cm = distance_cm

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if shoulder_angle is not None:
            self.shoulder_label.setText(f"어깨 각도: {shoulder_angle:.1f}°")
        else:
            self.shoulder_label.setText("어깨 각도: --°")

        if distance_cm is not None:
            self.distance_label.setText(f"거리: {distance_cm:.1f} cm")
        else:
            self.distance_label.setText("거리: -- cm")

        if posture_percentage is not None:
            self.posture_progress.setValue(int(posture_percentage))
            self.posture_label.setText(f"거북목: {int(posture_percentage)} %")
        else:
            self.posture_progress.setValue(0)
            self.posture_label.setText("거북목: -- %")

    def _on_set_baseline(self):
        if self.current_distance_cm is None or self.current_frame is None:
            self.status_label.setText("거리를 감지할 수 없습니다.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF9800;")
            return
        self.monitor.set_baseline_from_distance(self.current_distance_cm)
        image_base64 = self.camera.frame_to_base64(self.current_frame)
        ok = self.network_client.set_baseline(image_base64)
        if ok:
            self.status_label.setText("정자세가 설정되었습니다.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #03DAC6;")
        else:
            self.status_label.setText("ai_body 서버 연결 실패. 서버를 확인하세요.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #CF6679;")


class MonitoringPage(QWidget):
    """실시간 모니터링 화면"""
    stop_requested = pyqtSignal()
    set_baseline_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Top Bar
        top_bar = QHBoxLayout()
        header = QLabel("LIVE MONITORING")
        header.setObjectName("Header")
        top_bar.addWidget(header)
        top_bar.addStretch()

        self.set_baseline_btn = QPushButton("정자세 다시 설정")
        self.set_baseline_btn.setObjectName("SecondaryBtn")
        self.set_baseline_btn.clicked.connect(self.set_baseline_requested.emit)
        top_bar.addWidget(self.set_baseline_btn)

        stop_btn = QPushButton("STOP SESSION")
        stop_btn.setObjectName("StopBtn")
        stop_btn.clicked.connect(self.stop_requested.emit)
        top_bar.addWidget(stop_btn)
        main_layout.addLayout(top_bar)

        # Content Area
        content_layout = QHBoxLayout()
        
        # Video Card
        video_card = QFrame()
        video_card.setObjectName("Card")
        video_vbox = QVBoxLayout(video_card)
        
        # Container for Video and Overlay
        self.video_container = QWidget()
        self.video_container_layout = QGridLayout(self.video_container)
        self.video_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel("INITIALIZING...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_container_layout.addWidget(self.video_label, 0, 0)
        
        # 알림 컨테이너: 비집중(위) + 거북목(아래) 세로 배치로 동시 표시 시 겹치지 않음
        self.alert_container = QWidget()
        alert_layout = QVBoxLayout(self.alert_container)
        alert_layout.setContentsMargins(0, 0, 0, 0)
        alert_layout.setSpacing(10)

        # Overlay Notification Label (red, 비집중)
        self.overlay_label = QLabel("DISTRACTION DETECTED!")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("""
            background-color: rgba(207, 102, 121, 200); 
            color: white; 
            font-size: 24px; 
            font-weight: bold; 
            border-radius: 10px;
            padding: 20px;
        """)
        self.overlay_label.setFixedSize(400, 100)
        self.overlay_label.hide()
        alert_layout.addWidget(self.overlay_label, 0, Qt.AlignmentFlag.AlignCenter)

        # 거북목 경고 (노란색, 2초간 표시 + 경고음, 비집중으로 카운트 안 함)
        self.posture_alert_label = QLabel("거북목")
        self.posture_alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.posture_alert_label.setStyleSheet("""
            background-color: rgba(255, 193, 7, 220);
            color: #000;
            font-size: 28px;
            font-weight: bold;
            border-radius: 10px;
            padding: 24px;
        """)
        self.posture_alert_label.setFixedSize(320, 90)
        self.posture_alert_label.hide()
        alert_layout.addWidget(self.posture_alert_label, 0, Qt.AlignmentFlag.AlignCenter)

        self.video_container_layout.addWidget(self.alert_container, 0, 0, Qt.AlignmentFlag.AlignCenter)
        
        video_vbox.addWidget(self.video_container)
        content_layout.addWidget(video_card)

        # Stats Column
        stats_layout = QVBoxLayout()
        
        # Distraction Warning Image (Hidden by default)
        self.warning_card = QFrame()
        self.warning_card.setObjectName("Card")
        self.warning_card.setStyleSheet("QFrame#Card { border: 2px solid #CF6679; }") # Red border
        warning_vbox = QVBoxLayout(self.warning_card)
        warning_label = QLabel("DISTRACTION CAPTURE")
        warning_label.setStyleSheet("color: #CF6679; font-weight: bold; font-size: 12px;")
        self.warning_img_label = QLabel()
        self.warning_img_label.setFixedSize(280, 210)
        self.warning_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warning_vbox.addWidget(warning_label)
        warning_vbox.addWidget(self.warning_img_label)
        self.warning_card.hide()
        stats_layout.addWidget(self.warning_card)

        # Score Card
        score_card = QFrame()
        score_card.setObjectName("Card")
        score_vbox = QVBoxLayout(score_card)
        self.add_stat(score_vbox, "FOCUS SCORE", "100%", "score")
        stats_layout.addWidget(score_card)

        # Other Stats Card
        info_card = QFrame()
        info_card.setObjectName("Card")
        info_grid = QGridLayout(info_card)
        self.add_stat_grid(info_grid, "DISTRACTIONS", "0", 0, 0, "dist")
        self.add_stat_grid(info_grid, "TIME", "00:00", 0, 1, "time")
        self.add_stat_grid(info_grid, "POSE", "Center", 1, 0, "pose")
        stats_layout.addWidget(info_card)
        
        content_layout.addLayout(stats_layout)
        main_layout.addLayout(content_layout)

        # Chart Card
        chart_card = QFrame()
        chart_card.setObjectName("Card")
        chart_vbox = QVBoxLayout(chart_card)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1E1E1E')
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='#BB86FC', width=2))
        chart_vbox.addWidget(self.plot_widget)
        main_layout.addWidget(chart_card)

    def add_stat(self, layout, label, value, key):
        lbl = QLabel(label); lbl.setObjectName("StatLabel")
        val = QLabel(value); val.setObjectName("StatValue")
        layout.addWidget(lbl); layout.addWidget(val)
        setattr(self, f"stat_{key}", val)

    def add_stat_grid(self, layout, label, value, r, c, key):
        vbox = QVBoxLayout()
        lbl = QLabel(label); lbl.setObjectName("StatLabel")
        val = QLabel(value); val.setObjectName("StatValue")
        val.setStyleSheet("font-size: 24px;")
        vbox.addWidget(lbl); vbox.addWidget(val)
        layout.addLayout(vbox, r, c)
        setattr(self, f"stat_{key}", val)

class ReportPage(QWidget):
    """모니터링 종료 후 리포트 화면 (통계 중심)"""
    home_requested = pyqtSignal()
    llm_analysis_requested = pyqtSignal(str) # session_id
    
    def __init__(self):
        super().__init__()
        self.current_session_id = None
        self.network_client = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Top Bar
        top_bar = QHBoxLayout()
        title = QLabel("SESSION REPORT")
        title.setObjectName("Title")
        top_bar.addWidget(title)
        top_bar.addStretch()
        
        self.llm_btn = QPushButton("LLM 분석 시작")
        self.llm_btn.setObjectName("SecondaryBtn")
        self.llm_btn.clicked.connect(lambda: self.llm_analysis_requested.emit(self.current_session_id))
        top_bar.addWidget(self.llm_btn)
        
        home_btn = QPushButton("BACK TO HOME")
        home_btn.setObjectName("PrimaryBtn")
        home_btn.clicked.connect(self.home_requested.emit)
        top_bar.addWidget(home_btn)
        main_layout.addLayout(top_bar)

        # Stats Card
        stats_card = QFrame()
        stats_card.setObjectName("Card")
        stats_layout = QHBoxLayout(stats_card)
        stats_layout.setSpacing(30)
        stats_layout.setContentsMargins(30, 20, 30, 20)
        
        stats_grid = QGridLayout()
        stats_grid.setSpacing(20)
        self.add_report_stat(stats_grid, "FOCUS RATIO", "0%", 0, 0, "ratio")
        self.add_report_stat(stats_grid, "DISTRACTIONS", "0", 0, 1, "dist")
        self.add_report_stat(stats_grid, "DURATION", "00:00", 0, 2, "duration")
        stats_layout.addLayout(stats_grid)
        stats_layout.addStretch()
        main_layout.addWidget(stats_card)

        # Graphs Section
        graphs_layout = QHBoxLayout()
        graphs_layout.setSpacing(15)

        # Left Column: Bar Chart
        bar_card = QFrame()
        bar_card.setObjectName("Card")
        bar_vbox = QVBoxLayout(bar_card)
        bar_vbox.setContentsMargins(15, 15, 15, 15)
        bar_title = QLabel("집중 시간 vs 비집중 시간")
        bar_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #BB86FC; margin-bottom: 10px;")
        bar_vbox.addWidget(bar_title)
        self.bar_plot = pg.PlotWidget()
        self.bar_plot.setBackground('#1E1E1E')
        self.bar_plot.setLabel('left', '시간 (초)')
        self.bar_plot.setLabel('bottom', '')
        self.bar_plot.getAxis('bottom').setTicks([[(0, '집중'), (1, '비집중')]])
        bar_vbox.addWidget(self.bar_plot)
        graphs_layout.addWidget(bar_card, stretch=1)

        # Right Column: Two Line Charts
        right_column = QVBoxLayout()
        right_column.setSpacing(15)

        # Line Chart 1: Time-based focus status
        line1_card = QFrame()
        line1_card.setObjectName("Card")
        line1_vbox = QVBoxLayout(line1_card)
        line1_vbox.setContentsMargins(15, 15, 15, 15)
        line1_title = QLabel("시간에 따른 집중 여부")
        line1_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #BB86FC; margin-bottom: 10px;")
        line1_vbox.addWidget(line1_title)
        self.line1_plot = pg.PlotWidget()
        self.line1_plot.setBackground('#1E1E1E')
        self.line1_plot.setLabel('left', '집중 여부')
        self.line1_plot.setLabel('bottom', '시간 (초)')
        self.line1_plot.setYRange(0, 1.2)
        self.line1_plot.getAxis('left').setTicks([[(0, '비집중'), (1, '집중')]])
        line1_vbox.addWidget(self.line1_plot)
        right_column.addWidget(line1_card, stretch=1)

        # Line Chart 2: 10-minute interval focus duration count
        line2_card = QFrame()
        line2_card.setObjectName("Card")
        line2_vbox = QVBoxLayout(line2_card)
        line2_vbox.setContentsMargins(15, 15, 15, 15)
        line2_title = QLabel("10분 단위 구간별 연속 집중 시간 횟수")
        line2_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #BB86FC; margin-bottom: 10px;")
        line2_vbox.addWidget(line2_title)
        self.line2_plot = pg.PlotWidget()
        self.line2_plot.setBackground('#1E1E1E')
        self.line2_plot.setLabel('left', '횟수')
        self.line2_plot.setLabel('bottom', '구간 (10분 단위)')
        line2_vbox.addWidget(self.line2_plot)
        right_column.addWidget(line2_card, stretch=1)

        graphs_layout.addLayout(right_column, stretch=2)
        main_layout.addLayout(graphs_layout, stretch=1)

    def set_llm_button_visible(self, visible: bool):
        self.llm_btn.setVisible(visible)

    def set_network_client(self, network_client):
        """NetworkClient 설정"""
        self.network_client = network_client

    def add_report_stat(self, layout, label, value, r, c, key):
        vbox = QVBoxLayout()
        lbl = QLabel(label); lbl.setObjectName("StatLabel")
        val = QLabel(value); val.setObjectName("StatValue")
        val.setStyleSheet("font-size: 28px;")
        vbox.addWidget(lbl); vbox.addWidget(val)
        layout.addLayout(vbox, r, c)
        setattr(self, f"report_{key}", val)

    def set_report_data(self, data):
        self.current_session_id = data.get('session_id')
        self.report_ratio.setText(f"{int(data.get('focus_ratio', 0))}%")
        self.report_dist.setText(str(data.get('distraction_count', 0)))
        
        try:
            start = datetime.fromisoformat(data['start_time'].replace('Z', ''))
            end = datetime.fromisoformat(data['end_time'].replace('Z', ''))
            duration = end - start
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds() % 60)
            self.report_duration.setText(f"{minutes:02d}:{seconds:02d}")
        except:
            self.report_duration.setText("00:00")
        
        # 그래프 데이터 로드 및 표시
        if self.network_client and self.current_session_id:
            QTimer.singleShot(100, self.load_and_draw_graphs)

    def load_and_draw_graphs(self):
        """로그 데이터를 가져와서 그래프 그리기"""
        if not self.network_client or not self.current_session_id:
            return
        
        logs_data = self.network_client.get_session_logs(self.current_session_id)
        if not logs_data or not logs_data.get('logs'):
            return
        
        logs = logs_data['logs']
        if not logs:
            return
        
        # 로그 데이터 파싱
        timestamps = []
        is_distracted_list = []
        
        for log in logs:
            try:
                ts = datetime.fromisoformat(log['timestamp'].replace('Z', ''))
                timestamps.append(ts)
                is_distracted_list.append(1 if log['is_distracted'] else 0)
            except:
                continue
        
        if not timestamps:
            return
        
        # 시작 시간 기준으로 상대 시간 계산 (초 단위)
        start_time = timestamps[0]
        relative_times = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # 1. 막대 그래프: 집중 시간 vs 비집중 시간
        self.draw_bar_chart(is_distracted_list, relative_times)
        
        # 2. 선 그래프 1: 시간에 따른 집중 여부
        self.draw_focus_status_chart(relative_times, is_distracted_list)
        
        # 3. 선 그래프 2: 10분 단위 구간별 연속 집중 시간 횟수
        self.draw_focus_duration_chart(relative_times, is_distracted_list)

    def draw_bar_chart(self, is_distracted_list, relative_times):
        """막대 그래프: 집중 시간 vs 비집중 시간"""
        self.bar_plot.clear()
        
        if not relative_times or len(relative_times) < 2:
            return
        
        # 실제 타임스탬프 차이를 사용하여 시간 계산
        focused_time = 0.0
        distracted_time = 0.0
        
        for i in range(len(relative_times) - 1):
            time_diff = relative_times[i + 1] - relative_times[i]
            if is_distracted_list[i]:
                distracted_time += time_diff
            else:
                focused_time += time_diff
        
        # 마지막 로그의 시간도 포함 (마지막 로그부터 세션 종료까지)
        if len(relative_times) > 0:
            last_time_diff = 3.0  # 기본값: 마지막 로그 이후 3초
            if is_distracted_list[-1]:
                distracted_time += last_time_diff
            else:
                focused_time += last_time_diff
        
        # 막대 그래프 그리기
        bg1 = pg.BarGraphItem(x=[0], height=[focused_time], width=0.6, brush='#03DAC6')
        bg2 = pg.BarGraphItem(x=[1], height=[distracted_time], width=0.6, brush='#CF6679')
        self.bar_plot.addItem(bg1)
        self.bar_plot.addItem(bg2)
        self.bar_plot.setXRange(-0.5, 1.5)

    def draw_focus_status_chart(self, relative_times, is_distracted_list):
        """선 그래프 1: 시간에 따른 집중 여부"""
        self.line1_plot.clear()
        
        # 집중 여부를 0(비집중) 또는 1(집중)로 표시
        focus_values = [1 if not dist else 0 for dist in is_distracted_list]
        
        pen = pg.mkPen(color='#BB86FC', width=2)
        self.line1_plot.plot(relative_times, focus_values, pen=pen)
        self.line1_plot.setXRange(min(relative_times) if relative_times else 0, 
                                  max(relative_times) if relative_times else 1)

    def draw_focus_duration_chart(self, relative_times, is_distracted_list):
        """선 그래프 2: 10분 단위 구간별 연속 집중 시간 횟수"""
        self.line2_plot.clear()
        
        if not relative_times:
            return
        
        # 10분(600초) 단위로 구간 나누기
        interval_seconds = 600
        max_time = max(relative_times)
        num_intervals = int(max_time / interval_seconds) + 1
        
        # 각 구간별로 연속 집중 시간 세기
        interval_counts = []
        interval_labels = []
        
        for i in range(num_intervals):
            interval_start = i * interval_seconds
            interval_end = (i + 1) * interval_seconds
            
            # 해당 구간의 로그 찾기
            interval_logs = [(t, dist) for t, dist in zip(relative_times, is_distracted_list) 
                           if interval_start <= t < interval_end]
            
            if not interval_logs:
                interval_counts.append(0)
                interval_labels.append(f"{i*10}분")
                continue
            
            # 연속 집중 시간 세기 (연속된 집중 구간의 개수)
            consecutive_focus_count = 0
            in_focus_sequence = False
            
            for j, (t, dist) in enumerate(interval_logs):
                if not dist:  # 집중 중
                    if not in_focus_sequence:
                        # 새로운 집중 구간 시작
                        consecutive_focus_count += 1
                        in_focus_sequence = True
                else:  # 비집중
                    in_focus_sequence = False
            
            interval_counts.append(consecutive_focus_count)
            interval_labels.append(f"{i*10}분")
        
        # 그래프 그리기
        x_values = list(range(num_intervals))
        pen = pg.mkPen(color='#03DAC6', width=2)
        self.line2_plot.plot(x_values, interval_counts, pen=pen, symbol='o', symbolBrush='#03DAC6')
        self.line2_plot.setXRange(-0.5, num_intervals - 0.5)
        
        # X축 레이블 설정
        ticks = [(i, interval_labels[i]) for i in range(num_intervals) if i % max(1, num_intervals // 10) == 0]
        self.line2_plot.getAxis('bottom').setTicks([ticks])

class LlmAnalysisPage(QWidget):
    """LLM 분석 전용 화면"""
    home_requested = pyqtSignal()
    
    def __init__(self, network_client):
        super().__init__()
        self.network_client = network_client
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setFixedSize(700, 550)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(20)
        card_layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("AI FOCUS ANALYSIS")
        title.setObjectName("Title")
        card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # Feedback Area
        self.feedback_text = QLabel("AI가 당신의 집중 패턴을 분석하고 있습니다. 잠시만 기다려주세요...")
        self.feedback_text.setWordWrap(True)
        self.feedback_text.setStyleSheet("""
            font-size: 16px; 
            color: #E0E0E0; 
            background: #2D2D2D; 
            padding: 25px; 
            border-radius: 12px;
            line-height: 1.6;
        """)
        self.feedback_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        card_layout.addWidget(self.feedback_text)

        home_btn = QPushButton("BACK TO HOME")
        home_btn.setObjectName("PrimaryBtn")
        home_btn.clicked.connect(self.home_requested.emit)
        card_layout.addWidget(home_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(card)

    def start_analysis(self, session_id):
        self.feedback_text.setText("AI가 당신의 집중 패턴을 분석하고 있습니다. 잠시만 기다려주세요...")
        # 비동기 요청을 위해 타이머 사용
        QTimer.singleShot(100, lambda: self._run_request(session_id))

    def _run_request(self, session_id):
        result = self.network_client.request_llm_feedback(session_id)
        if result and result.get('llm_comment'):
            self.feedback_text.setText(result['llm_comment'])
        else:
            self.feedback_text.setText("분석 중 오류가 발생했습니다. 나중에 다시 시도해주세요.")

from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

class HistoryPage(QWidget):
    """과거 기록 조회 화면 (FM-601)"""
    home_requested = pyqtSignal()
    def __init__(self, network_client):
        super().__init__()
        self.network_client = network_client
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        top_bar = QHBoxLayout()
        header = QLabel("PAST SESSIONS")
        header.setObjectName("Title")
        top_bar.addWidget(header)
        top_bar.addStretch()
        
        home_btn = QPushButton("HOME")
        home_btn.setObjectName("SecondaryBtn")
        home_btn.clicked.connect(self.home_requested.emit)
        top_bar.addWidget(home_btn)
        layout.addLayout(top_bar)

        # History Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Date", "Focus %", "Distractions", "Duration", "ID"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget { 
                background-color: #1E1E1E; 
                gridline-color: #333; 
                border-radius: 8px;
                font-size: 14px;
            }
            QHeaderView::section { 
                background-color: #2D2D2D; 
                color: #BB86FC; 
                padding: 10px;
                font-weight: bold;
            }
            QTableWidget::item { padding: 10px; }
        """)
        self.table.itemDoubleClicked.connect(self.show_detail)
        layout.addWidget(self.table)

    def load_data(self):
        history = self.network_client.get_history()
        self.table.setRowCount(len(history))
        for i, session in enumerate(history):
            # Date
            start_time = datetime.fromisoformat(session['start_time'].replace('Z', ''))
            self.table.setItem(i, 0, QTableWidgetItem(start_time.strftime("%Y-%m-%d %H:%M")))
            # Focus Ratio
            self.table.setItem(i, 1, QTableWidgetItem(f"{int(session['focus_ratio'])}%"))
            # Distractions
            self.table.setItem(i, 2, QTableWidgetItem(str(session['distraction_count'])))
            # Duration
            try:
                end_time = datetime.fromisoformat(session['end_time'].replace('Z', ''))
                duration = end_time - start_time
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                self.table.setItem(i, 3, QTableWidgetItem(f"{minutes:02d}:{seconds:02d}"))
            except:
                self.table.setItem(i, 3, QTableWidgetItem("-"))
            # ID
            self.table.setItem(i, 4, QTableWidgetItem(session['session_id'][:8] + "..."))
            self.table.item(i, 4).setData(Qt.ItemDataRole.UserRole, session)

    def show_detail(self, item):
        row = item.row()
        session_data = self.table.item(row, 4).data(Qt.ItemDataRole.UserRole)
        self.parent().parent().show_report_detail(session_data)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FOCUS MONITOR PRO")
        self.resize(1100, 850)
        self.setStyleSheet(STYLE_SHEET)

        self.camera = Camera()
        self.network_client = NetworkClient()
        self.is_monitoring = False
        
        # Pages Setup
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.main_page = MainPage()
        self.calibration_page = DistanceCalibrationPage(self.camera, self.network_client)
        self.monitoring_page = MonitoringPage()
        self.report_page = ReportPage()
        self.report_page.set_network_client(self.network_client)
        self.analysis_page = LlmAnalysisPage(self.network_client)
        self.history_page = HistoryPage(self.network_client)

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.calibration_page)
        self.stack.addWidget(self.monitoring_page)
        self.stack.addWidget(self.report_page)
        self.stack.addWidget(self.analysis_page)
        self.stack.addWidget(self.history_page)

        # Signals
        self.main_page.start_requested.connect(self.start_session)
        self.main_page.history_requested.connect(self.show_history)
        self.main_page.calibration_requested.connect(lambda: self.stack.setCurrentWidget(self.calibration_page))
        self.calibration_page.done_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        self.monitoring_page.stop_requested.connect(self.stop_session)
        self.monitoring_page.set_baseline_requested.connect(self._on_set_baseline_from_monitoring)
        self.report_page.home_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        self.report_page.llm_analysis_requested.connect(self.show_analysis)
        self.analysis_page.home_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        self.history_page.home_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))

        # Timers
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_ui)
        self.preview_timer.start(33)

        self.conn_timer = QTimer()
        self.conn_timer.timeout.connect(self.check_server_connection)
        self.conn_timer.start(2000)

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.request_inference)

        # State
        self.start_time = 0
        self.history_scores = []
        self.distraction_count = 0
        self.current_session_id = None

    def start_session(self):
        session_data = self.network_client.start_session()
        if session_data and session_data.get("session_id"):
            self.current_session_id = session_data.get("session_id")
            logger.info(f"Session started on server: {self.current_session_id}")
            
            self.is_monitoring = True
            self.start_time = time.time()
            self.distraction_count = 0
            self.history_scores = []
            self.stack.setCurrentWidget(self.monitoring_page)
            self.inference_timer.start(3000)
        else:
            logger.error("Failed to connect to Operation Server.")
            QMessageBox.critical(self, "Connection Error", 
                                "운영 서버와 연결할 수 없습니다.\n서버 상태를 확인하고 다시 시도해주세요.")
            self.check_server_connection()

    def stop_session(self):
        if not self.is_monitoring: return
        
        self.is_monitoring = False
        self.inference_timer.stop()
        
        summary = None
        if self.current_session_id and not str(self.current_session_id).startswith("local_"):
            summary = self.network_client.stop_session(self.current_session_id)
        
        if not summary:
            elapsed = int(time.time() - self.start_time)
            summary = {
                "session_id": self.current_session_id,
                "focus_ratio": max(0, 100 - (self.distraction_count * 2)),
                "distraction_count": self.distraction_count,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        self.report_page.set_report_data(summary)
        self.report_page.set_llm_button_visible(True) # 새로 종료된 세션은 분석 버튼 보임
        self.stack.setCurrentWidget(self.report_page)
        self.current_session_id = None

    def show_analysis(self, session_id):
        self.analysis_page.start_analysis(session_id)
        self.stack.setCurrentWidget(self.analysis_page)

    def show_history(self):
        self.history_page.load_data()
        self.stack.setCurrentWidget(self.history_page)

    def _on_set_baseline_from_monitoring(self):
        """공부 중 정자세 다시 설정: 현재 프레임으로 ai_body set_baseline 호출."""
        if not hasattr(self, "current_frame") or self.current_frame is None:
            QMessageBox.warning(self, "정자세 설정", "카메라 프레임을 읽을 수 없습니다.")
            return
        image_base64 = self.camera.frame_to_base64(self.current_frame)
        ok = self.network_client.set_baseline(image_base64)
        if ok:
            QMessageBox.information(self, "정자세 설정", "정자세가 다시 설정되었습니다.")
        else:
            QMessageBox.warning(self, "정자세 설정", "ai_body 서버 연결에 실패했습니다.\n서버가 실행 중인지 확인하세요.")

    def show_report_detail(self, session_data):
        self.report_page.set_report_data(session_data)
        self.report_page.set_llm_button_visible(False) # 과거 기록 조회 시에는 버튼 숨김
        self.stack.setCurrentWidget(self.report_page)

    def check_server_connection(self):
        if not self.is_monitoring:
            connected = self.network_client.check_connection()
            self.main_page.set_connection_status(connected)

    def update_ui(self):
        frame = self.camera.get_frame()
        if frame is not None:
            if self.stack.currentWidget() == self.monitoring_page:
                frame_mirror = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                self.monitoring_page.video_label.setPixmap(QPixmap.fromImage(img).scaled(
                    640, 480, Qt.AspectRatioMode.KeepAspectRatio))
                
                elapsed = int(time.time() - self.start_time)
                self.monitoring_page.stat_time.setText(f"{elapsed//60:02d}:{elapsed%60:02d}")
            self.current_frame = frame

    def request_inference(self):
        if self.is_monitoring and hasattr(self, 'current_frame'):
            self.sent_frame = self.current_frame.copy()
            image_base64 = self.camera.frame_to_base64(self.sent_frame)
            self.thread = InferenceThread(self.network_client, image_base64, self.current_session_id)
            self.thread.result_ready.connect(self.handle_result)
            self.thread.start()

    def handle_result(self, result):
        if not self.is_monitoring: return
        
        m_page = self.monitoring_page
        if result.is_distracted:
            self.distraction_count += 1
            m_page.stat_dist.setText(str(self.distraction_count))
            m_page.stat_score.setStyleSheet("color: #CF6679;")
            m_page.stat_pose.setText("Away")
            
            if hasattr(self, 'sent_frame'):
                rgb = cv2.cvtColor(cv2.flip(self.sent_frame, 1), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                m_page.warning_img_label.setPixmap(QPixmap.fromImage(img).scaled(
                    280, 210, Qt.AspectRatioMode.KeepAspectRatio))
                m_page.warning_card.show()
            
            m_page.overlay_label.setText(f"ATTENTION!\n{result.status_message}")
            m_page.overlay_label.show()
            QApplication.beep()
            QTimer.singleShot(2000, m_page.overlay_label.hide)
        else:
            m_page.stat_score.setStyleSheet("color: #03DAC6;")
            m_page.stat_pose.setText("Centered")
            m_page.overlay_label.hide()

        # 거북목 경고: 비집중으로 카운트하지 않고, 노란 경고 2초 + 경고음만
        if result.body_pose and result.body_pose.get("posture_alert"):
            m_page.posture_alert_label.setText("거북목")
            m_page.posture_alert_label.show()
            QApplication.beep()
            QTimer.singleShot(2000, m_page.posture_alert_label.hide)

        score = max(0, 100 - (self.distraction_count * 2))
        m_page.stat_score.setText(f"{int(score)}%")
        self.history_scores.append(score)
        if len(self.history_scores) > 100: self.history_scores.pop(0)
        m_page.curve.setData(self.history_scores)

    def closeEvent(self, event):
        self.camera.release()
        event.accept()
