import sys
import cv2
import time
import logging
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QGridLayout, QStackedWidget, QMessageBox, QApplication)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont
import pyqtgraph as pg

from client.core.camera import Camera
from client.core.network import NetworkClient

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
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Welcome Card
        card = QFrame()
        card.setObjectName("Card")
        card.setFixedSize(500, 400)
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
        self.status_label.setStyleSheet("color: #FFB74D; font-weight: bold;") # Orange for waiting
        card_layout.addWidget(self.status_label)

        self.start_btn = QPushButton("START NEW SESSION")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setEnabled(False) # Disabled by default
        self.start_btn.setStyleSheet("QPushButton:disabled { background-color: #333; color: #666; }")
        self.start_btn.clicked.connect(self.start_requested.emit)
        card_layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

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

class MonitoringPage(QWidget):
    """실시간 모니터링 화면"""
    stop_requested = pyqtSignal()
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
        
        # Overlay Notification Label
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
        self.video_container_layout.addWidget(self.overlay_label, 0, 0, Qt.AlignmentFlag.AlignCenter)
        
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
    """모니터링 종료 후 리포트 화면 (FM-501)"""
    home_requested = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setFixedSize(600, 550)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(20)

        title = QLabel("SESSION REPORT")
        title.setObjectName("Title")
        card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # Stats Grid
        stats_grid = QGridLayout()
        self.add_report_stat(stats_grid, "FOCUS RATIO", "0%", 0, 0, "ratio")
        self.add_report_stat(stats_grid, "DISTRACTIONS", "0", 0, 1, "dist")
        self.add_report_stat(stats_grid, "DURATION", "00:00", 1, 0, "duration")
        card_layout.addLayout(stats_grid)

        # LLM Feedback
        feedback_label = QLabel("AI FEEDBACK")
        feedback_label.setObjectName("StatLabel")
        card_layout.addWidget(feedback_label)
        
        self.feedback_text = QLabel("Calculating your focus pattern...")
        self.feedback_text.setWordWrap(True)
        self.feedback_text.setStyleSheet("font-size: 16px; color: #E0E0E0; background: #2D2D2D; padding: 15px; border-radius: 8px;")
        card_layout.addWidget(self.feedback_text)

        home_btn = QPushButton("BACK TO HOME")
        home_btn.setObjectName("PrimaryBtn")
        home_btn.clicked.connect(self.home_requested.emit)
        card_layout.addWidget(home_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(card)

    def add_report_stat(self, layout, label, value, r, c, key):
        vbox = QVBoxLayout()
        lbl = QLabel(label); lbl.setObjectName("StatLabel")
        val = QLabel(value); val.setObjectName("StatValue")
        vbox.addWidget(lbl); vbox.addWidget(val)
        layout.addLayout(vbox, r, c)
        setattr(self, f"report_{key}", val)

    def set_report_data(self, data):
        self.report_ratio.setText(f"{int(data.get('focus_ratio', 0))}%")
        self.report_dist.setText(str(data.get('distraction_count', 0)))
        
        # Duration calculation
        try:
            from datetime import datetime
            start = datetime.fromisoformat(data['start_time'].replace('Z', ''))
            end = datetime.fromisoformat(data['end_time'].replace('Z', ''))
            duration = end - start
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds() % 60)
            self.report_duration.setText(f"{minutes:02d}:{seconds:02d}")
        except:
            self.report_duration.setText("00:00")
            
        self.feedback_text.setText(data.get('llm_comment') or "No feedback available.")

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
        self.monitoring_page = MonitoringPage()
        self.report_page = ReportPage()

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.monitoring_page)
        self.stack.addWidget(self.report_page)

        # Signals
        self.main_page.start_requested.connect(self.start_session)
        self.monitoring_page.stop_requested.connect(self.stop_session)
        self.report_page.home_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))

        # Timers
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_ui)
        self.preview_timer.start(33)

        self.conn_timer = QTimer()
        self.conn_timer.timeout.connect(self.check_server_connection)
        self.conn_timer.start(2000) # Check every 2 seconds

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.request_inference)

        # State
        self.start_time = 0
        self.history_scores = []
        self.distraction_count = 0
        self.current_session_id = None

    def start_session(self):
        # Start session in DB
        session_data = self.network_client.start_session()
        if session_data:
            self.current_session_id = session_data.get("session_id")
            logger.info(f"Session started: {self.current_session_id}")
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.distraction_count = 0
        self.history_scores = []
        self.stack.setCurrentWidget(self.monitoring_page)
        self.inference_timer.start(3000)

    def stop_session(self):
        self.is_monitoring = False
        self.inference_timer.stop()
        
        # Stop session and get summary
        if self.current_session_id:
            summary = self.network_client.stop_session(self.current_session_id)
            if summary:
                self.report_page.set_report_data(summary)
                self.stack.setCurrentWidget(self.report_page)
                self.current_session_id = None
                return

        self.stack.setCurrentWidget(self.main_page)

    def check_server_connection(self):
        if not self.is_monitoring:
            connected = self.network_client.check_connection()
            self.main_page.set_connection_status(connected)

    def update_ui(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # 씬이 모니터링 화면일 때만 카메라 피드 업데이트
            if self.stack.currentWidget() == self.monitoring_page:
                frame_mirror = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                self.monitoring_page.video_label.setPixmap(QPixmap.fromImage(img).scaled(
                    640, 480, Qt.AspectRatioMode.KeepAspectRatio))
                
                # 타이머 업데이트
                elapsed = int(time.time() - self.start_time)
                self.monitoring_page.stat_time.setText(f"{elapsed//60:02d}:{elapsed%60:02d}")
            self.current_frame = frame

    def request_inference(self):
        if self.is_monitoring and hasattr(self, 'current_frame'):
            self.sent_frame = self.current_frame.copy() # Capture current frame for warning display
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
            
            # Show Distraction Image
            if hasattr(self, 'sent_frame'):
                rgb = cv2.cvtColor(cv2.flip(self.sent_frame, 1), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                m_page.warning_img_label.setPixmap(QPixmap.fromImage(img).scaled(
                    280, 210, Qt.AspectRatioMode.KeepAspectRatio))
                m_page.warning_card.show()
            
            # FM-402: Show Overlay and Play Beep
            m_page.overlay_label.setText(f"ATTENTION!\n{result.status_message}")
            m_page.overlay_label.show()
            QApplication.beep()
            
            # Hide overlay after 2 seconds
            QTimer.singleShot(2000, m_page.overlay_label.hide)
            
        else:
            m_page.stat_score.setStyleSheet("color: #03DAC6;")
            m_page.stat_pose.setText("Centered")
            # Hide overlay if user returns to focus
            m_page.overlay_label.hide()

        score = max(0, 100 - (self.distraction_count * 2))
        m_page.stat_score.setText(f"{int(score)}%")
        self.history_scores.append(score)
        if len(self.history_scores) > 100: self.history_scores.pop(0)
        m_page.curve.setData(self.history_scores)

    def closeEvent(self, event):
        self.camera.release()
        event.accept()
