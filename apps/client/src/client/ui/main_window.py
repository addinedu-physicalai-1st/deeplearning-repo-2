import sys
import json
import cv2
import time
import logging
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFrame, QGridLayout, QStackedWidget, QMessageBox, QApplication,
                             QProgressBar, QDateEdit, QScrollArea)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect, QRectF, QEasingCurve, QDate, QElapsedTimer
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen, QBrush, QPainterPath
import pyqtgraph as pg
from datetime import datetime, timedelta, timezone, date

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
    head_calibration_requested = pyqtSignal()

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

        self.head_calibration_btn = QPushButton("머리 각도 캘리브레이션")
        self.head_calibration_btn.setObjectName("SecondaryBtn")
        self.head_calibration_btn.clicked.connect(self.head_calibration_requested.emit)
        card_layout.addWidget(self.head_calibration_btn, alignment=Qt.AlignmentFlag.AlignCenter)

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


class HeadPoseCalibrationPage(QWidget):
    """머리 각도 캘리브레이션 화면 (yaw, pitch 임계값 설정). ai_body DistanceCalibrationPage 참조."""
    done_requested = pyqtSignal()

    def __init__(self, camera: Camera, network_client: NetworkClient):
        super().__init__()
        self.camera = camera
        self.network_client = network_client
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.current_step = 0
        self.measuring = False
        self.measure_timer = QElapsedTimer()
        self.measure_duration_ms = 3000
        self.measured_values = {"pitch_up": None, "pitch_down": None, "yaw_left": None, "yaw_right": None}
        self.current_pitch = None
        self.current_yaw = None
        self.current_roll = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self.video_label = QLabel("Camera")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setFixedSize(640, 480)
        left.addWidget(self.video_label)
        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFB74D;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.status_label)
        layout.addLayout(left)

        panel = QFrame()
        panel.setObjectName("Card")
        panel.setStyleSheet("QFrame#Card { background-color: #1E1E1E; border-radius: 12px; border: 1px solid #333; }")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(12)
        title = QLabel("머리 각도 캘리브레이션")
        title.setObjectName("Title")
        title.setStyleSheet("font-size: 22px; color: #BB86FC;")
        panel_layout.addWidget(title)
        self.step_label = QLabel("단계: 대기 중")
        self.step_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #03DAC6;")
        panel_layout.addWidget(self.step_label)
        self.pitch_label = QLabel("Pitch (위/아래): --°")
        self.pitch_label.setStyleSheet("font-size: 16px; color: #9E9E9E;")
        panel_layout.addWidget(self.pitch_label)
        self.yaw_label = QLabel("Yaw (좌/우): --°")
        self.yaw_label.setStyleSheet("font-size: 16px; color: #9E9E9E;")
        panel_layout.addWidget(self.yaw_label)
        self.roll_label = QLabel("Roll (기울기): --°")
        self.roll_label.setStyleSheet("font-size: 16px; color: #9E9E9E;")
        panel_layout.addWidget(self.roll_label)
        panel_layout.addSpacing(10)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { border: 2px solid #03DAC6; border-radius: 5px; background-color: #333; } QProgressBar::chunk { background-color: #03DAC6; }")
        self.progress_bar.hide()
        panel_layout.addWidget(self.progress_bar)
        self.measured_label = QLabel("")
        self.measured_label.setStyleSheet("font-size: 14px; color: #64B5F6;")
        self.measured_label.setWordWrap(True)
        panel_layout.addWidget(self.measured_label)
        panel_layout.addStretch(1)
        self.measure_btn = QPushButton("측정 시작")
        self.measure_btn.setObjectName("PrimaryBtn")
        self.measure_btn.clicked.connect(self._on_measure)
        panel_layout.addWidget(self.measure_btn)
        self.next_step_btn = QPushButton("다음 단계")
        self.next_step_btn.setObjectName("SecondaryBtn")
        self.next_step_btn.clicked.connect(self._on_next_step)
        self.next_step_btn.setEnabled(False)
        panel_layout.addWidget(self.next_step_btn)
        self.set_thresholds_btn = QPushButton("임계값 설정")
        self.set_thresholds_btn.setObjectName("PrimaryBtn")
        self.set_thresholds_btn.clicked.connect(self._on_set_thresholds)
        self.set_thresholds_btn.setEnabled(False)
        panel_layout.addWidget(self.set_thresholds_btn)
        self.done_btn = QPushButton("완료")
        self.done_btn.setObjectName("SecondaryBtn")
        self.done_btn.clicked.connect(self.done_requested.emit)
        panel_layout.addWidget(self.done_btn)
        layout.addWidget(panel)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start(33)
        self._reset_calibration()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def _reset_calibration(self):
        self.current_step = 0
        self.measuring = False
        self.measured_values = {"pitch_up": None, "pitch_down": None, "yaw_left": None, "yaw_right": None}
        self._update_ui()

    def _update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return
        self.current_frame = frame
        image_base64 = self.camera.frame_to_base64(frame)
        head_pose = self.network_client.get_head_pose(image_base64)
        if head_pose:
            self.current_pitch = head_pose.get("pitch")
            self.current_yaw = head_pose.get("yaw")
            self.current_roll = head_pose.get("roll")
        else:
            self.current_pitch = None
            self.current_yaw = None
            self.current_roll = None
        display_frame = cv2.flip(frame, 1)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if self.current_pitch is not None:
            self.pitch_label.setText(f"Pitch (위/아래): {self.current_pitch:.1f}°")
        else:
            self.pitch_label.setText("Pitch (위/아래): --°")
        if self.current_yaw is not None:
            self.yaw_label.setText(f"Yaw (좌/우): {self.current_yaw:.1f}°")
        else:
            self.yaw_label.setText("Yaw (좌/우): --°")
        if self.current_roll is not None:
            self.roll_label.setText(f"Roll (기울기): {self.current_roll:.1f}°")
        else:
            self.roll_label.setText("Roll (기울기): --°")
        if self.measuring and self.measure_timer.isValid():
            elapsed_ms = self.measure_timer.elapsed()
            progress = min(100, int((elapsed_ms / self.measure_duration_ms) * 100))
            self.progress_bar.setValue(progress)
            if elapsed_ms >= self.measure_duration_ms:
                self._finish_measurement()
            else:
                self._track_angle()

    def _track_angle(self):
        if self.current_step == 1 and self.current_pitch is not None:
            if self.measured_values["pitch_up"] is None or self.current_pitch < self.measured_values["pitch_up"]:
                self.measured_values["pitch_up"] = self.current_pitch
        elif self.current_step == 2 and self.current_pitch is not None:
            if self.measured_values["pitch_down"] is None or self.current_pitch > self.measured_values["pitch_down"]:
                self.measured_values["pitch_down"] = self.current_pitch
        elif self.current_step == 3 and self.current_yaw is not None:
            if self.measured_values["yaw_left"] is None or self.current_yaw < self.measured_values["yaw_left"]:
                self.measured_values["yaw_left"] = self.current_yaw
        elif self.current_step == 4 and self.current_yaw is not None:
            if self.measured_values["yaw_right"] is None or self.current_yaw > self.measured_values["yaw_right"]:
                self.measured_values["yaw_right"] = self.current_yaw

    def _on_measure(self):
        if self.current_step == 0:
            self.current_step = 1
            self._update_ui()
        if 1 <= self.current_step <= 4:
            self.measuring = True
            self.measure_timer.start()
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.measure_btn.setEnabled(False)
            self.next_step_btn.setEnabled(False)

    def _finish_measurement(self):
        self.measuring = False
        self.progress_bar.hide()
        self.measure_btn.setEnabled(True)
        self.next_step_btn.setEnabled(True)
        self._update_measured_display()

    def _on_next_step(self):
        if self.current_step < 4:
            self.current_step += 1
            self._update_ui()
        elif self.current_step == 4:
            self.current_step = 5
            self._update_ui()

    def _update_ui(self):
        step_messages = {0: "준비 중...", 1: "1/4 단계: 위를 보세요", 2: "2/4 단계: 아래를 보세요", 3: "3/4 단계: 좌측을 보세요", 4: "4/4 단계: 우측을 보세요", 5: "측정 완료"}
        self.step_label.setText(f"단계: {step_messages.get(self.current_step, '')}")
        if self.current_step == 0:
            self.status_label.setText("측정을 시작하세요")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFB74D;")
            self.measure_btn.setText("측정 시작")
            self.measure_btn.setEnabled(True)
            self.next_step_btn.setEnabled(False)
            self.set_thresholds_btn.setEnabled(False)
        elif 1 <= self.current_step <= 4:
            self.status_label.setText(step_messages[self.current_step])
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #03DAC6;")
            self.measure_btn.setEnabled(not self.measuring)
            key = ["pitch_up", "pitch_down", "yaw_left", "yaw_right"][self.current_step - 1]
            self.next_step_btn.setEnabled(not self.measuring and self.measured_values.get(key) is not None)
        elif self.current_step == 5:
            self.status_label.setText("모든 측정이 완료되었습니다.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #03DAC6;")
            self.measure_btn.setEnabled(False)
            self.next_step_btn.setEnabled(False)
            self.set_thresholds_btn.setEnabled(True)
            self._update_measured_display()

    def _update_measured_display(self):
        parts = []
        if self.measured_values["pitch_up"] is not None:
            parts.append(f"위: {self.measured_values['pitch_up']:.1f}°")
        if self.measured_values["pitch_down"] is not None:
            parts.append(f"아래: {self.measured_values['pitch_down']:.1f}°")
        if self.measured_values["yaw_left"] is not None:
            parts.append(f"좌: {self.measured_values['yaw_left']:.1f}°")
        if self.measured_values["yaw_right"] is not None:
            parts.append(f"우: {self.measured_values['yaw_right']:.1f}°")
        self.measured_label.setText("측정값:\n" + "\n".join(parts) if parts else "")

    def _on_set_thresholds(self):
        pitch_up = self.measured_values.get("pitch_up")
        pitch_down = self.measured_values.get("pitch_down")
        yaw_left = self.measured_values.get("yaw_left")
        yaw_right = self.measured_values.get("yaw_right")
        if pitch_up is None or pitch_down is None or yaw_left is None or yaw_right is None:
            self.status_label.setText("모든 방향의 측정이 완료되지 않았습니다.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF9800;")
            return
        yaw_limit = max(abs(yaw_left), abs(yaw_right))
        ok = self.network_client.set_head_thresholds(pitch_up, pitch_down, yaw_limit)
        if ok:
            self.status_label.setText(f"임계값이 설정되었습니다.\nPitch: {pitch_up:.1f}° ~ {pitch_down:.1f}°\nYaw: ±{yaw_limit:.1f}°")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #03DAC6;")
        else:
            self.status_label.setText("ai_head 서버 연결 실패. 서버를 확인하세요.")
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


class DonutChartWidget(QWidget):
    """QPainter 기반 도넛 차트. 12시 방향에서 집중(시계방향), 비집중 순. 중앙에 집중도(%) 표시."""
    def __init__(self):
        super().__init__()
        self._focused_time = 0.0
        self._distracted_time = 0.0
        self._focus_ratio = 0
        self.setMinimumSize(200, 200)

    def setData(self, focused_time: float, distracted_time: float):
        self._focused_time = max(0.0, focused_time)
        self._distracted_time = max(0.0, distracted_time)
        total = self._focused_time + self._distracted_time
        self._focus_ratio = int(round((self._focused_time / total * 100) if total > 0 else 0))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        side = min(w, h)
        margin = 10
        rect_side = side - 2 * margin
        left = (w - rect_side) // 2
        top = (h - rect_side) // 2
        rect_f = QRectF(left, top, rect_side, rect_side)

        total = self._focused_time + self._distracted_time
        if total <= 0:
            painter.setPen(QPen(QColor("#555")))
            painter.setFont(QFont("Sans", 12))
            painter.drawText(rect_f, Qt.AlignmentFlag.AlignCenter, "--")
            return

        start_deg = 90.0
        focus_ratio = self._focused_time / total
        focus_span_deg = -(focus_ratio * 360.0)
        distract_span_deg = -((1.0 - focus_ratio) * 360.0)

        ring_width = max(12, rect_side // 6)
        inner_side = rect_side - 2 * ring_width
        inner_left = left + ring_width
        inner_top = top + ring_width
        inner_rect_f = QRectF(inner_left, inner_top, inner_side, inner_side)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#03DAC6")))
        path_focus = QPainterPath()
        path_focus.arcMoveTo(rect_f, start_deg)
        path_focus.arcTo(rect_f, start_deg, focus_span_deg)
        path_focus.arcTo(inner_rect_f, start_deg + focus_span_deg, -focus_span_deg)
        path_focus.closeSubpath()
        painter.drawPath(path_focus)

        painter.setBrush(QBrush(QColor("#CF6679")))
        path_distract = QPainterPath()
        path_distract.arcMoveTo(rect_f, start_deg + focus_span_deg)
        path_distract.arcTo(rect_f, start_deg + focus_span_deg, distract_span_deg)
        path_distract.arcTo(inner_rect_f, start_deg + focus_span_deg + distract_span_deg, -distract_span_deg)
        path_distract.closeSubpath()
        painter.drawPath(path_distract)

        painter.setPen(QPen(QColor("#E0E0E0")))
        painter.setFont(QFont("Sans", 14, QFont.Weight.Bold))
        painter.drawText(inner_rect_f, Qt.AlignmentFlag.AlignCenter, f"{self._focus_ratio}%")


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
        self.add_report_stat(stats_grid, "DURATION", "00:00", 0, 0, "duration")
        self.add_report_stat(stats_grid, "DISTRACTIONS", "0", 0, 1, "dist")
        self.add_report_stat(stats_grid, "최장 집중시간", "--", 0, 2, "longest_focus")
        self.add_report_stat(stats_grid, "평균 집중시간", "--", 0, 3, "avg_focus")
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
        self.donut_widget = DonutChartWidget()
        self.donut_widget.setMinimumSize(200, 200)
        self.donut_widget.setStyleSheet("background-color: #1E1E1E; border-radius: 8px;")
        bar_vbox.addWidget(self.donut_widget)
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
        self.line1_plot.setLabel('bottom', '시간')
        self.line1_plot.setYRange(0, 1.2)
        self.line1_plot.getAxis('left').setTicks([[(0, '비집중'), (1, '집중')]])
        line1_vbox.addWidget(self.line1_plot)
        right_column.addWidget(line1_card, stretch=1)

        # Line Chart 2: 10-minute interval focus duration count
        line2_card = QFrame()
        line2_card.setObjectName("Card")
        line2_vbox = QVBoxLayout(line2_card)
        line2_vbox.setContentsMargins(15, 15, 15, 15)
        line2_title = QLabel("10분 간격 Sleepy 감지 횟수")
        line2_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #BB86FC; margin-bottom: 10px;")
        line2_vbox.addWidget(line2_title)
        self.line2_plot = pg.PlotWidget()
        self.line2_plot.setBackground('#1E1E1E')
        self.line2_plot.setLabel('left', '횟수')
        self.line2_plot.setLabel('bottom', '')
        line2_vbox.addWidget(self.line2_plot)
        right_column.addWidget(line2_card, stretch=1)

        graphs_layout.addLayout(right_column, stretch=2)
        main_layout.addLayout(graphs_layout, stretch=1)

        # LLM 피드백 카드 (저장된 피드백 표시)
        feedback_card = QFrame()
        feedback_card.setObjectName("Card")
        feedback_card_vbox = QVBoxLayout(feedback_card)
        feedback_card_vbox.setContentsMargins(15, 15, 15, 15)
        feedback_title = QLabel("LLM 피드백")
        feedback_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #BB86FC; margin-bottom: 10px;")
        feedback_card_vbox.addWidget(feedback_title)
        self.report_llm_feedback_label = QLabel("저장된 LLM 피드백이 없습니다.")
        self.report_llm_feedback_label.setWordWrap(True)
        self.report_llm_feedback_label.setStyleSheet("""
            font-size: 14px; color: #E0E0E0; background: #2D2D2D; padding: 15px; border-radius: 8px; line-height: 1.5;
        """)
        self.report_llm_feedback_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.report_llm_feedback_label.setMinimumHeight(80)
        feedback_scroll = QScrollArea()
        feedback_scroll.setWidgetResizable(True)
        feedback_scroll.setWidget(self.report_llm_feedback_label)
        feedback_scroll.setMinimumHeight(130)
        feedback_scroll.setMaximumHeight(400)
        feedback_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feedback_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        feedback_card_vbox.addWidget(feedback_scroll)
        main_layout.addWidget(feedback_card)

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
        self.report_dist.setText(str(data.get('distraction_count', 0)))
        self.report_longest_focus.setText("--")
        self.report_avg_focus.setText("--")
        self._session_start_dt = None
        self._session_end_dt = None

        llm_comment = (data.get('llm_comment') or "").strip()
        if llm_comment:
            self.report_llm_feedback_label.setText(llm_comment)
        else:
            self.report_llm_feedback_label.setText("저장된 LLM 피드백이 없습니다.")
        # 스크롤 영역의 실제 너비에 맞춰 라벨 크기 조정
        QTimer.singleShot(50, lambda: self.report_llm_feedback_label.adjustSize())

        try:
            start = datetime.fromisoformat(data['start_time'].replace('Z', ''))
            end = datetime.fromisoformat(data['end_time'].replace('Z', ''))
            self._session_start_dt = start
            self._session_end_dt = end
            duration = end - start
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds() % 60)
            self.report_duration.setText(f"{minutes:02d}:{seconds:02d}")
        except Exception:
            self.report_duration.setText("00:00")
            self._session_start_dt = None
            self._session_end_dt = None
        
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
        sleepy_timestamps = []
        
        for log in logs:
            try:
                ts = datetime.fromisoformat(log['timestamp'].replace('Z', ''))
                timestamps.append(ts)
                is_distracted_list.append(1 if log['is_distracted'] else 0)
                raw_emotion = log.get('emotion_data')
                if isinstance(raw_emotion, str):
                    try:
                        emotion_data = json.loads(raw_emotion) or {}
                    except (json.JSONDecodeError, TypeError):
                        emotion_data = {}
                else:
                    emotion_data = raw_emotion or {}
                emotion = (emotion_data.get('emotion') or '').strip().lower()
                if emotion == 'sleepy':
                    sleepy_timestamps.append(ts)
            except Exception:
                continue
        
        if not timestamps:
            return
        
        # 시작 시간 기준으로 상대 시간 계산 (초 단위)
        start_time = timestamps[0]
        relative_times = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # 연속 집중 구간 길이 계산 (최장/평균 집중시간용)
        focus_segment_durations = []
        current_segment = 0.0
        for i in range(len(relative_times) - 1):
            time_diff = relative_times[i + 1] - relative_times[i]
            if is_distracted_list[i]:  # 비집중
                if current_segment > 0:
                    focus_segment_durations.append(current_segment)
                    current_segment = 0.0
            else:  # 집중
                current_segment += time_diff
        if len(relative_times) > 0 and not is_distracted_list[-1]:
            current_segment += 3.0  # 마지막 로그 이후 3초 (bar chart와 동일)
        if current_segment > 0:
            focus_segment_durations.append(current_segment)
        
        def _format_duration(seconds: float) -> str:
            if seconds <= 0:
                return "0초"
            if seconds >= 60:
                m = int(seconds // 60)
                s = int(seconds % 60)
                return f"{m}분 {s}초"
            return f"{int(seconds)}초"
        
        if focus_segment_durations:
            longest_sec = max(focus_segment_durations)
            avg_sec = sum(focus_segment_durations) / len(focus_segment_durations)
            self.report_longest_focus.setText(_format_duration(longest_sec))
            self.report_avg_focus.setText(_format_duration(avg_sec))
        else:
            self.report_longest_focus.setText("0초")
            self.report_avg_focus.setText("0초")
        
        # 1. 도넛 차트: 집중 시간 vs 비집중 시간
        focused_time = 0.0
        distracted_time = 0.0
        for i in range(len(relative_times) - 1):
            time_diff = relative_times[i + 1] - relative_times[i]
            if is_distracted_list[i]:
                distracted_time += time_diff
            else:
                focused_time += time_diff
        if len(relative_times) > 0:
            last_time_diff = 3.0
            if is_distracted_list[-1]:
                distracted_time += last_time_diff
            else:
                focused_time += last_time_diff
        self.donut_widget.setData(focused_time, distracted_time)

        # 2. 선 그래프 1: 시간에 따른 집중 여부
        self.draw_focus_status_chart(
            relative_times,
            is_distracted_list,
            getattr(self, "_session_start_dt", None),
            getattr(self, "_session_end_dt", None),
        )
        
        # 3. 막대 그래프: 10분 간격 Sleepy 감지 횟수 (x축 = 세션 start_time ~ end_time)
        self.draw_sleepy_count_chart(
            getattr(self, "_session_start_dt", None),
            getattr(self, "_session_end_dt", None),
            sleepy_timestamps,
        )

    def draw_focus_status_chart(
        self,
        relative_times,
        is_distracted_list,
        session_start_dt=None,
        session_end_dt=None,
    ):
        """선 그래프 1: 시간에 따른 집중 여부. X축 min=start_time, max=end_time(세션 구간)."""
        self.line1_plot.clear()
        
        # 집중 여부를 0(비집중) 또는 1(집중)로 표시
        focus_values = [1 if not dist else 0 for dist in is_distracted_list]
        
        pen = pg.mkPen(color='#BB86FC', width=2)
        self.line1_plot.plot(relative_times, focus_values, pen=pen)
        
        if session_start_dt is not None and session_end_dt is not None:
            duration_seconds = (session_end_dt - session_start_dt).total_seconds()
            self.line1_plot.setXRange(0, duration_seconds)
            # 서버 시각(UTC 또는 KST 등)을 로컬 시각으로 변환 후 X축 눈금 표시
            def _to_local(dt):
                if dt.tzinfo is not None:
                    return dt.astimezone()
                return dt.replace(tzinfo=timezone.utc).astimezone()
            start_local = _to_local(session_start_dt)
            end_local = _to_local(session_end_dt)
            d = duration_seconds
            tick_positions = [0, d / 4, d / 2, 3 * d / 4, d]
            tick_labels = [
                start_local.strftime("%H:%M"),
                _to_local(session_start_dt + timedelta(seconds=d / 4)).strftime("%H:%M"),
                _to_local(session_start_dt + timedelta(seconds=d / 2)).strftime("%H:%M"),
                _to_local(session_start_dt + timedelta(seconds=3 * d / 4)).strftime("%H:%M"),
                end_local.strftime("%H:%M"),
            ]
            ticks = [(pos, label) for pos, label in zip(tick_positions, tick_labels)]
            self.line1_plot.getAxis("bottom").setTicks([ticks])
        else:
            self.line1_plot.setXRange(
                min(relative_times) if relative_times else 0,
                max(relative_times) if relative_times else 1,
            )

    def draw_sleepy_count_chart(self, session_start_dt, session_end_dt, sleepy_timestamps):
        """10분 간격 Sleepy 감지 횟수 막대 그래프. X축 min~max = 세션 start_time ~ end_time(오른쪽 위 그래프와 동일)."""
        self.line2_plot.clear()
        
        if session_start_dt is None or session_end_dt is None:
            return
        duration_seconds = (session_end_dt - session_start_dt).total_seconds()
        if duration_seconds <= 0:
            return
        
        interval_seconds = 600  # 10분
        num_buckets = max(1, int((duration_seconds + interval_seconds - 1) // interval_seconds))
        
        # 각 10분 구간별 sleepy 횟수 (세션 start 기준)
        counts = [0] * num_buckets
        for ts in sleepy_timestamps:
            try:
                sec = (ts - session_start_dt).total_seconds()
            except (TypeError, OverflowError):
                try:
                    sec = ts.timestamp() - session_start_dt.timestamp()
                except (TypeError, OverflowError, OSError):
                    continue
            if 0 <= sec < duration_seconds:
                bucket = min(int(sec // interval_seconds), num_buckets - 1)
                counts[bucket] += 1
        
        # 막대: x = 구간 중심(초), 높이 = 횟수, 너비 = 10분보다 약간 작게
        x_centers = [i * interval_seconds + interval_seconds / 2.0 for i in range(num_buckets)]
        bar_width = interval_seconds * 0.85
        bg = pg.BarGraphItem(x=x_centers, height=counts, width=bar_width, brush='#CF6679')
        self.line2_plot.addItem(bg)
        self.line2_plot.setXRange(0, duration_seconds)
        self.line2_plot.setYRange(0, (max(counts) + 1) if counts else 1)
        
        # X축 눈금: 오른쪽 위 그래프처럼 start_time ~ end_time HH:MM
        def _to_local(dt):
            if dt is None:
                return None
            if getattr(dt, 'tzinfo', None) is not None:
                return dt.astimezone()
            return dt.replace(tzinfo=timezone.utc).astimezone()
        start_local = _to_local(session_start_dt)
        end_local = _to_local(session_end_dt)
        d = duration_seconds
        tick_positions = [0, d / 4, d / 2, 3 * d / 4, d]
        tick_labels = [
            start_local.strftime("%H:%M"),
            _to_local(session_start_dt + timedelta(seconds=d / 4)).strftime("%H:%M"),
            _to_local(session_start_dt + timedelta(seconds=d / 2)).strftime("%H:%M"),
            _to_local(session_start_dt + timedelta(seconds=3 * d / 4)).strftime("%H:%M"),
            end_local.strftime("%H:%M"),
        ]
        ticks = [(pos, label) for pos, label in zip(tick_positions, tick_labels)]
        self.line2_plot.getAxis("bottom").setTicks([ticks])

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

        # Feedback Area (scrollable when content is long)
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
        self.feedback_text.setMinimumWidth(600)
        feedback_scroll = QScrollArea()
        feedback_scroll.setWidgetResizable(True)
        feedback_scroll.setWidget(self.feedback_text)
        feedback_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feedback_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        feedback_scroll.setMinimumHeight(280)
        feedback_scroll.setMaximumHeight(400)
        card_layout.addWidget(feedback_scroll)

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

        # Date filter row
        self._all_sessions = []
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("From:"))
        self.from_date_edit = QDateEdit()
        self.from_date_edit.setCalendarPopup(True)
        self.from_date_edit.setDisplayFormat("yyyy-MM-dd")
        today = date.today()
        self.from_date_edit.setDate(QDate(today.year, today.month, today.day))
        self.from_date_edit.setStyleSheet("background-color: #2D2D2D; color: #E0E0E0; padding: 6px; border-radius: 6px;")
        filter_row.addWidget(self.from_date_edit)
        filter_row.addWidget(QLabel("To:"))
        self.to_date_edit = QDateEdit()
        self.to_date_edit.setCalendarPopup(True)
        self.to_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.to_date_edit.setDate(QDate(2030, 12, 31))
        self.to_date_edit.setStyleSheet("background-color: #2D2D2D; color: #E0E0E0; padding: 6px; border-radius: 6px;")
        filter_row.addWidget(self.to_date_edit)
        search_btn = QPushButton("Search")
        search_btn.setObjectName("SecondaryBtn")
        search_btn.clicked.connect(self._apply_filter)
        filter_row.addWidget(search_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("SecondaryBtn")
        def clear_filter():
            self.from_date_edit.setDate(QDate(2000, 1, 1))
            self.to_date_edit.setDate(QDate(2030, 12, 31))
            self._apply_filter()
        clear_btn.clicked.connect(clear_filter)
        filter_row.addWidget(clear_btn)
        filter_row.addStretch()
        layout.addLayout(filter_row)

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

    def _parse_dt_local(self, dt_str):
        """Parse server datetime (UTC or timezone-aware) and return naive datetime in user's local time for display."""
        if not dt_str:
            return None
        s = dt_str.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        else:
            dt = dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
        return dt

    def load_data(self):
        self._all_sessions = self.network_client.get_history()
        self._apply_filter()

    def _apply_filter(self):
        from_q = self.from_date_edit.date()
        to_q = self.to_date_edit.date()
        from_date = date(from_q.year(), from_q.month(), from_q.day())
        to_date = date(to_q.year(), to_q.month(), to_q.day())

        filtered = []
        for session in self._all_sessions:
            start_time = self._parse_dt_local(session.get('start_time'))
            if start_time is None:
                filtered.append(session)
                continue
            session_date = start_time.date()
            if from_date <= session_date <= to_date:
                filtered.append(session)

        self.table.setRowCount(len(filtered))
        for i, session in enumerate(filtered):
            start_time = self._parse_dt_local(session.get('start_time'))
            if start_time is not None:
                self.table.setItem(i, 0, QTableWidgetItem(start_time.strftime("%Y-%m-%d %H:%M")))
            else:
                self.table.setItem(i, 0, QTableWidgetItem("-"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{int(session['focus_ratio'])}%"))
            self.table.setItem(i, 2, QTableWidgetItem(str(session['distraction_count'])))
            try:
                end_time = self._parse_dt_local(session.get('end_time'))
                if start_time is not None and end_time is not None:
                    duration = end_time - start_time
                    minutes = int(duration.total_seconds() // 60)
                    seconds = int(duration.total_seconds() % 60)
                    self.table.setItem(i, 3, QTableWidgetItem(f"{minutes:02d}:{seconds:02d}"))
                else:
                    self.table.setItem(i, 3, QTableWidgetItem("-"))
            except Exception:
                self.table.setItem(i, 3, QTableWidgetItem("-"))
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
        self.head_calibration_page = HeadPoseCalibrationPage(self.camera, self.network_client)
        self.monitoring_page = MonitoringPage()
        self.report_page = ReportPage()
        self.report_page.set_network_client(self.network_client)
        self.analysis_page = LlmAnalysisPage(self.network_client)
        self.history_page = HistoryPage(self.network_client)

        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.calibration_page)
        self.stack.addWidget(self.head_calibration_page)
        self.stack.addWidget(self.monitoring_page)
        self.stack.addWidget(self.report_page)
        self.stack.addWidget(self.analysis_page)
        self.stack.addWidget(self.history_page)

        # Signals
        self.main_page.start_requested.connect(self.start_session)
        self.main_page.history_requested.connect(self.show_history)
        self.main_page.calibration_requested.connect(lambda: self.stack.setCurrentWidget(self.calibration_page))
        self.main_page.head_calibration_requested.connect(lambda: self.stack.setCurrentWidget(self.head_calibration_page))
        self.calibration_page.done_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))
        self.head_calibration_page.done_requested.connect(lambda: self.stack.setCurrentWidget(self.main_page))
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
        # 과거 세션: llm_comment가 없을 때만 "LLM 분석 시작" 버튼 표시 (나중에 분석 요청 가능)
        has_llm = bool((session_data.get('llm_comment') or "").strip())
        self.report_page.set_llm_button_visible(not has_llm)
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
