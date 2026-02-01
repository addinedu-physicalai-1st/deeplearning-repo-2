import sys
import cv2
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from client.core.camera import Camera
from client.core.network import NetworkClient

class InferenceThread(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, network_client, image_base64):
        super().__init__()
        self.network_client = network_client
        self.image_base64 = image_base64

    def run(self):
        result = self.network_client.send_inference_request(self.image_base64)
        self.result_ready.emit(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Work Focus Monitor")
        self.resize(800, 600)

        self.camera = Camera()
        self.network_client = NetworkClient()
        self.is_monitoring = False
        
        # UI Setup
        self.init_ui()

        # Timers
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(33)  # ~30 FPS

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.request_inference)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Video Preview
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.toggle_monitoring)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)

        # Distraction Warning Area (Hidden initially)
        self.warning_image_label = QLabel()
        self.warning_image_label.setFixedSize(320, 240)
        self.warning_image_label.setStyleSheet("border: 2px solid red;")
        self.warning_image_label.hide()
        layout.addWidget(self.warning_image_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def update_preview(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # Convert to RGB for Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.width(), self.video_label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio))
            self.current_frame = frame

    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.start_button.setText("Stop Monitoring")
            self.status_label.setText("Status: Monitoring...")
            self.inference_timer.start(3000)  # Every 3 seconds
        else:
            self.is_monitoring = False
            self.start_button.setText("Start Monitoring")
            self.status_label.setText("Status: Idle")
            self.inference_timer.stop()
            self.warning_image_label.hide()

    def request_inference(self):
        if hasattr(self, 'current_frame'):
            image_base64 = self.camera.frame_to_base64(self.current_frame)
            self.inference_thread = InferenceThread(self.network_client, image_base64)
            self.inference_thread.result_ready.connect(self.handle_inference_result)
            # Save the frame that was sent for warning display
            self.sent_frame = self.current_frame.copy()
            self.inference_thread.start()

    def handle_inference_result(self, result):
        if result.is_distracted:
            self.status_label.setText(f"Status: DISTRACTED ({result.status_message})")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            self.show_distraction_warning()
        else:
            self.status_label.setText(f"Status: FOCUSED")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
            self.warning_image_label.hide()

    def show_distraction_warning(self):
        if hasattr(self, 'sent_frame'):
            frame_rgb = cv2.cvtColor(self.sent_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.warning_image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.warning_image_label.width(), self.warning_image_label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio))
            self.warning_image_label.show()
            # In a real app, you'd play a sound here as well.
            print("WARNING: Distraction detected!")

    def closeEvent(self, event):
        self.camera.release()
        event.accept()
