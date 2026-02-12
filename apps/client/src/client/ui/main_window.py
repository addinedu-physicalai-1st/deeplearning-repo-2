import sys
import json
import cv2
import time
import logging
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QGridLayout, QStackedWidget, QMessageBox, QApplication, QProgressBar, QDateEdit, QScrollArea, QLineEdit, QSizePolicy, QGraphicsOpacityEffect, QGraphicsDropShadowEffect
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect, QRectF, QEasingCurve, QDate, QElapsedTimer, QEvent, QObject
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen, QBrush, QPainterPath, QPalette, QLinearGradient
import pyqtgraph as pg
pg.setConfigOption('foreground', (240, 240, 245, 140))
pg.setConfigOption('background', '#0d0d12')
from datetime import datetime, timedelta, timezone, date

from client.core.camera import Camera
from client.core.network import NetworkClient
from client.core.posture import PostureMonitor

# Logging setup
logger = logging.getLogger(__name__)

# --- 공통 스타일 정의 (Glassmorphism Dark) ---
STYLE_SHEET = """
    QMainWindow { background-color: #0d0d12; }
    QWidget {
        color: #f0f0f5;
        font-family: 'Inter', 'SF Pro Display', 'Ubuntu', 'Segoe UI', sans-serif;
    }
    QFrame#Card {
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    QLabel#Title {
        font-size: 26px; font-weight: bold; color: #f0f0f5;
        margin-bottom: 12px; letter-spacing: 0.5px;
    }
    QLabel#Header {
        font-size: 17px; color: rgba(240, 240, 245, 0.55);
        font-weight: 500; letter-spacing: 0.3px;
    }
    QLabel#StatValue { font-size: 34px; font-weight: bold; color: #22d3ee; }
    QLabel#StatLabel {
        font-size: 11px; color: rgba(240, 240, 245, 0.45);
        text-transform: uppercase; font-weight: bold; letter-spacing: 1.5px;
    }
    QPushButton#PrimaryBtn {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
        color: #ffffff; font-weight: 600;
        border-radius: 12px; padding: 14px 28px; font-size: 15px; border: none;
    }
    QPushButton#PrimaryBtn:hover {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c7ff7, stop:1 #9d75f8);
    }
    QPushButton#PrimaryBtn:pressed {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5558e0, stop:1 #7c4fe6);
    }
    QPushButton#PrimaryBtn:disabled {
        background-color: rgba(255, 255, 255, 0.08);
        color: rgba(240, 240, 245, 0.25);
    }
    QPushButton#SecondaryBtn {
        background-color: rgba(255, 255, 255, 0.06);
        color: #a78bfa; font-weight: 600;
        border: 1px solid rgba(167, 139, 250, 0.30);
        border-radius: 10px; padding: 10px 20px;
    }
    QPushButton#SecondaryBtn:hover {
        background-color: rgba(167, 139, 250, 0.12);
        border: 1px solid rgba(167, 139, 250, 0.50);
    }
    QPushButton#SecondaryBtn:pressed {
        background-color: rgba(167, 139, 250, 0.20);
    }
    QPushButton#StopBtn {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ef4444, stop:1 #f87171);
        color: #fff; font-weight: 600;
        border-radius: 10px; padding: 12px; font-size: 15px; border: none;
    }
    QPushButton#StopBtn:hover {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f87171, stop:1 #fca5a5);
    }
    /* --- QMessageBox Glassmorphism --- */
    QMessageBox {
        background-color: #16161e;
    }
    QMessageBox QLabel {
        color: #f0f0f5;
        font-size: 14px;
        line-height: 1.5;
        min-width: 320px;
    }
    QMessageBox QPushButton {
        background-color: rgba(255, 255, 255, 0.08);
        color: #f0f0f5;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 8px 24px;
        font-size: 13px;
        font-weight: 600;
        min-width: 100px;
    }
    QMessageBox QPushButton:hover {
        background-color: rgba(139, 92, 246, 0.20);
        border: 1px solid rgba(139, 92, 246, 0.50);
        color: #c4b5fd;
    }
    QMessageBox QPushButton:pressed {
        background-color: rgba(139, 92, 246, 0.35);
    }
    QMessageBox QPushButton:default {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
        color: #ffffff;
        border: none;
    }
    QMessageBox QPushButton:default:hover {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c7ff7, stop:1 #9d75f8);
    }
"""

INPUT_STYLE = """
    QLineEdit {
        background-color: rgba(255, 255, 255, 0.07);
        color: #f0f0f5;
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 10px; padding: 12px 16px; font-size: 14px;
        selection-background-color: rgba(99, 102, 241, 0.4);
    }
    QLineEdit:focus { border: 1px solid rgba(139, 92, 246, 0.60); }
    QLineEdit::placeholder { color: rgba(240, 240, 245, 0.30); }
"""

SIDEBAR_STYLE = """
    QWidget#Sidebar {
        background-color: rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
"""

SIDEBAR_BTN_STYLE = """
    QPushButton {
        background-color: transparent;
        color: rgba(240, 240, 245, 0.50);
        border: none;
        border-radius: 10px;
        padding: 11px 16px;
        font-size: 13px;
        font-weight: 500;
        text-align: left;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 0.06);
        color: #f0f0f5;
    }
"""

SIDEBAR_BTN_ACTIVE_STYLE = """
    QPushButton {
        background-color: rgba(99, 102, 241, 0.15);
        color: #a78bfa;
        border: none;
        border-left: 3px solid #8b5cf6;
        border-radius: 10px;
        padding: 11px 13px 11px 16px;
        font-size: 13px;
        font-weight: 600;
        text-align: left;
    }
"""

class ButtonAnimationFilter(QObject):
    """QPushButton에 press/release opacity 애니메이션을 부여하는 이벤트 필터."""
    PRESS_DURATION = 100
    RELEASE_DURATION = 150
    PRESS_OPACITY = 0.7

    def __init__(self, parent=None):
        super().__init__(parent)
        self._animations = {}

    @staticmethod
    def install_on_all(root_widget):
        for btn in root_widget.findChildren(QPushButton):
            f = ButtonAnimationFilter(btn)
            btn.installEventFilter(f)

    def eventFilter(self, obj, event):
        if not isinstance(obj, QPushButton) or not obj.isEnabled():
            return False
        etype = event.type()
        if etype == QEvent.Type.MouseButtonPress:
            self._animate_press(obj)
        elif etype == QEvent.Type.MouseButtonRelease:
            self._animate_release(obj)
        return False

    def _animate_press(self, btn):
        if btn in self._animations:
            self._animations[btn].stop()
        effect = btn.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(btn)
            effect.setOpacity(1.0)
            btn.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(self.PRESS_DURATION)
        anim.setStartValue(effect.opacity())
        anim.setEndValue(self.PRESS_OPACITY)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animations[btn] = anim
        anim.start()

    def _animate_release(self, btn):
        if btn in self._animations:
            self._animations[btn].stop()
        effect = btn.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            return
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(self.RELEASE_DURATION)
        anim.setStartValue(effect.opacity())
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: btn.setGraphicsEffect(None))
        self._animations[btn] = anim
        anim.start()


def apply_glass_shadow(widget, blur=30, y=4, color=QColor(0, 0, 0, 80)):
    """카드에 글래스모피즘 드롭 쉐도우를 적용."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setXOffset(0)
    shadow.setYOffset(y)
    shadow.setColor(color)
    widget.setGraphicsEffect(shadow)


class FocusMonitorIcon(QWidget):
    """모니터 + 포커스 컨셉의 커스텀 아이콘 위젯."""

    def __init__(self, size=96, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._size = size

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._size

        # 배경: 보라색 그라데이션 둥근 사각형
        bg_grad = QLinearGradient(0, 0, s, s)
        bg_grad.setColorAt(0, QColor("#6366f1"))
        bg_grad.setColorAt(1, QColor("#8b5cf6"))
        p.setBrush(QBrush(bg_grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, s, s, s * 0.29, s * 0.29)

        # 모니터 본체 (둥근 사각형)
        mon_w = s * 0.58
        mon_h = s * 0.40
        mon_x = (s - mon_w) / 2
        mon_y = s * 0.18
        pen = QPen(QColor("#ffffff"), s * 0.025)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(255, 255, 255, 18)))
        p.drawRoundedRect(QRectF(mon_x, mon_y, mon_w, mon_h), s * 0.04, s * 0.04)

        # 모니터 스탠드 (세로 막대 + 받침)
        stand_cx = s / 2
        stand_top = mon_y + mon_h
        stand_bot = stand_top + s * 0.10
        p.setPen(QPen(QColor("#ffffff"), s * 0.025, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(int(stand_cx), int(stand_top), int(stand_cx), int(stand_bot))
        base_w = s * 0.22
        p.drawLine(int(stand_cx - base_w / 2), int(stand_bot), int(stand_cx + base_w / 2), int(stand_bot))

        # 화면 안 포커스 타겟 (동심원 + 십자선)
        cx = s / 2
        cy = mon_y + mon_h / 2
        r_outer = min(mon_w, mon_h) * 0.32
        r_inner = r_outer * 0.45

        target_pen = QPen(QColor("#ffffff"), s * 0.02)
        p.setPen(target_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(cx - r_outer, cy - r_outer, r_outer * 2, r_outer * 2))
        p.drawEllipse(QRectF(cx - r_inner, cy - r_inner, r_inner * 2, r_inner * 2))

        # 십자선
        cross_len = r_outer * 1.3
        p.drawLine(int(cx - cross_len), int(cy), int(cx + cross_len), int(cy))
        p.drawLine(int(cx), int(cy - cross_len), int(cx), int(cy + cross_len))

        # 중심점
        p.setBrush(QBrush(QColor("#ffffff")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(cx - s * 0.02, cy - s * 0.02, s * 0.04, s * 0.04))

        p.end()


class SplashPage(QWidget):
    """앱 시작 시 프로그램명 + 팀명을 보여주는 스플래시 화면."""
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._shown = False
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 아이콘
        icon_widget = FocusMonitorIcon(96)
        layout.addWidget(icon_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(32)

        # 프로그램명
        title = QLabel("FocusMonitor AI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #f0f0f5; letter-spacing: 2px;")
        layout.addWidget(title)
        layout.addSpacing(12)

        # 팀명
        team = QLabel("가산 집중력 연구소")
        team.setAlignment(Qt.AlignmentFlag.AlignCenter)
        team.setStyleSheet("font-size: 15px; color: rgba(240, 240, 245, 0.50); letter-spacing: 1px;")
        layout.addWidget(team)

    def showEvent(self, event):
        super().showEvent(event)
        if self._shown:
            return
        self._shown = True
        # 페이드인 애니메이션
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._anim.setDuration(600)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()
        # 2초 후 전환
        QTimer.singleShot(2500, self._on_finish)

    def _on_finish(self):
        self.setGraphicsEffect(None)
        self.finished.emit()


class LoginPage(QWidget):
    login_success = pyqtSignal(dict)
    register_requested = pyqtSignal()

    def __init__(self, network_client):
        super().__init__()
        self.network_client = network_client
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        apply_glass_shadow(card)
        card.setFixedSize(420, 480)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(16)
        card_layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("FOCUS MONITOR")
        title.setObjectName("Title")
        card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("로그인하여 시작하세요")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: rgba(240, 240, 245, 0.45);")
        card_layout.addWidget(subtitle)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("아이디")
        self.username_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("비밀번호")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.password_input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #f87171; font-size: 12px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()
        card_layout.addWidget(self.error_label)

        login_btn = QPushButton("LOGIN")
        login_btn.setObjectName("PrimaryBtn")
        login_btn.clicked.connect(self._on_login)
        card_layout.addWidget(login_btn)

        register_btn = QPushButton("CREATE ACCOUNT")
        register_btn.setObjectName("SecondaryBtn")
        register_btn.clicked.connect(self.register_requested.emit)
        card_layout.addWidget(register_btn)

        layout.addWidget(card)

        self.password_input.returnPressed.connect(self._on_login)

    def _on_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if not username or not password:
            self.error_label.setText("아이디와 비밀번호를 입력해주세요.")
            self.error_label.show()
            return
        result = self.network_client.login(username, password)
        if result and "error" not in result:
            self.error_label.hide()
            self.login_success.emit(result)
        elif result and result.get("error") == "invalid_credentials":
            self.error_label.setText(result.get("detail", "아이디 또는 비밀번호가 올바르지 않습니다."))
            self.error_label.show()
        else:
            self.error_label.setText("서버 연결 오류")
            self.error_label.show()

    def clear_fields(self):
        self.username_input.clear()
        self.password_input.clear()
        self.error_label.hide()


class RegisterPage(QWidget):
    register_success = pyqtSignal(dict)
    back_to_login = pyqtSignal()

    def __init__(self, network_client):
        super().__init__()
        self.network_client = network_client
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        apply_glass_shadow(card)
        card.setFixedSize(420, 560)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(14)
        card_layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("회원가입")
        title.setObjectName("Title")
        card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("아이디 (3자 이상)")
        self.username_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.username_input)

        self.display_name_input = QLineEdit()
        self.display_name_input.setPlaceholderText("이름")
        self.display_name_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.display_name_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("비밀번호 (6자 이상)")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.password_input)

        self.password_confirm_input = QLineEdit()
        self.password_confirm_input.setPlaceholderText("비밀번호 확인")
        self.password_confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_confirm_input.setStyleSheet(INPUT_STYLE)
        card_layout.addWidget(self.password_confirm_input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #f87171; font-size: 12px;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        card_layout.addWidget(self.error_label)

        register_btn = QPushButton("REGISTER")
        register_btn.setObjectName("PrimaryBtn")
        register_btn.clicked.connect(self._on_register)
        card_layout.addWidget(register_btn)

        back_btn = QPushButton("BACK TO LOGIN")
        back_btn.setObjectName("SecondaryBtn")
        back_btn.clicked.connect(self.back_to_login.emit)
        card_layout.addWidget(back_btn)

        layout.addWidget(card)

    def _on_register(self):
        username = self.username_input.text().strip()
        display_name = self.display_name_input.text().strip()
        password = self.password_input.text()
        confirm = self.password_confirm_input.text()

        if not username or not display_name or not password:
            self.error_label.setText("모든 항목을 입력해주세요.")
            self.error_label.show()
            return
        if len(username) < 3:
            self.error_label.setText("아이디는 3자 이상이어야 합니다.")
            self.error_label.show()
            return
        if len(password) < 6:
            self.error_label.setText("비밀번호는 6자 이상이어야 합니다.")
            self.error_label.show()
            return
        if password != confirm:
            self.error_label.setText("비밀번호가 일치하지 않습니다.")
            self.error_label.show()
            return

        result = self.network_client.register(username, password, display_name)
        if result and "error" not in result:
            self.error_label.hide()
            self.register_success.emit(result)
        elif result and result.get("error") == "duplicate":
            self.error_label.setText(result.get("detail", "이미 사용 중인 아이디입니다."))
            self.error_label.show()
        else:
            self.error_label.setText("서버 연결 오류")
            self.error_label.show()

    def clear_fields(self):
        self.username_input.clear()
        self.display_name_input.clear()
        self.password_input.clear()
        self.password_confirm_input.clear()
        self.error_label.hide()


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
    """프로그램 시작 메인 화면 (GNOME-inspired)"""
    start_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._server_connected = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        content = QVBoxLayout()
        content.setSpacing(0)
        content.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Hero icon
        icon_label = QLabel("\u25C9")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("""
            font-size: 48px; color: #ffffff;
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
            border-radius: 20px;
            padding: 12px;
        """)
        icon_label.setFixedSize(80, 80)
        content.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
        content.addSpacing(24)

        # Title
        title = QLabel("집중할 준비가 되셨나요?")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #f0f0f5;")
        content.addWidget(title)
        content.addSpacing(8)

        desc = QLabel("AI 시스템이 초기화되었습니다.\n모니터링을 시작하려면 아래 버튼을 누르세요.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("font-size: 14px; color: rgba(240, 240, 245, 0.45);")
        desc.setFixedWidth(400)
        desc.setMinimumHeight(70)
        content.addWidget(desc, alignment=Qt.AlignmentFlag.AlignCenter)
        content.addSpacing(32)

        # Start button
        self.start_btn = QPushButton("  \u25B6  모니터링 시작")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumWidth(320)
        self.start_btn.setMinimumHeight(52)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
                color: #ffffff; font-weight: bold;
                border-radius: 12px; padding: 14px 32px; font-size: 17px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #818cf8, stop:1 #a78bfa);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
            }
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 0.06); color: rgba(240, 240, 245, 0.25);
            }
        """)
        self.start_btn.clicked.connect(self.start_requested.emit)
        content.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        content.addSpacing(24)

        # Status cards row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        # Server status card
        self.server_card = QFrame()
        self.server_card.setObjectName("Card")
        apply_glass_shadow(self.server_card, blur=20, y=2, color=QColor(0, 0, 0, 50))
        self.server_card.setFixedSize(200, 90)
        sc_layout = QVBoxLayout(self.server_card)
        sc_layout.setContentsMargins(16, 12, 16, 12)
        sc_label = QLabel("서버 상태")
        sc_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #f59e0b; text-transform: uppercase;")
        sc_layout.addWidget(sc_label)
        self.server_status_row = QHBoxLayout()
        self.server_dot = QLabel("\u25CF")
        self.server_dot.setStyleSheet("font-size: 10px; color: #ef4444;")
        self.server_dot.setFixedWidth(14)
        self.server_status_text = QLabel("연결 대기")
        self.server_status_text.setStyleSheet("font-size: 16px; font-weight: bold; color: #f0f0f5;")
        self.server_status_row.addWidget(self.server_dot)
        self.server_status_row.addWidget(self.server_status_text)
        self.server_status_row.addStretch()
        sc_layout.addLayout(self.server_status_row)
        cards_row.addWidget(self.server_card)

        # AI engine card
        ai_card = QFrame()
        ai_card.setObjectName("Card")
        apply_glass_shadow(ai_card, blur=20, y=2, color=QColor(0, 0, 0, 50))
        ai_card.setFixedSize(200, 90)
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(16, 12, 16, 12)
        ai_label = QLabel("AI 엔진")
        ai_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #a78bfa; text-transform: uppercase;")
        ai_layout.addWidget(ai_label)
        ai_status_row = QHBoxLayout()
        ai_dot = QLabel("\u25CF")
        ai_dot.setStyleSheet("font-size: 10px; color: #10b981;")
        ai_dot.setFixedWidth(14)
        ai_text = QLabel("온라인")
        ai_text.setStyleSheet("font-size: 16px; font-weight: bold; color: #f0f0f5;")
        ai_status_row.addWidget(ai_dot)
        ai_status_row.addWidget(ai_text)
        ai_status_row.addStretch()
        ai_layout.addLayout(ai_status_row)
        cards_row.addWidget(ai_card)

        content.addLayout(cards_row)
        layout.addLayout(content)

    def set_connection_status(self, connected: bool):
        self._server_connected = connected
        if connected:
            self.server_dot.setStyleSheet("font-size: 10px; color: #10b981;")
            self.server_status_text.setText("온라인")
            self.start_btn.setEnabled(True)
        else:
            self.server_dot.setStyleSheet("font-size: 10px; color: #ef4444;")
            self.server_status_text.setText("연결 대기")
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
        self.video_label.setStyleSheet("background-color: #0a0a0f; border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 8px;")
        self.video_label.setFixedSize(640, 480)
        left.addWidget(self.video_label)

        self.status_label = QLabel("상태: 대기중")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #fbbf24;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.status_label)
        layout.addLayout(left)

        # Control panel right
        panel = QFrame()
        panel.setObjectName("Card")
        panel.setStyleSheet("QFrame#Card { background-color: rgba(255, 255, 255, 0.04); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.10); }")
        apply_glass_shadow(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(12)

        title = QLabel("거리 캘리브레이션")
        title.setObjectName("Title")
        title.setStyleSheet("font-size: 22px; color: #c084fc;")
        panel_layout.addWidget(title)

        self.shoulder_label = QLabel("어깨 각도: --°")
        self.shoulder_label.setStyleSheet("font-size: 16px; color: rgba(240, 240, 245, 0.45);")
        panel_layout.addWidget(self.shoulder_label)

        self.distance_label = QLabel("거리: -- cm")
        self.distance_label.setStyleSheet("font-size: 16px; color: #22d3ee;")
        panel_layout.addWidget(self.distance_label)

        self.posture_label = QLabel("거북목: -- %")
        self.posture_label.setStyleSheet("font-size: 16px; color: #60a5fa;")
        panel_layout.addWidget(self.posture_label)

        self.posture_progress = QProgressBar()
        self.posture_progress.setMinimum(0)
        self.posture_progress.setMaximum(100)
        self.posture_progress.setValue(0)
        self.posture_progress.setStyleSheet("""
            QProgressBar { border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 5px; background-color: rgba(255, 255, 255, 0.06); }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #60a5fa); border-radius: 4px; }
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

        # 화면 표시용으로만 좌우 반전 (거울처럼 보이도록)
        display_image = cv2.flip(image_rgb, 1)

        h, w, ch = display_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if shoulder_angle is not None:
            self.shoulder_label.setText(f"어깨 각도: {shoulder_angle:.1f}°")
        else:
            self.shoulder_label.setText("어깨 각도: --°")

        # 정자세 설정 후: 이동거리 0 기준, 가까우면 음수(앞), 멀면 양수(뒤)
        if distance_offset_cm is not None:
            if distance_offset_cm > 0:
                self.distance_label.setText(f"이동거리: +{distance_offset_cm:.1f} cm (뒤)")
            elif distance_offset_cm < 0:
                self.distance_label.setText(f"이동거리: {distance_offset_cm:.1f} cm (앞)")
            else:
                self.distance_label.setText("이동거리: 0.0 cm")
        elif distance_cm is not None:
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
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f59e0b;")
            return
        self.monitor.set_baseline_from_distance(self.current_distance_cm)
        image_base64 = self.camera.frame_to_base64(self.current_frame)
        ok = self.network_client.set_baseline(image_base64)
        if ok:
            self.status_label.setText("정자세가 설정되었습니다.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #22d3ee;")
        else:
            self.status_label.setText("ai_body 서버 연결 실패. 서버를 확인하세요.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f87171;")


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
        self.video_label.setStyleSheet("background-color: #0a0a0f; border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 8px;")
        self.video_label.setFixedSize(640, 480)
        left.addWidget(self.video_label)
        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #fbbf24;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.status_label)
        layout.addLayout(left)

        panel = QFrame()
        panel.setObjectName("Card")
        panel.setStyleSheet("QFrame#Card { background-color: rgba(255, 255, 255, 0.04); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.10); }")
        apply_glass_shadow(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(12)
        title = QLabel("머리 각도 캘리브레이션")
        title.setObjectName("Title")
        title.setStyleSheet("font-size: 22px; color: #c084fc;")
        panel_layout.addWidget(title)
        self.step_label = QLabel("단계: 대기 중")
        self.step_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #22d3ee;")
        panel_layout.addWidget(self.step_label)
        self.pitch_label = QLabel("Pitch (위/아래): --°")
        self.pitch_label.setStyleSheet("font-size: 16px; color: rgba(240, 240, 245, 0.45);")
        panel_layout.addWidget(self.pitch_label)
        self.yaw_label = QLabel("Yaw (좌/우): --°")
        self.yaw_label.setStyleSheet("font-size: 16px; color: rgba(240, 240, 245, 0.45);")
        panel_layout.addWidget(self.yaw_label)
        self.roll_label = QLabel("Roll (기울기): --°")
        self.roll_label.setStyleSheet("font-size: 16px; color: rgba(240, 240, 245, 0.45);")
        panel_layout.addWidget(self.roll_label)
        panel_layout.addSpacing(10)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 5px; background-color: rgba(255, 255, 255, 0.06); } QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #22d3ee); border-radius: 4px; }")
        self.progress_bar.hide()
        panel_layout.addWidget(self.progress_bar)
        self.measured_label = QLabel("")
        self.measured_label.setStyleSheet("font-size: 14px; color: #60a5fa;")
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
        # 각 방향 측정이 끝날 때 효과음 재생
        QApplication.beep()

    def _on_next_step(self):
        if self.current_step < 4:
            self.current_step += 1
            self._update_ui()
        elif self.current_step == 4:
            self.current_step = 5
            self._update_ui()
            # 각 방향 측정이 끝날 때 효과음 재생
            QApplication.beep()

    def _update_ui(self):
        step_messages = {0: "준비 중...", 1: "1/4 단계: 위를 보세요", 2: "2/4 단계: 아래를 보세요", 3: "3/4 단계: 좌측을 보세요", 4: "4/4 단계: 우측을 보세요", 5: "측정 완료"}
        self.step_label.setText(f"단계: {step_messages.get(self.current_step, '')}")
        if self.current_step == 0:
            self.status_label.setText("측정을 시작하세요")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #fbbf24;")
            self.measure_btn.setText("측정 시작")
            self.measure_btn.setEnabled(True)
            self.next_step_btn.setEnabled(False)
            self.set_thresholds_btn.setEnabled(False)
        elif 1 <= self.current_step <= 4:
            self.status_label.setText(step_messages[self.current_step])
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #22d3ee;")
            self.measure_btn.setEnabled(not self.measuring)
            key = ["pitch_up", "pitch_down", "yaw_left", "yaw_right"][self.current_step - 1]
            self.next_step_btn.setEnabled(not self.measuring and self.measured_values.get(key) is not None)
        elif self.current_step == 5:
            self.status_label.setText("모든 측정이 완료되었습니다.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #22d3ee;")
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
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f59e0b;")
            return
        yaw_limit = max(abs(yaw_left), abs(yaw_right))
        ok = self.network_client.set_head_thresholds(pitch_up, pitch_down, yaw_limit)
        if ok:
            self.status_label.setText(f"임계값이 설정되었습니다.\nPitch: {pitch_up:.1f}° ~ {pitch_down:.1f}°\nYaw: ±{yaw_limit:.1f}°")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #22d3ee;")
        else:
            self.status_label.setText("ai_head 서버 연결 실패. 서버를 확인하세요.")
            self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f87171;")        

class GazeCalibrationCanvas(QWidget):
    """캘리브레이션 영역 전용 위젯: 검은 배경 + 빨간 목표점을 그려 가려지지 않게 함."""
    POINTS = [
        (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
        (0.05, 0.5),  (0.5, 0.5),  (0.95, 0.5),
        (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #0a0a0f;")
        self._point_index = 0
        self._click_count = 0

    def set_state(self, point_index: int, click_count: int):
        self._point_index = point_index
        self._click_count = click_count
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            painter.end()
            return
        if self._point_index >= len(self.POINTS):
            painter.end()
            return
        rx, ry = self.POINTS[self._point_index]
        x = int(w * rx)
        y = int(h * ry)
        painter.setPen(QPen(QColor("white"), 3))
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.drawEllipse(x - 15, y - 15, 30, 30)
        for i in range(self._click_count):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#22d3ee")))
            offset_x = x - 12 + i * 6
            painter.drawEllipse(offset_x, y + 20, 5, 5)
        painter.end()


class GazeCalibrationPage(QWidget):
    """9-point 시선 캘리브레이션 화면 (외부 WebGazer 코드의 개념을 PyQt로 구현)."""
    done_requested = pyqtSignal()

    POINTS = [
        (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
        (0.05, 0.5),  (0.5, 0.5),  (0.95, 0.5),
        (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
    ]
    CLICKS_PER_POINT = 5

    def __init__(self, camera: Camera, network_client: NetworkClient):
        super().__init__()
        self.camera = camera
        self.network_client = network_client

        self.calibration_data = []
        self.current_point_index = 0
        self.current_click_count = 0
        self._user_id = None
        self._completed = False

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_preview)

        # 로컬 MediaPipe Face Mesh (iris 추출용)
        import mediapipe as mp_lib
        self._mp_face_mesh = mp_lib.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 상단 안내 영역
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(20, 10, 20, 10)

        self.instruction_label = QLabel("빨간 점을 바라보며 클릭하세요")
        self.instruction_label.setStyleSheet("font-size: 18px; color: #c084fc; font-weight: bold;")
        top_bar.addWidget(self.instruction_label)
        top_bar.addStretch()

        self.progress_label = QLabel("포인트 1/9 - 클릭 0/5")
        self.progress_label.setStyleSheet("font-size: 16px; color: #22d3ee;")
        top_bar.addWidget(self.progress_label)

        layout.addLayout(top_bar)

        # 캘리브레이션 영역: 전용 캔버스에서 빨간 점을 그려 가려지지 않게 함
        self.calib_canvas = GazeCalibrationCanvas(self)
        layout.addWidget(self.calib_canvas, stretch=1)

        # 하단 상태/컨트롤 영역
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(20, 10, 20, 10)

        self.status_label = QLabel("준비됨")
        self.status_label.setStyleSheet("font-size: 14px; color: rgba(240, 240, 245, 0.45);")
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addStretch()

        self.video_label = QLabel()
        self.video_label.setFixedSize(160, 120)
        self.video_label.setStyleSheet("border: 1px solid rgba(255, 255, 255, 0.10); background-color: #0a0a0f; border-radius: 6px;")
        bottom_bar.addWidget(self.video_label)

        cancel_btn = QPushButton("취소")
        cancel_btn.setObjectName("SecondaryBtn")
        cancel_btn.clicked.connect(self.done_requested.emit)
        bottom_bar.addWidget(cancel_btn)

        layout.addLayout(bottom_bar)

    def showEvent(self, event):
        super().showEvent(event)
        self._reset_calibration()
        self.timer.start(33)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def set_user_id(self, user_id):
        """로컬 저장을 위한 현재 사용자 ID 설정."""
        self._user_id = user_id

    def _reset_calibration(self):
        self.calibration_data = []
        self.current_point_index = 0
        self.current_click_count = 0
        self._completed = False
        self._update_progress_text()
        self.calib_canvas.set_state(self.current_point_index, self.current_click_count)
        self.status_label.setText("준비됨")
        self.status_label.setStyleSheet("font-size: 14px; color: rgba(240, 240, 245, 0.45);")

    def _update_progress_text(self):
        if self.current_point_index < len(self.POINTS):
            self.progress_label.setText(
                f"포인트 {self.current_point_index + 1}/{len(self.POINTS)} - "
                f"클릭 {self.current_click_count}/{self.CLICKS_PER_POINT}"
            )
        else:
            self.progress_label.setText("캘리브레이션 완료")

    def _update_preview(self):
        """작은 카메라 미리보기 업데이트."""
        frame = self.camera.get_frame()
        if frame is None:
            return
        frame_mirror = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            160, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event):
        """캘리브레이션 클릭 처리: iris 위치 추출 + 화면 좌표 기록."""
        if self.current_point_index >= len(self.POINTS):
            return

        frame = self.camera.get_frame()
        if frame is None:
            self.status_label.setText("카메라 프레임을 가져올 수 없습니다")
            self.status_label.setStyleSheet("font-size: 14px; color: #f87171;")
            return

        iris_pos = self._extract_iris(frame)
        if iris_pos is None:
            self.status_label.setText("얼굴이 감지되지 않습니다 - 카메라를 바라보세요")
            self.status_label.setStyleSheet("font-size: 14px; color: #fbbf24;")
            return

        iris_x, iris_y = iris_pos
        rx, ry = self.POINTS[self.current_point_index]

        # 실제 화면 해상도 기준 좌표 (캘리브레이션은 전체 화면 좌표 필요)
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_x = screen_size.width() * rx
        screen_y = screen_size.height() * ry

        self.calibration_data.append({
            "iris_x": iris_x,
            "iris_y": iris_y,
            "screen_x": screen_x,
            "screen_y": screen_y,
        })

        self.current_click_count += 1
        self.status_label.setText(f"기록됨 (iris: {iris_x:.3f}, {iris_y:.3f})")
        self.status_label.setStyleSheet("font-size: 14px; color: #22d3ee;")

        if self.current_click_count >= self.CLICKS_PER_POINT:
            self.current_click_count = 0
            self.current_point_index += 1
            if self.current_point_index >= len(self.POINTS):
                self._finish_calibration()
                return

        self._update_progress_text()
        self.calib_canvas.set_state(self.current_point_index, self.current_click_count)

    def _extract_iris(self, frame):
        """로컬 MediaPipe Face Mesh로 iris 정규화 위치 추출."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        LEFT_IRIS = [473, 474, 475, 476, 477]
        RIGHT_IRIS = [468, 469, 470, 471, 472]

        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        def normalize_iris(iris_indices, inner_idx, outer_idx, top_idx, bottom_idx):
            iris_center = np.mean([get_point(i) for i in iris_indices], axis=0)
            inner = get_point(inner_idx)
            outer = get_point(outer_idx)
            top = get_point(top_idx)
            bottom = get_point(bottom_idx)
            eye_width = np.linalg.norm(outer - inner)
            eye_height = np.linalg.norm(bottom - top)
            if eye_width < 1 or eye_height < 1:
                return 0.5, 0.5
            eye_dir = (outer - inner) / eye_width
            iris_vec = iris_center - inner
            x_ratio = np.dot(iris_vec, eye_dir) / eye_width
            eye_vdir = (bottom - top) / eye_height
            y_ratio = np.dot(iris_center - top, eye_vdir) / eye_height
            return float(np.clip(x_ratio, 0, 1)), float(np.clip(y_ratio, 0, 1))

        lx, ly = normalize_iris(LEFT_IRIS, 362, 263, 386, 374)
        rx, ry = normalize_iris(RIGHT_IRIS, 133, 33, 159, 145)
        return (lx + rx) / 2, (ly + ry) / 2

    def _finish_calibration(self):
        """캘리브레이션 데이터를 gaze 서버로 전송 및 로컬 저장."""
        self.instruction_label.setText("캘리브레이션 전송 중...")
        screen = QApplication.primaryScreen()
        screen_size = screen.size()

        ok = self.network_client.set_gaze_calibration(
            self.calibration_data,
            screen_size.width(),
            screen_size.height(),
        )
        if ok:
            self._completed = True
            if self._user_id is not None:
                from client.core.calibration_store import save_gaze_calibration
                save_gaze_calibration(
                    self._user_id,
                    self.calibration_data,
                    screen_size.width(),
                    screen_size.height(),
                )
            self.status_label.setText("시선 캘리브레이션이 완료되었습니다!")
            self.status_label.setStyleSheet("font-size: 16px; color: #22d3ee; font-weight: bold;")
            self.instruction_label.setText("캘리브레이션 완료!")
            QTimer.singleShot(1500, self.done_requested.emit)
        else:
            self.status_label.setText("서버 전송 실패 - ai_gaze 서버를 확인하세요")
            self.status_label.setStyleSheet("font-size: 16px; color: #f87171; font-weight: bold;")
            self.instruction_label.setText("캘리브레이션 실패")



class MonitoringPage(QWidget):
    """실시간 모니터링 화면"""
    stop_requested = pyqtSignal()
    set_baseline_requested = pyqtSignal()
    recalibrate_gaze_requested = pyqtSignal()

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

        self.recalibrate_gaze_btn = QPushButton("시선 다시 설정")
        self.recalibrate_gaze_btn.setObjectName("SecondaryBtn")
        self.recalibrate_gaze_btn.clicked.connect(self.recalibrate_gaze_requested.emit)
        top_bar.addWidget(self.recalibrate_gaze_btn)

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
        apply_glass_shadow(video_card)
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
        self.overlay_label.setWordWrap(True)
        self.overlay_label.setStyleSheet("""
            background-color: rgba(239, 68, 68, 0.85);
            color: white;
            font-size: 24px;
            font-weight: bold;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.15);
        """)
        self.overlay_label.setMaximumWidth(600)
        self.overlay_label.setMinimumHeight(100)
        self.overlay_label.hide()
        alert_layout.addWidget(self.overlay_label, 0, Qt.AlignmentFlag.AlignCenter)

        # 거북목 경고 (노란색, 2초간 표시 + 경고음, 비집중으로 카운트 안 함)
        self.posture_alert_label = QLabel("거북목")
        self.posture_alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.posture_alert_label.setStyleSheet("""
            background-color: rgba(245, 158, 11, 0.90);
            color: #000;
            font-size: 28px;
            font-weight: bold;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.15);
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
        self.warning_card.setStyleSheet("QFrame#Card { border: 1px solid rgba(239, 68, 68, 0.50); }")
        apply_glass_shadow(self.warning_card)
        warning_vbox = QVBoxLayout(self.warning_card)
        warning_label = QLabel("DISTRACTION CAPTURE")
        warning_label.setStyleSheet("color: #f87171; font-weight: bold; font-size: 12px;")
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
        apply_glass_shadow(score_card)
        score_vbox = QVBoxLayout(score_card)
        self.add_stat(score_vbox, "FOCUS SCORE", "100%", "score")
        stats_layout.addWidget(score_card)

        # Other Stats Card
        info_card = QFrame()
        info_card.setObjectName("Card")
        apply_glass_shadow(info_card)
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
        apply_glass_shadow(chart_card)
        chart_vbox = QVBoxLayout(chart_card)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0d0d12')
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='#a78bfa', width=2))
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
        painter.setBrush(QBrush(QColor("#22d3ee")))
        path_focus = QPainterPath()
        path_focus.arcMoveTo(rect_f, start_deg)
        path_focus.arcTo(rect_f, start_deg, focus_span_deg)
        path_focus.arcTo(inner_rect_f, start_deg + focus_span_deg, -focus_span_deg)
        path_focus.closeSubpath()
        painter.drawPath(path_focus)

        painter.setBrush(QBrush(QColor("#f87171")))
        path_distract = QPainterPath()
        path_distract.arcMoveTo(rect_f, start_deg + focus_span_deg)
        path_distract.arcTo(rect_f, start_deg + focus_span_deg, distract_span_deg)
        path_distract.arcTo(inner_rect_f, start_deg + focus_span_deg + distract_span_deg, -distract_span_deg)
        path_distract.closeSubpath()
        painter.drawPath(path_distract)

        painter.setPen(QPen(QColor("#f0f0f5")))
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

        # Graphs Section
        graphs_layout = QHBoxLayout()
        graphs_layout.setSpacing(15)

        # Left Column: 통계 + 도넛 차트 (한 카드)
        bar_card = QFrame()
        bar_card.setObjectName("Card")
        apply_glass_shadow(bar_card)
        bar_vbox = QVBoxLayout(bar_card)
        bar_vbox.setContentsMargins(15, 15, 15, 15)
        bar_vbox.setSpacing(15)
        # 상단: DURATION, DISTRACTIONS / 최장·평균 집중시간
        stats_grid = QGridLayout()
        stats_grid.setSpacing(20)
        self.add_report_stat(stats_grid, "DURATION", "00:00", 0, 0, "duration")
        self.add_report_stat(stats_grid, "DISTRACTIONS", "0", 0, 1, "dist")
        self.add_report_stat(stats_grid, "최장 집중시간", "--", 1, 0, "longest_focus")
        self.add_report_stat(stats_grid, "평균 집중시간", "--", 1, 1, "avg_focus")
        bar_vbox.addLayout(stats_grid)
        bar_title = QLabel("집중 시간 vs 비집중 시간")
        bar_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #c084fc; margin-bottom: 10px;")
        bar_vbox.addWidget(bar_title)
        self.donut_widget = DonutChartWidget()
        self.donut_widget.setMinimumSize(200, 200)
        self.donut_widget.setStyleSheet("background-color: rgba(255, 255, 255, 0.04); border-radius: 8px;")
        bar_vbox.addWidget(self.donut_widget)
        graphs_layout.addWidget(bar_card, stretch=1)

        # Right Column: Two Line Charts
        right_column = QVBoxLayout()
        right_column.setSpacing(15)

        # Line Chart 1: Time-based focus status
        line1_card = QFrame()
        line1_card.setObjectName("Card")
        apply_glass_shadow(line1_card)
        line1_vbox = QVBoxLayout(line1_card)
        line1_vbox.setContentsMargins(15, 15, 15, 15)
        line1_title = QLabel("시간에 따른 집중 여부")
        line1_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #c084fc; margin-bottom: 10px;")
        line1_vbox.addWidget(line1_title)
        self.line1_plot = pg.PlotWidget()
        self.line1_plot.setMinimumHeight(200)
        self.line1_plot.setBackground('#0d0d12')
        self.line1_plot.setLabel('left', '집중 여부')
        self.line1_plot.setLabel('bottom', '시간')
        self.line1_plot.setYRange(0, 1.2)
        self.line1_plot.getAxis('left').setTicks([[(0, '비집중'), (1, '집중')]])
        line1_vbox.addWidget(self.line1_plot)
        right_column.addWidget(line1_card, stretch=1)

        # Line Chart 2: 10-minute interval focus duration count
        line2_card = QFrame()
        line2_card.setObjectName("Card")
        apply_glass_shadow(line2_card)
        line2_vbox = QVBoxLayout(line2_card)
        line2_vbox.setContentsMargins(15, 15, 15, 15)
        line2_title = QLabel("10분 간격 Sleepy 감지 횟수")
        line2_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #c084fc; margin-bottom: 10px;")
        line2_vbox.addWidget(line2_title)
        self.line2_plot = pg.PlotWidget()
        self.line2_plot.setMinimumHeight(200)
        self.line2_plot.setBackground('#0d0d12')
        self.line2_plot.setLabel('left', '횟수')
        self.line2_plot.setLabel('bottom', '')
        line2_vbox.addWidget(self.line2_plot)
        right_column.addWidget(line2_card, stretch=1)

        graphs_layout.addLayout(right_column, stretch=3)
        main_layout.addLayout(graphs_layout, stretch=1)

        # LLM 피드백 카드 (저장된 피드백 표시)
        feedback_card = QFrame()
        feedback_card.setObjectName("Card")
        apply_glass_shadow(feedback_card)
        feedback_card_vbox = QVBoxLayout(feedback_card)
        feedback_card_vbox.setContentsMargins(15, 15, 15, 15)
        feedback_title = QLabel("LLM 피드백")
        feedback_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #c084fc; margin-bottom: 10px;")
        feedback_card_vbox.addWidget(feedback_title)
        self.report_llm_feedback_label = QLabel("저장된 LLM 피드백이 없습니다.")
        self.report_llm_feedback_label.setWordWrap(True)
        self.report_llm_feedback_label.setStyleSheet("""
            font-size: 14px; color: #f0f0f5; background: rgba(255, 255, 255, 0.04); padding: 15px; border-radius: 8px; line-height: 1.5;
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
        
        pen = pg.mkPen(color='#a78bfa', width=2)
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
        bg = pg.BarGraphItem(x=x_centers, height=counts, width=bar_width, brush='#f87171')
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
        apply_glass_shadow(card)
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
            color: #f0f0f5;
            background: rgba(255, 255, 255, 0.04);
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
        self.from_date_edit.setStyleSheet("background-color: rgba(255, 255, 255, 0.07); color: #f0f0f5; padding: 6px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.12);")
        filter_row.addWidget(self.from_date_edit)
        filter_row.addWidget(QLabel("To:"))
        self.to_date_edit = QDateEdit()
        self.to_date_edit.setCalendarPopup(True)
        self.to_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.to_date_edit.setDate(QDate(2030, 12, 31))
        self.to_date_edit.setStyleSheet("background-color: rgba(255, 255, 255, 0.07); color: #f0f0f5; padding: 6px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.12);")
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
        self.table.viewport().setStyleSheet("background-color: #0f0f15;")
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0f0f15;
                color: #8b8b90;
                gridline-color: #1e1e26;
                border-radius: 8px;
                font-size: 14px;
                border: 1px solid #1e1e26;
            }
            QTableWidget QHeaderView, QTableWidget QScrollBar {
                background-color: #0f0f15;
            }
            QTableWidget QTableCornerButton::section,
            QTableWidget QHeaderView::section:empty {
                background-color: #0f0f15;
            }
            QHeaderView::section {
                background-color: #151520;
                color: #a78bfa;
                padding: 10px;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #1e1e26;
            }
            QTableWidget::item {
                padding: 10px;
                color: #8b8b90;
            }
            QTableWidget::item:selected {
                background-color: #1e1b4b;
                color: #f0f0f5;
            }
            QTableCornerButton::section {
                background-color: #151520;
                border: none;
            }
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

    def load_data(self, user_id=None):
        self._all_sessions = self.network_client.get_history(user_id=user_id)
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
        self.window().show_report_detail(session_data)

class DebugInferenceThread(QThread):
    """웹캠 프레임 1장을 4개 AI 서버의 /debug_inference에 병렬 전송."""
    results_ready = pyqtSignal(dict)

    def __init__(self, network_client, image_base64):
        super().__init__()
        self.network_client = network_client
        self.image_base64 = image_base64

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        servers = {
            "head": self.network_client.ai_head_url,
            "emotion": self.network_client.ai_emotion_url,
            "body": self.network_client.ai_body_url,
            "gaze": self.network_client.ai_gaze_url,
        }
        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    self.network_client.send_debug_inference, url, self.image_base64
                ): key
                for key, url in servers.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception:
                    results[key] = None

        self.results_ready.emit(results)


class DebugPage(QWidget):
    """2x2 그리드로 4개 AI 모델의 디버그 어노테이션 이미지를 동시 표시."""

    def __init__(self, camera, network_client):
        super().__init__()
        self.camera = camera
        self.network_client = network_client
        self._debug_thread = None
        self._has_annotated = set()  # 어노테이션 이미지를 받은 셀 추적

        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)

        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self._request_debug_inference)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Title bar
        title_bar = QHBoxLayout()
        title = QLabel("DEBUG VIEW")
        title.setObjectName("Header")
        title_bar.addWidget(title)
        title_bar.addStretch()
        layout.addLayout(title_bar)

        # 2x2 Grid
        grid = QGridLayout()
        grid.setSpacing(8)

        self.cells = {}
        cell_configs = [
            ("head",    "Head Pose (ai_head:8001)",   0, 0),
            ("emotion", "Emotion (ai_emotion:8002)",  0, 1),
            ("body",    "Body (ai_body:8003)",        1, 0),
            ("gaze",    "Gaze (ai_gaze:8005)",        1, 1),
        ]

        for key, title_text, row, col in cell_configs:
            card = QFrame()
            card.setObjectName("Card")
            apply_glass_shadow(card)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(4)

            cell_title = QLabel(title_text)
            cell_title.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #22d3ee; text-transform: uppercase;"
            )
            card_layout.addWidget(cell_title)

            img_label = QLabel("Waiting...")
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("background-color: rgba(0, 0, 0, 0.40); border-radius: 4px;")
            img_label.setMinimumSize(320, 240)
            card_layout.addWidget(img_label, stretch=1)

            info_label = QLabel("--")
            info_label.setStyleSheet("font-size: 11px; color: rgba(240, 240, 245, 0.45);")
            info_label.setWordWrap(True)
            info_label.setFixedHeight(40)
            card_layout.addWidget(info_label)

            grid.addWidget(card, row, col)
            self.cells[key] = {"image_label": img_label, "info_label": info_label}

        layout.addLayout(grid, stretch=1)

    def showEvent(self, event):
        super().showEvent(event)
        self._has_annotated.clear()
        self.preview_timer.start(33)
        self.inference_timer.start(500)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.preview_timer.stop()
        self.inference_timer.stop()

    def _update_preview(self):
        """아직 어노테이션을 받지 못한 셀에만 카메라 프리뷰 표시. 어노테이션 받은 셀은 건드리지 않음."""
        # 모든 셀이 이미 어노테이션을 받았으면 프리뷰 불필요
        if len(self._has_annotated) >= 4:
            return

        frame = self.camera.get_frame()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        for key, cell in self.cells.items():
            if key in self._has_annotated:
                continue  # 이미 어노테이션 이미지가 있는 셀은 스킵
            label = cell["image_label"]
            cell_w = max(label.width(), 320)
            cell_h = max(label.height(), 240)
            label.setPixmap(pixmap.scaled(
                cell_w, cell_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def _request_debug_inference(self):
        """현재 프레임을 4개 서버에 전송."""
        if self._debug_thread is not None and self._debug_thread.isRunning():
            return

        frame = self.camera.get_frame()
        if frame is None:
            return

        image_base64 = self.camera.frame_to_base64(frame)
        self._debug_thread = DebugInferenceThread(self.network_client, image_base64)
        self._debug_thread.results_ready.connect(self._on_debug_results)
        self._debug_thread.start()

    def _on_debug_results(self, results):
        """서버 응답의 어노테이션 이미지 + 수치로 각 셀 갱신."""
        import base64 as b64module

        for key in ("head", "emotion", "body", "gaze"):
            result = results.get(key)
            cell = self.cells[key]

            if result is None:
                cell["info_label"].setText("Server unreachable")
                cell["info_label"].setStyleSheet("font-size: 11px; color: #f87171;")
                continue

            # 어노테이션 이미지 디코딩 및 표시
            annotated_b64 = result.get("annotated_image")
            if annotated_b64:
                img_data = b64module.b64decode(annotated_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                ann_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if ann_frame is not None:
                    rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    label = cell["image_label"]
                    cell_w = max(label.width(), 320)
                    cell_h = max(label.height(), 240)
                    label.setPixmap(QPixmap.fromImage(qt_img).scaled(
                        cell_w, cell_h,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    ))
                    self._has_annotated.add(key)

            # 수치 텍스트 갱신
            data = result.get("data", {})
            cell["info_label"].setStyleSheet("font-size: 11px; color: rgba(240, 240, 245, 0.45);")

            if key == "head":
                p = data.get("pitch", 0)
                y = data.get("yaw", 0)
                r = data.get("roll", 0)
                cell["info_label"].setText(f"Pitch: {p:.1f}  Yaw: {y:.1f}  Roll: {r:.1f}")
            elif key == "emotion":
                emo = data.get("emotion", "?")
                conf = data.get("confidence", 0)
                cell["info_label"].setText(f"{emo} ({conf:.0%})" if emo else "No detection")
            elif key == "body":
                offset = data.get("distance_offset_cm")
                a = data.get("shoulder_angle")
                if offset is not None:
                    if offset > 0:
                        d_str = f"+{offset:.1f}cm (뒤)"
                    elif offset < 0:
                        d_str = f"{offset:.1f}cm (앞)"
                    else:
                        d_str = "0.0cm"
                else:
                    d = data.get("distance_cm")
                    d_str = f"{d:.1f}cm" if d is not None else "--"
                a_str = f"{a:.1f} deg" if a is not None else "--"
                cell["info_label"].setText(f"이동거리: {d_str}  Angle: {a_str}")
            elif key == "gaze":
                ix = data.get("iris_x", 0)
                iy = data.get("iris_y", 0)
                cal = "Yes" if data.get("calibrated") else "No"
                cell["info_label"].setText(f"Iris: ({ix:.3f}, {iy:.3f})  Calibrated: {cal}")


class FadeTransitionHelper:
    """QStackedWidget 페이지 전환 시 페이드인 애니메이션을 적용하는 헬퍼."""
    DURATION_MS = 200

    def __init__(self, stacked_widget: QStackedWidget):
        self._stack = stacked_widget
        self._in_progress = False
        self._anim = None
        self._outgoing_widget = None

    @property
    def outgoing_widget(self):
        return self._outgoing_widget if self._in_progress else None

    def fade_to(self, target):
        current = self._stack.currentWidget()
        if current is target:
            return
        if self._in_progress:
            self._finish_immediately()

        self._in_progress = True
        self._outgoing_widget = current

        effect = QGraphicsOpacityEffect(target)
        effect.setOpacity(0.0)
        target.setGraphicsEffect(effect)

        self._stack.setCurrentWidget(target)

        self._anim = QPropertyAnimation(effect, b"opacity")
        self._anim.setDuration(self.DURATION_MS)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._anim.finished.connect(self._on_done)
        self._anim.start()

    def _finish_immediately(self):
        if self._anim and self._anim.state() == QPropertyAnimation.State.Running:
            self._anim.stop()
        self._cleanup()

    def _on_done(self):
        self._cleanup()

    def _cleanup(self):
        self._in_progress = False
        current = self._stack.currentWidget()
        if current:
            current.setGraphicsEffect(None)
        self._outgoing_widget = None
        self._anim = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusMonitor AI")
        self.resize(1100, 850)
        self.setStyleSheet(STYLE_SHEET)

        self.camera = Camera()
        self.network_client = NetworkClient()
        self.is_monitoring = False
        self.current_user = None

        # --- Central widget: VBox(header, HBox(sidebar, stack), statusbar) ---
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header Bar
        self.header_bar = QFrame()
        self.header_bar.setFixedHeight(48)
        self.header_bar.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.03); border-bottom: 1px solid rgba(255, 255, 255, 0.06); }"
        )
        hb_layout = QHBoxLayout(self.header_bar)
        hb_layout.setContentsMargins(16, 0, 16, 0)
        hb_layout.addStretch()
        header_title = QLabel("FocusMonitor AI (포커스모니터 AI)")
        header_title.setStyleSheet("font-size: 14px; font-weight: 600; color: rgba(240, 240, 245, 0.70); letter-spacing: 1.5px;")
        hb_layout.addWidget(header_title)
        hb_layout.addStretch()
        self.header_bar.hide()
        root_layout.addWidget(self.header_bar)

        # Middle: Sidebar + Stack
        middle_layout = QHBoxLayout()
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet(SIDEBAR_STYLE)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(8, 16, 8, 16)
        sb_layout.setSpacing(4)

        self.sidebar_buttons = {}
        nav_items = [
            ("home", "\u2302  홈"),
            ("history", "\u29D6  기록"),
            ("debug", "\u2699  디버그"),
        ]
        for key, text in nav_items:
            btn = QPushButton(text)
            btn.setStyleSheet(SIDEBAR_BTN_STYLE)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            sb_layout.addWidget(btn)
            self.sidebar_buttons[key] = btn

        # Section divider
        sb_layout.addSpacing(8)
        divider_label = QLabel("환경 설정")
        divider_label.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: rgba(240, 240, 245, 0.30); "
            "text-transform: uppercase; letter-spacing: 2px; padding: 8px 16px;"
        )
        sb_layout.addWidget(divider_label)

        logout_btn = QPushButton("\u2190  로그아웃")
        logout_btn.setStyleSheet(SIDEBAR_BTN_STYLE)
        logout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        sb_layout.addWidget(logout_btn)
        self.sidebar_buttons["logout"] = logout_btn

        quit_btn = QPushButton("\u2716  종료")
        quit_btn.setStyleSheet("""
            QPushButton {
                text-align: left; padding: 12px 20px; border: none;
                border-radius: 10px; font-size: 14px;
                color: #f87171; background: transparent;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.12);
            }
        """)
        quit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_btn.clicked.connect(self.close)
        sb_layout.addWidget(quit_btn)

        sb_layout.addStretch()

        # User profile area at bottom
        self.user_profile_frame = QFrame()
        self.user_profile_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        """)
        up_layout = QHBoxLayout(self.user_profile_frame)
        up_layout.setContentsMargins(12, 10, 12, 10)
        up_layout.setSpacing(10)
        user_avatar = QLabel("\u263A")
        user_avatar.setFixedSize(36, 36)
        user_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        user_avatar.setStyleSheet("""
            font-size: 18px;
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
            border-radius: 18px;
            color: #fff;
        """)
        up_layout.addWidget(user_avatar)
        user_info = QVBoxLayout()
        user_info.setSpacing(0)
        self.user_name_label = QLabel("사용자")
        self.user_name_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #f0f0f5;")
        user_info.addWidget(self.user_name_label)
        version_label = QLabel("FocusMonitor v1.0")
        version_label.setStyleSheet("font-size: 10px; color: rgba(240, 240, 245, 0.40);")
        user_info.addWidget(version_label)
        up_layout.addLayout(user_info)
        up_layout.addStretch()
        sb_layout.addWidget(self.user_profile_frame)

        self.sidebar.hide()
        middle_layout.addWidget(self.sidebar)

        # Pages Setup (QStackedWidget)
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background-color: #0d0d12; }")
        middle_layout.addWidget(self.stack)
        self._transition = FadeTransitionHelper(self.stack)
        root_layout.addLayout(middle_layout, stretch=1)

        # Status Bar
        self.status_bar_widget = QFrame()
        self.status_bar_widget.setFixedHeight(32)
        self.status_bar_widget.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.03); border-top: 1px solid rgba(255, 255, 255, 0.06); }"
        )
        stb_layout = QHBoxLayout(self.status_bar_widget)
        stb_layout.setContentsMargins(16, 0, 16, 0)
        self.status_dot = QLabel("\u25CF")
        self.status_dot.setStyleSheet("font-size: 8px; color: #ef4444;")
        self.status_dot.setFixedWidth(12)
        stb_layout.addWidget(self.status_dot)
        self.status_text = QLabel("연결 대기")
        self.status_text.setStyleSheet("font-size: 11px; color: rgba(240, 240, 245, 0.40);")
        stb_layout.addWidget(self.status_text)
        sep = QLabel("|")
        sep.setStyleSheet("font-size: 11px; color: rgba(255, 255, 255, 0.12); padding: 0 4px;")
        stb_layout.addWidget(sep)
        ver = QLabel("FocusMonitor Core v1.0")
        ver.setStyleSheet("font-size: 11px; color: rgba(240, 240, 245, 0.40);")
        stb_layout.addWidget(ver)
        stb_layout.addStretch()
        self.status_bar_widget.hide()
        root_layout.addWidget(self.status_bar_widget)

        # --- Create Pages ---
        self.splash_page = SplashPage()
        self.login_page = LoginPage(self.network_client)
        self.register_page = RegisterPage(self.network_client)

        self.main_page = MainPage()
        self.calibration_page = DistanceCalibrationPage(self.camera, self.network_client)
        self.head_calibration_page = HeadPoseCalibrationPage(self.camera, self.network_client)
        self.gaze_calibration_page = GazeCalibrationPage(self.camera, self.network_client)
        self.monitoring_page = MonitoringPage()
        self.report_page = ReportPage()
        self.report_page.set_network_client(self.network_client)
        self.analysis_page = LlmAnalysisPage(self.network_client)
        self.history_page = HistoryPage(self.network_client)
        self.debug_page = DebugPage(self.camera, self.network_client)

        self.stack.addWidget(self.splash_page)
        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.register_page)
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.calibration_page)
        self.stack.addWidget(self.head_calibration_page)
        self.stack.addWidget(self.gaze_calibration_page)
        self.stack.addWidget(self.monitoring_page)
        self.stack.addWidget(self.report_page)
        self.stack.addWidget(self.analysis_page)
        self.stack.addWidget(self.history_page)
        self.stack.addWidget(self.debug_page)

        # Auth pages (no sidebar)
        self._auth_pages = {self.splash_page, self.login_page, self.register_page}

        # Map sidebar buttons to pages for active state
        self._sidebar_page_map = {
            "home": self.main_page,
            "history": self.history_page,
            "debug": self.debug_page,
        }

        # --- Signals: Splash ---
        self.splash_page.finished.connect(lambda: self._animated_switch(self.login_page))

        # --- Signals: Auth ---
        self.login_page.login_success.connect(self._on_login_success)
        self.login_page.register_requested.connect(lambda: self._animated_switch(self.register_page))
        self.register_page.register_success.connect(self._on_register_success)
        self.register_page.back_to_login.connect(lambda: self._animated_switch(self.login_page))

        # --- Signals: Sidebar Navigation ---
        self.sidebar_buttons["home"].clicked.connect(lambda: self._navigate_to("home"))
        self.sidebar_buttons["history"].clicked.connect(self._nav_to_history)
        self.sidebar_buttons["debug"].clicked.connect(lambda: self._navigate_to("debug"))
        self.sidebar_buttons["logout"].clicked.connect(self._on_logout)

        # --- Signals: Page Actions ---
        self.main_page.start_requested.connect(self._begin_session_flow)
        self.calibration_page.done_requested.connect(self._on_distance_calibration_done)
        self.gaze_calibration_page.done_requested.connect(self._on_gaze_calibration_done)
        self._gaze_return_to_monitoring = False
        self._session_start_flow = False
        self.monitoring_page.stop_requested.connect(self.stop_session)
        self.monitoring_page.set_baseline_requested.connect(self._on_set_baseline_from_monitoring)
        self.monitoring_page.recalibrate_gaze_requested.connect(self._on_recalibrate_gaze_from_monitoring)
        self.report_page.home_requested.connect(lambda: self._navigate_to("home"))
        self.report_page.llm_analysis_requested.connect(self.show_analysis)
        self.analysis_page.home_requested.connect(lambda: self._navigate_to("home"))
        self.history_page.home_requested.connect(lambda: self._navigate_to("home"))

        # Track page changes for sidebar active state
        self.stack.currentChanged.connect(self._on_page_changed)

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
        self._saved_geometry = None
        self._was_maximized = False

        # 버튼 프레스 애니메이션 설치
        ButtonAnimationFilter.install_on_all(self)

    def _animated_switch(self, page):
        """페이드 애니메이션과 함께 페이지 전환."""
        self._transition.fade_to(page)

    def _navigate_to(self, key):
        if key == "home":
            self._session_start_flow = False
        page = self._sidebar_page_map.get(key)
        if page:
            self._animated_switch(page)

    def _nav_to_history(self):
        self.show_history()

    def _update_sidebar_active(self):
        current = self.stack.currentWidget()
        for key, btn in self.sidebar_buttons.items():
            page = self._sidebar_page_map.get(key)
            if page and page == current:
                btn.setStyleSheet(SIDEBAR_BTN_ACTIVE_STYLE)
            elif key != "logout":
                btn.setStyleSheet(SIDEBAR_BTN_STYLE)

    def _on_page_changed(self, index):
        current = self.stack.currentWidget()
        is_auth = current in self._auth_pages
        self.header_bar.setVisible(not is_auth)
        self.sidebar.setVisible(not is_auth)
        self.status_bar_widget.setVisible(not is_auth)
        if current is not self.gaze_calibration_page:
            self._restore_window_from_gaze_calibration()
        if not is_auth:
            self._update_sidebar_active()

    def _restore_window_from_gaze_calibration(self):
        """시선 캘리브레이션에서 나올 때 전체화면 해제 후 저장해 둔 크기/위치로 복원."""
        if self._saved_geometry is None:
            return
        self.showNormal()
        self.restoreGeometry(self._saved_geometry)
        if self._was_maximized:
            self.showMaximized()
        self._saved_geometry = None
        self._was_maximized = False

    def _enter_gaze_calibration_fullscreen(self):
        """시선 캘리브레이션 진입 시 전체화면 전환 (복원용 상태 저장)."""
        self._was_maximized = self.isMaximized()
        self._saved_geometry = self.saveGeometry()
        self.showFullScreen()

    def _on_login_success(self, user_data: dict):
        self.current_user = user_data
        # Update user profile in sidebar
        display_name = user_data.get("display_name") or user_data.get("username") or "사용자"
        self.user_name_label.setText(display_name)
        self.login_page.clear_fields()
        self._animated_switch(self.main_page)

    def _on_register_success(self, user_data: dict):
        self.register_page.clear_fields()
        QMessageBox.information(self, "회원가입 완료", "계정이 생성되었습니다. 로그인해주세요.")
        self._animated_switch(self.login_page)

    def _on_logout(self):
        self.current_user = None
        self._session_start_flow = False
        self.login_page.clear_fields()
        self._animated_switch(self.login_page)

    def start_session(self):
        """모니터링 시작: 거리 캘리브레이션 → 시선 캘리브레이션 → 모니터링 플로우."""
        self._begin_session_flow()

    def _begin_session_flow(self):
        """Step 1: 거리 캘리브레이션 페이지로 이동."""
        self._session_start_flow = True
        self._animated_switch(self.calibration_page)

    def _on_distance_calibration_done(self):
        """거리 캘리브레이션 완료 후 플로우 분기."""
        if self._session_start_flow:
            self._show_gaze_calibration_choice()
        else:
            self._navigate_to("home")

    def _show_gaze_calibration_choice(self):
        """Step 2: 저장된 시선 캘리브레이션 데이터 확인 및 선택."""
        from client.core.calibration_store import load_gaze_calibration

        user_id = self.current_user.get("user_id") if self.current_user else None
        saved = load_gaze_calibration(user_id) if user_id else None

        if saved is None:
            self._navigate_to_gaze_for_flow()
            return

        timestamp_str = saved.get("timestamp", "")
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(timestamp_str)
            display_time = ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            display_time = "알 수 없음"

        msg = QMessageBox(self)
        msg.setWindowTitle("시선 캘리브레이션")
        msg.setText(f"이전 시선 캘리브레이션 데이터가 있습니다.\n(저장 시각: {display_time})")
        msg.setInformativeText("이전 설정을 사용하시겠습니까?")
        recalibrate_btn = msg.addButton("새로 설정", QMessageBox.ButtonRole.RejectRole)
        use_recent_btn = msg.addButton("최근 설정 사용", QMessageBox.ButtonRole.AcceptRole)
        use_recent_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
                color: #ffffff; border: none;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c7ff7, stop:1 #9d75f8);
            }
        """)
        msg.setDefaultButton(use_recent_btn)
        msg.exec()

        if msg.clickedButton() == use_recent_btn:
            ok = self.network_client.set_gaze_calibration(
                saved["calibration_data"],
                saved["screen_width"],
                saved["screen_height"],
            )
            if ok:
                self._finalize_session_start()
            else:
                QMessageBox.warning(self, "시선 캘리브레이션",
                    "저장된 캘리브레이션을 서버에 전송하지 못했습니다.\n새로 캘리브레이션을 진행합니다.")
                self._navigate_to_gaze_for_flow()
        else:
            self._navigate_to_gaze_for_flow()

    def _navigate_to_gaze_for_flow(self):
        """시선 캘리브레이션 페이지로 이동 (user_id 설정 포함). 캘리브레이션 중에만 전체화면."""
        user_id = self.current_user.get("user_id") if self.current_user else None
        self.gaze_calibration_page.set_user_id(user_id)
        self._enter_gaze_calibration_fullscreen()
        self._animated_switch(self.gaze_calibration_page)

    def _finalize_session_start(self):
        """최종 단계: 서버 세션 시작 후 모니터링 진입."""
        self._session_start_flow = False
        user_id = self.current_user.get("user_id") if self.current_user else None
        session_data = self.network_client.start_session(user_id=user_id)
        if session_data and session_data.get("session_id"):
            self.current_session_id = session_data.get("session_id")
            logger.info(f"Session started on server: {self.current_session_id}")

            self.is_monitoring = True
            self.start_time = time.time()
            self.distraction_count = 0
            self.history_scores = []
            self._animated_switch(self.monitoring_page)
            self.request_inference()
            self.inference_timer.start(3000)
        else:
            logger.error("Failed to connect to Operation Server.")
            QMessageBox.critical(self, "Connection Error",
                                "운영 서버와 연결할 수 없습니다.\n서버 상태를 확인하고 다시 시도해주세요.")
            self.check_server_connection()
            self._navigate_to("home")

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
        self._animated_switch(self.report_page)
        self.current_session_id = None

    def show_analysis(self, session_id):
        self.analysis_page.start_analysis(session_id)
        self._animated_switch(self.analysis_page)

    def show_history(self):
        user_id = self.current_user.get("user_id") if self.current_user else None
        self.history_page.load_data(user_id=user_id)
        self._animated_switch(self.history_page)

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

    def _on_recalibrate_gaze_from_monitoring(self):
        """모니터링 중 시선 캘리브레이션 다시 설정: gaze 캘리브레이션 페이지로 이동."""
        self._gaze_return_to_monitoring = True
        user_id = self.current_user.get("user_id") if self.current_user else None
        self.gaze_calibration_page.set_user_id(user_id)
        self._enter_gaze_calibration_fullscreen()
        self._animated_switch(self.gaze_calibration_page)

    def _on_gaze_calibration_done(self):
        """시선 캘리브레이션 완료 후 원래 페이지로 복귀."""
        self._restore_window_from_gaze_calibration()
        if self._session_start_flow:
            if self.gaze_calibration_page._completed:
                self._finalize_session_start()
            else:
                self._session_start_flow = False
                self._navigate_to("home")
        elif self._gaze_return_to_monitoring:
            self._gaze_return_to_monitoring = False
            self._animated_switch(self.monitoring_page)
        else:
            self._animated_switch(self.main_page)

    def show_report_detail(self, session_data):
        self.report_page.set_report_data(session_data)
        # 과거 세션: llm_comment가 없을 때만 "LLM 분석 시작" 버튼 표시 (나중에 분석 요청 가능)
        has_llm = bool((session_data.get('llm_comment') or "").strip())
        self.report_page.set_llm_button_visible(not has_llm)
        self._animated_switch(self.report_page)

    def check_server_connection(self):
        if not self.is_monitoring:
            connected = self.network_client.check_connection()
            # Update MainPage status cards
            self.main_page.set_connection_status(connected)
            # Update status bar
            if connected:
                self.status_dot.setStyleSheet("font-size: 8px; color: #10b981;")
                self.status_text.setText("세션 준비됨")
            else:
                self.status_dot.setStyleSheet("font-size: 8px; color: #ef4444;")
                self.status_text.setText("연결 대기")

    def update_ui(self):
        frame = self.camera.get_frame()
        if frame is not None:
            if (self.stack.currentWidget() == self.monitoring_page
                    or self._transition.outgoing_widget == self.monitoring_page):
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
            m_page.stat_score.setStyleSheet("color: #f87171;")
            m_page.stat_pose.setText("Away")
            
            if hasattr(self, 'sent_frame'):
                rgb = cv2.cvtColor(cv2.flip(self.sent_frame, 1), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                m_page.warning_img_label.setPixmap(QPixmap.fromImage(img).scaled(
                    280, 210, Qt.AspectRatioMode.KeepAspectRatio))
                m_page.warning_card.show()
            
            details = "<br>".join(f"• {m.strip()}" for m in result.status_message.split(" | "))
            m_page.overlay_label.setText(f"<div style='text-align:center;'><b style='font-size:26px;'>ATTENTION!</b><br><span style='font-size:16px;'>{details}</span></div>")
            m_page.overlay_label.show()
            QApplication.beep()
            QTimer.singleShot(2000, m_page.overlay_label.hide)
        else:
            m_page.stat_score.setStyleSheet("color: #22d3ee;")
            m_page.stat_pose.setText("Centered")
            m_page.overlay_label.hide()

        # 거북목 경고: 비집중으로 카운트하지 않고, 노란 경고 2초 + 경고음만
        if result.body_pose and result.body_pose.get("posture_alert"):
            m_page.posture_alert_label.setText("거북목")
            m_page.posture_alert_label.show()
            QApplication.beep()
            QTimer.singleShot(2000, m_page.posture_alert_label.hide)
        # concentration_score가 오면 그 값을 FOCUS SCORE로 사용, 없으면 기존 로직 fallback
        score = getattr(result, "concentration_score", None)
        if score is None:
            score = max(0, 100 - (self.distraction_count * 2))

        m_page.stat_score.setText(f"{int(score)}%")
        self.history_scores.append(score)
        if len(self.history_scores) > 100: self.history_scores.pop(0)
        m_page.curve.setData(self.history_scores)

    def closeEvent(self, event):
        self.camera.release()
        event.accept()
