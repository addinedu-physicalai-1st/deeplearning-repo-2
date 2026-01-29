import sys
import json
import os

# --- 경로 설정: src 폴더를 path에 추가하여 서브모듈 참조 가능하게 함 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QStackedWidget, 
                             QRadioButton, QSlider, QListWidget, QListWidgetItem, 
                             QMessageBox, QFrame, QTextBrowser)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QColor
from core.session_manager import SessionManager
from core.video_thread import VideoThread
from database.database_handler import DatabaseHandler
from llm.llm_client import LLMClient

# --- Phase 3: 비동기 LLM 워커 스레드 ---
class LLMWorker(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, session_data):
        super().__init__()
        self.session_data = session_data
        self.client = LLMClient()

    def run(self):
        feedback = self.client.generate_feedback(self.session_data)
        self.finished_signal.emit(feedback)

# --- GUI Logic ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Focus Coach v4.2 (Feedback UI)")
        self.setGeometry(100, 100, 1000, 750)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        self.session_manager = SessionManager()
        self.db_handler = DatabaseHandler()
        self.init_ui()

    def init_ui(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.page_main = MainMenuPage(self)
        self.page_selection = ModelSelectionPage(self)
        self.page_monitor = MonitoringPage(self)
        self.page_result = ResultPage(self)
        self.page_history = HistoryPage(self)

        self.stacked_widget.addWidget(self.page_main)
        self.stacked_widget.addWidget(self.page_selection)
        self.stacked_widget.addWidget(self.page_monitor)
        self.stacked_widget.addWidget(self.page_result)
        self.stacked_widget.addWidget(self.page_history)
        self.stacked_widget.setCurrentIndex(0)

    def go_to_main(self): self.stacked_widget.setCurrentIndex(0)
    def go_to_selection(self): self.stacked_widget.setCurrentIndex(1)
    def go_to_monitor(self, model_type):
        self.page_monitor.set_model(model_type)
        self.stacked_widget.setCurrentIndex(2)
        self.page_monitor.start_camera()
    def go_to_result(self, session_data):
        self.page_result.update_data(session_data)
        self.stacked_widget.setCurrentIndex(3)
    def go_to_history(self):
        self.page_history.load_history()
        self.stacked_widget.setCurrentIndex(4)

class MainMenuPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Focus Coach AI")
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        
        btn_start = QPushButton("집중 시작하기")
        btn_start.setFixedSize(250, 60)
        btn_start.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px; font-size: 18px;")
        btn_start.clicked.connect(self.main_window.go_to_selection)
        
        btn_history = QPushButton("통계 기록 보기")
        btn_history.setFixedSize(250, 60)
        btn_history.setStyleSheet("background-color: #555; color: white; border-radius: 10px; font-size: 18px; margin-top: 10px;")
        btn_history.clicked.connect(self.main_window.go_to_history)

        btn_exit = QPushButton("종료")
        btn_exit.setFixedSize(250, 60)
        btn_exit.setStyleSheet("background-color: #d32f2f; color: white; border-radius: 10px; font-size: 18px; margin-top: 10px;")
        btn_exit.clicked.connect(QApplication.instance().quit)

        layout.addWidget(title)
        layout.addWidget(btn_start)
        layout.addWidget(btn_history)
        layout.addWidget(btn_exit)
        self.setLayout(layout)

class ModelSelectionPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 50, 50, 50)
        lbl = QLabel("오늘 관리할 요소를 선택하세요")
        lbl.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        layout.addWidget(lbl)
        self.rb_head = QRadioButton("자세 관리 (Head Pose)")
        self.rb_head.setChecked(True)
        self.rb_head.setFont(QFont("Arial", 14))
        layout.addWidget(self.rb_head)
        layout.addStretch()
        btn_next = QPushButton("Start Monitoring")
        btn_next.setFixedHeight(50)
        btn_next.setStyleSheet("background-color: #2196F3; color: white; border-radius: 5px;")
        btn_next.clicked.connect(lambda: self.main_window.go_to_monitor("HEAD"))
        layout.addWidget(btn_next)
        btn_back = QPushButton("뒤로가기")
        btn_back.setStyleSheet("background: none; border: none; color: #aaa;")
        btn_back.clicked.connect(self.main_window.go_to_main)
        layout.addWidget(btn_back)
        self.setLayout(layout)

class MonitoringPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.thread = None
        self.init_ui()
    def init_ui(self):
        layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        self.image_label = QLabel("Camera Loading...")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: #000;")
        video_layout.addWidget(self.image_label)
        layout.addLayout(video_layout, 7)
        
        control_layout = QVBoxLayout()
        self.status_label = QLabel("READY")
        self.status_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #888; border: 2px solid #555; padding: 20px;")
        
        slider_box = QVBoxLayout()
        slider_label = QLabel("민감도 조절 (Angle)")
        self.slider_val_label = QLabel("20°")
        self.slider_val_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.slider_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 45)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.update_sensitivity)

        slider_box.addWidget(slider_label)
        slider_box.addWidget(self.slider)
        slider_box.addWidget(self.slider_val_label)

        self.btn_stop = QPushButton("중지 및 분석")
        self.btn_stop.setFixedHeight(60)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; border-radius: 10px;")
        self.btn_stop.clicked.connect(self.stop_monitoring)
        
        control_layout.addWidget(self.status_label)
        control_layout.addSpacing(30)
        control_layout.addLayout(slider_box)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_stop)
        layout.addLayout(control_layout, 3)
        self.setLayout(layout)
        
    def set_model(self, model_type): self.model_type = model_type
    def start_camera(self):
        self.main_window.session_manager.start_session(self.model_type)
        self.thread = VideoThread(self.main_window.session_manager)
        self.thread.change_pixmap_signal.connect(lambda img: self.image_label.setPixmap(QPixmap.fromImage(img)))
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.sensitivity = self.slider.value()
        self.thread.start()
    def stop_monitoring(self):
        if self.thread: 
            self.thread.stop()
            self.thread.deleteLater()
            self.thread = None
        session_id, data = self.main_window.session_manager.stop_session()
        self.main_window.go_to_result(data)
    def update_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; border: 3px solid {color}; padding: 20px;")
    def update_sensitivity(self, val):
        self.slider_val_label.setText(f"{val}°")
        if self.thread: self.thread.update_sensitivity(val)

class ResultPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.worker = None
        self.current_session_id = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        lbl_title = QLabel("Session Report")
        lbl_title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        layout.addWidget(lbl_title)
        
        # 상단: 수치 데이터
        stats_layout = QHBoxLayout()
        self.lbl_score = QLabel("Score: -")
        self.lbl_score.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.lbl_time = QLabel("Time: -")
        self.lbl_count = QLabel("Distractions: -")
        stats_layout.addWidget(self.lbl_score)
        stats_layout.addSpacing(20)
        stats_layout.addWidget(self.lbl_time)
        stats_layout.addSpacing(20)
        stats_layout.addWidget(self.lbl_count)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)

        layout.addSpacing(20)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        layout.addSpacing(20)

        # 하단: AI 코칭 영역 (분리된 UI)
        self.lbl_coach_title = QLabel("🤖 AI Focus Coach")
        self.lbl_coach_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.lbl_coach_title.setStyleSheet("color: #2196F3;")
        layout.addWidget(self.lbl_coach_title)

        # 1. 코멘트 (위로/격려) 섹션
        self.lbl_comment_header = QLabel("💬 Comment")
        self.lbl_comment_header.setStyleSheet("font-weight: bold; margin-top: 10px; color: #FFD54F;")
        self.lbl_comment_content = QLabel("분석 대기 중...")
        self.lbl_comment_content.setStyleSheet("background-color: #424242; padding: 15px; border-radius: 8px; font-style: italic;")
        self.lbl_comment_content.setWordWrap(True)
        
        layout.addWidget(self.lbl_comment_header)
        layout.addWidget(self.lbl_comment_content)

        # 2. 피드백 (집중 향상 팁) 섹션
        self.lbl_feedback_header = QLabel("⚡ Focus Tips")
        self.lbl_feedback_header.setStyleSheet("font-weight: bold; margin-top: 15px; color: #69F0AE;")
        self.lbl_feedback_content = QLabel("")
        self.lbl_feedback_content.setStyleSheet("background-color: #333333; padding: 15px; border-radius: 8px;")
        self.lbl_feedback_content.setWordWrap(True)

        layout.addWidget(self.lbl_feedback_header)
        layout.addWidget(self.lbl_feedback_content)
        layout.addStretch()

        btn_home = QPushButton("메인으로")
        btn_home.setStyleSheet("background-color: #555; color: white; padding: 12px; border-radius: 5px;")
        btn_home.clicked.connect(self.main_window.go_to_main)
        layout.addWidget(btn_home)
        self.setLayout(layout)

    def update_data(self, data):
        self.current_session_id = data.get("id") 
        score = data.get("focus_score", 0)
        self.lbl_score.setText(f"{score}점")
        self.lbl_time.setText(f"{data.get('duration', 0)}초")
        self.lbl_count.setText(f"{data.get('distract_cnt', 0)}회")
        
        color = "#4CAF50" if score >= 80 else "#FFC107" if score >= 50 else "#F44336"
        self.lbl_score.setStyleSheet(f"color: {color};")
        
        # 로딩 상태 표시
        self.lbl_comment_content.setText("데이터를 분석하고 있습니다...")
        self.lbl_feedback_content.setText("잠시만 기다려주세요...")
        
        self.worker = LLMWorker(data)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self, json_text):
        try:
            # JSON 파싱
            result = json.loads(json_text)
            comment = result.get("comment", "분석 결과가 없습니다.")
            feedback = result.get("feedback", "피드백이 없습니다.")
            
            self.lbl_comment_content.setText(f"\"{comment}\"")
            self.lbl_feedback_content.setText(feedback)
            
            # DB 업데이트 (JSON 원본 저장)
            if self.current_session_id:
                self.main_window.db_handler.update_feedback(self.current_session_id, json_text)
                
        except json.JSONDecodeError:
            self.lbl_comment_content.setText("AI 응답 형식이 올바르지 않습니다.")
            self.lbl_feedback_content.setText(json_text) # 원본 텍스트 출력

class HistoryPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()
        lbl_title = QLabel("기록 조회")
        lbl_title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        layout.addWidget(lbl_title)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        btn_delete = QPushButton("선택 항목 삭제")
        btn_delete.setStyleSheet("background-color: #d32f2f; color: white; padding: 10px; border-radius: 5px;")
        btn_delete.clicked.connect(self.delete_item)
        btn_back = QPushButton("뒤로가기")
        btn_back.setStyleSheet("padding: 10px;")
        btn_back.clicked.connect(self.main_window.go_to_main)
        
        btn_layout.addWidget(btn_delete)
        btn_layout.addWidget(btn_back)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
    def load_history(self):
        self.list_widget.clear()
        sessions = self.main_window.db_handler.get_all_sessions()
        if not sessions:
            self.list_widget.addItem("저장된 기록이 없습니다.")
            return

        for s in sessions:
            # 리스트에는 간단히 표시
            item_text = f"[{s['start_time'][:16]}] {s['focus_score']}점 (이탈 {s['distract_cnt']}회)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, s['id'])
            
            # 툴팁에 상세 피드백 표시 (JSON 파싱 시도)
            if s['ai_feedback']:
                try:
                    fb_data = json.loads(s['ai_feedback'])
                    tooltip = f"Comment: {fb_data.get('comment')}\nTips: {fb_data.get('feedback')}"
                    item.setToolTip(tooltip)
                except:
                    item.setToolTip(s['ai_feedback'])
                    
            self.list_widget.addItem(item)
            
    def delete_item(self):
        current_item = self.list_widget.currentItem()
        if not current_item: return
        session_id = current_item.data(Qt.ItemDataRole.UserRole)
        if not session_id: return
        self.main_window.db_handler.delete_session(session_id)
        self.load_history()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
