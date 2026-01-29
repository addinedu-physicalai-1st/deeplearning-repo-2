import cv2
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_status_signal = pyqtSignal(str, str)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager
        self._run_flag = True
        self.sensitivity = 20

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            processed_frame, status_text, status_color = self.session_manager.process_frame(frame, self.sensitivity)
            
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = qt_img.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            
            self.change_pixmap_signal.emit(p)
            self.update_status_signal.emit(status_text, status_color)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_sensitivity(self, value):
        self.sensitivity = value
