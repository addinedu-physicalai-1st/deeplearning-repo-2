import cv2
import base64
import numpy as np

class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def frame_to_base64(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text

    def release(self):
        self.cap.release()
