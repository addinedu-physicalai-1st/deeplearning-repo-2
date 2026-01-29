import cv2
import numpy as np
import mediapipe.python.solutions as mp_solutions
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image): pass

class HeadPoseDetector(BaseDetector):
    def __init__(self):
        self.mp_face_mesh = mp_solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True
        )
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, 330.0, -65.0), (-225.0, -170.0, -135.0),
            (225.0, -170.0, -135.0), (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)
        ])

    def detect(self, image):
        img_h, img_w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return None, None

        face_landmarks = results.multi_face_landmarks[0]
        idx_list = [1, 152, 33, 263, 61, 291]
        image_points = []
        for idx in idx_list:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            image_points.append([x, y])

        image_points = np.array(image_points, dtype="double")
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, r_vec, t_vec = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        rmat, _ = cv2.Rodrigues(r_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        if roll > 0: roll -= 180
        else: roll += 180
        pitch = -pitch 

        nose_tip = (int(image_points[0][0]), int(image_points[0][1]))
        return (pitch, yaw, roll), (nose_tip, r_vec, t_vec, camera_matrix, dist_coeffs)

    def draw_debug(self, image, pose_data):
        if pose_data is None: return
        nose_tip, r_vec, t_vec, cam_mat, dist = pose_data
        axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        imgpts, _ = cv2.projectPoints(axis, r_vec, t_vec, cam_mat, dist)
        imgpts = imgpts.astype(int)
        cv2.line(image, nose_tip, tuple(imgpts[0].ravel()), (0, 0, 255), 3)
        cv2.line(image, nose_tip, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        cv2.line(image, nose_tip, tuple(imgpts[2].ravel()), (255, 0, 0), 3)
