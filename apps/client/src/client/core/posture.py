"""
Posture/distance logic from posture.py: MediaPipe Holistic, shoulder-based distance and baseline.
"""
import math
import cv2
import mediapipe as mp


class PostureMonitor:
    """MediaPipe Holistic-based posture monitor: shoulder angle, distance_cm, posture % vs baseline."""

    REAL_SHOULDER_CM = 40.0
    FOCAL_MULTIPLIER = 1.2

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.baseline_distance = None
        self.baseline_distance_cm = None
        self.show_landmarks = True
        self.show_guideline = True

    def set_baseline_from_distance(self, distance_cm: float) -> None:
        """Set baseline to current distance (call after user clicks '정자세 설정')."""
        if distance_cm is not None:
            self.baseline_distance = distance_cm
            self.baseline_distance_cm = distance_cm

    def process_frame(self, frame):
        """
        Returns (image_rgb, shoulder_angle, distance_cm, posture_percentage, distance_offset_cm).
        posture_percentage and distance_offset_cm are None until baseline is set.
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

        height, width, _ = image.shape
        shoulder_angle = None
        distance_cm = None
        posture_percentage = None
        distance_offset_cm = None

        if results.face_landmarks and results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            left_sh = pose_landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_sh = pose_landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            lx, ly = left_sh.x * width, left_sh.y * height
            rx, ry = right_sh.x * width, right_sh.y * height
            dx = rx - lx
            dy = ry - ly
            shoulder_angle = math.degrees(math.atan2(dy, dx))
            shoulder_angle = abs(shoulder_angle)

            pixel_shoulder_dist = math.hypot(dx, dy)
            focal_px = width * self.FOCAL_MULTIPLIER
            if pixel_shoulder_dist > 0:
                distance_cm = (self.REAL_SHOULDER_CM * focal_px) / pixel_shoulder_dist
            else:
                distance_cm = None

            if distance_cm is not None:
                if self.baseline_distance is not None:
                    distance_decrease = max(0, self.baseline_distance - distance_cm)
                    decrease_percentage = (distance_decrease / self.baseline_distance) * 30
                    posture_percentage = 100 - decrease_percentage
                    posture_percentage = max(0, min(100, posture_percentage))
                if self.baseline_distance_cm is not None:
                    distance_offset_cm = distance_cm - self.baseline_distance_cm

            if self.show_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    self.mp_holistic.FACEMESH_TESSELATION,
                    None,
                    mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )
            if self.show_guideline:
                cv2.line(image, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
                cv2.line(image, (0, height // 2), (width, height // 2), (255, 0, 0), 2)

        if results.pose_landmarks and results.face_landmarks:
            return image, shoulder_angle, distance_cm, posture_percentage, distance_offset_cm
        return image, None, None, None, None
