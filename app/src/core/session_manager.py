from datetime import datetime
from database.database_handler import DatabaseHandler
from vision.detectors import HeadPoseDetector

class SessionManager:
    def __init__(self):
        self.db = DatabaseHandler()
        self.current_detector = None
        self.is_running = False
        self.session_data = {}
        self.start_timestamp = None
        
        # [Fix 3] 상태 변화 감지용 변수
        self.last_distracted_state = False 

    def start_session(self, model_type="HEAD"):
        self.is_running = True
        self.last_distracted_state = False # 초기화
        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "model_type": model_type,
            "distract_cnt": 0,
            "log_json": []
        }
        self.start_timestamp = datetime.now()
        
        if model_type == "HEAD":
            self.current_detector = HeadPoseDetector()
            
        print(f"Session Started: {model_type}")

    def stop_session(self):
        if not self.is_running: return None, {}
        self.is_running = False
        end_time = datetime.now()
        duration_seconds = int((end_time - self.start_timestamp).total_seconds())
        
        # 점수 계산 (단순화된 버전: 이탈 1회당 5점 감점)
        base_score = 100
        penalty = self.session_data["distract_cnt"] * 5
        focus_score = max(0, base_score - penalty)

        final_record = {
            "start_time": self.session_data["start_time"],
            "end_time": end_time.isoformat(),
            "model_type": self.session_data.get("model_type", "HEAD"),
            "duration": duration_seconds,
            "focus_score": focus_score,
            "distract_cnt": self.session_data["distract_cnt"],
            "log_json": self.session_data["log_json"]
        }
        
        # DB 저장
        session_id = self.db.insert_session(final_record)
        final_record['id'] = session_id 
        
        print(f"Session Saved. ID: {session_id}, Score: {focus_score}, Count: {self.session_data['distract_cnt']}")
        return session_id, final_record

    def process_frame(self, frame, sensitivity=20):
        """[Fix 3] 상태 변화 기반 카운팅 로직 적용"""
        if not self.current_detector:
            return frame, "READY", "#808080"
        
        angles, pose_data = self.current_detector.detect(frame)
        status_text = "FOCUS ✅"
        status_color = "#00FF00"
        is_currently_distracted = False

        if angles:
            pitch, yaw, roll = angles
            self.current_detector.draw_debug(frame, pose_data)
            
            # 민감도 기준 체크
            if abs(yaw) > sensitivity or abs(pitch) > (sensitivity * 1.5):
                is_currently_distracted = True
                status_text = "DISTRACTED 🚨"
                status_color = "#FF0000"

        # [핵심 로직] Focus -> Distracted 상태로 변할 때만 카운트 +1
        if is_currently_distracted and not self.last_distracted_state:
            self.increment_distraction()
            print("Distraction Detected! (+1)")
        
        # 상태 업데이트
        self.last_distracted_state = is_currently_distracted
                
        return frame, status_text, status_color

    def increment_distraction(self):
        if self.is_running: 
            self.session_data["distract_cnt"] += 1
