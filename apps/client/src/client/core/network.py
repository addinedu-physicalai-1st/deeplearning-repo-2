import requests
import json
import os
import logging
from shared.schemas import InferenceRequest, InferenceResponse

# Logging setup
logger = logging.getLogger(__name__)

class NetworkClient:
    def __init__(self, operation_url=None, api_key=None, ai_body_url=None, ai_gaze_url=None):
        self.operation_url = operation_url or os.getenv("OPERATION_SERVER_URL", "http://localhost:8000/inference")
        self.api_key = api_key or os.getenv("API_KEY")
        self.ai_body_url = (ai_body_url or os.getenv("AI_BODY_URL", "http://localhost:8003")).rstrip("/")
        self.ai_gaze_url = (ai_gaze_url or os.getenv("AI_GAZE_URL", "http://localhost:8005")).rstrip("/")

        if not self.api_key:
            logger.warning("API_KEY is not set. Requests will likely fail.")

    def register(self, username: str, password: str, display_name: str) -> dict:
        url = self.operation_url.replace("/inference", "/auth/register")
        try:
            response = requests.post(
                url,
                json={"username": username, "password": password, "display_name": display_name},
                timeout=5, verify=True
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                return {"error": "duplicate", "detail": e.response.json().get("detail", "")}
            logger.error(f"Error registering: {e}")
            return None
        except Exception as e:
            logger.error(f"Error registering: {e}")
            return None

    def login(self, username: str, password: str) -> dict:
        url = self.operation_url.replace("/inference", "/auth/login")
        try:
            response = requests.post(
                url,
                json={"username": username, "password": password},
                timeout=5, verify=True
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                return {"error": "invalid_credentials", "detail": e.response.json().get("detail", "")}
            logger.error(f"Error logging in: {e}")
            return None
        except Exception as e:
            logger.error(f"Error logging in: {e}")
            return None

    def send_inference_request(self, image_base64: str, session_id: str = None) -> InferenceResponse:
        payload = InferenceRequest(image_base64=image_base64, session_id=session_id)
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(
                self.operation_url,
                json=payload.dict(),
                headers=headers,
                timeout=5,
                verify=True
            )
            response.raise_for_status()
            return InferenceResponse(**response.json())
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return InferenceResponse(is_distracted=False, status_message="Connection Error")

    def start_session(self, user_id: int = None) -> dict:
        url = self.operation_url.replace("/inference", "/sessions/start")
        headers = {"X-API-Key": self.api_key}
        payload = {}
        if user_id is not None:
            payload["user_id"] = user_id
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5, verify=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return None

    def stop_session(self, session_id: str) -> dict:
        url = self.operation_url.replace("/inference", f"/sessions/stop/{session_id}")
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(url, headers=headers, timeout=10, verify=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error stopping session: {e}")
            return None

    def check_connection(self) -> bool:
        try:
            # Replace /inference with /health for health check
            health_url = self.operation_url.replace("/inference", "/health")
            headers = {"X-API-Key": self.api_key}
            response = requests.get(health_url, headers=headers, timeout=2, verify=True)
            return response.status_code == 200
        except:
            return False

    def get_history(self, user_id: int = None) -> list:
        url = self.operation_url.replace("/inference", "/sessions")
        headers = {"X-API-Key": self.api_key}
        params = {}
        if user_id is not None:
            params["user_id"] = user_id
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5, verify=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return []

    def get_session_logs(self, session_id: str) -> dict:
        """세션의 로그 데이터를 가져옴"""
        url = self.operation_url.replace("/inference", f"/sessions/{session_id}/logs")
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching session logs: {e}")
            return None

    def request_llm_feedback(self, session_id: str) -> dict:
        url = self.operation_url.replace("/inference", f"/sessions/{session_id}/feedback")
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(url, headers=headers, timeout=60, verify=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error requesting LLM feedback: {e}")
            return None

    def set_gaze_calibration(self, calibration_points: list, screen_width: int, screen_height: int) -> bool:
        """POST calibration data to ai_gaze /set_calibration."""
        url = f"{self.ai_gaze_url}/set_calibration"
        payload = {
            "points": calibration_points,
            "screen_width": screen_width,
            "screen_height": screen_height,
        }
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5, verify=True)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending gaze calibration: {e}")
            return False

    def set_baseline(self, image_base64: str) -> bool:
        """POST current frame to ai_body /set_baseline. Returns True on success."""
        url = f"{self.ai_body_url}/set_baseline"
        payload = InferenceRequest(image_base64=image_base64, session_id=None)
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload.dict(), headers=headers, timeout=5, verify=True)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending set_baseline to ai_body: {e}")
            return False
