import requests
import json
import os
import logging
from shared.schemas import InferenceRequest, InferenceResponse

# Logging setup
logger = logging.getLogger(__name__)

class NetworkClient:
    def __init__(self, operation_url=None, api_key=None):
        self.operation_url = operation_url or os.getenv("OPERATION_SERVER_URL", "http://localhost:8000/inference")
        self.api_key = api_key or os.getenv("API_KEY")
        
        if not self.api_key:
            logger.warning("API_KEY is not set. Requests will likely fail.")

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

    def start_session(self) -> dict:
        url = self.operation_url.replace("/inference", "/sessions/start")
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(url, headers=headers, timeout=5, verify=True)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return None

    def stop_session(self, session_id: str) -> dict:
        url = self.operation_url.replace("/inference", f"/sessions/stop/{session_id}")
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(url, headers=headers, timeout=5, verify=True)
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
