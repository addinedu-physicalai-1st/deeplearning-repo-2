import requests
import json
import os
from shared.schemas import InferenceRequest, InferenceResponse

class NetworkClient:
    def __init__(self, orchestrator_url=None, api_key=None):
        self.orchestrator_url = orchestrator_url or os.getenv("ORCHESTRATOR_URL", "http://localhost:8000/inference")
        self.api_key = api_key or os.getenv("API_KEY")
        
        if not self.api_key:
            print("[!] WARNING: API_KEY is not set. Requests will likely fail.")

    def send_inference_request(self, image_base64: str) -> InferenceResponse:
        payload = InferenceRequest(image_base64=image_base64)
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.post(
                self.orchestrator_url,
                json=payload.dict(),
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            return InferenceResponse(**response.json())
        except Exception as e:
            print(f"Error sending request: {e}")
            return InferenceResponse(is_distracted=False, status_message=f"Error: {str(e)}")
