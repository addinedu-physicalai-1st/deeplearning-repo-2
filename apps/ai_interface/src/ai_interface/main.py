import os
import httpx
import asyncio
import logging
import secrets
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import math

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor AI Interface")

# CORS Setup - More restrictive in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_upload_size:
                raise HTTPException(status_code=413, detail="Payload too large (Max 10MB)")
        return await call_next(request)

app.add_middleware(LimitUploadSize, max_upload_size=10_000_000) # 10MB

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("ENVIRONMENT ERROR: API_KEY is not set in .env file.")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# AI Model Server URLs
AI_HEAD_URL = os.getenv("AI_HEAD_URL", "http://localhost:8001/inference")
AI_EMOTION_URL = os.getenv("AI_EMOTION_URL", "http://localhost:8002/inference")
AI_BODY_URL = os.getenv("AI_BODY_URL", "http://localhost:8003/inference")

async def call_ai_server(client: httpx.AsyncClient, url: str, request: InferenceRequest):
    try:
        headers = {API_KEY_NAME: API_KEY}
        response = await client.post(url, json=request.dict(), headers=headers, timeout=2.0)
        if response.status_code == 200:
            return response.json()
        logger.error(f"Server {url} returned {response.status_code}")
        return {"error": "Sub-server error", "status_code": response.status_code}
    except Exception as e:
        logger.error(f"Error calling {url}: {str(e)}") # Log the actual error
        return {"error": "Connection error to sub-server"} # Return a generic message

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_interface"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    async with httpx.AsyncClient(verify=True) as client:
        try:
            # Orchestrate multiple AI models
            tasks = [
                call_ai_server(client, AI_HEAD_URL, request),
                call_ai_server(client, AI_EMOTION_URL, request),
                call_ai_server(client, AI_BODY_URL, request),
                # call_ai_server(client, AI_GAZE_URL, request),
            ]
            
            results = await asyncio.gather(*tasks)
            head_result = results[0]
            emotion_result = results[1]
            body_result = results[2]
            # gaze_result = results[3]

            # Basic Error Handling for the required model
            if "error" in head_result:
                return InferenceResponse(
                    is_distracted=False,
                    status_message=f"AI Sub-server Error", # Hide internal error
                    head_pose=None
                )

            # Error Handling for emotion server (graceful degradation)
            if "error" in emotion_result:
                logger.warning(f"Emotion server error: {emotion_result.get('error')}")
                emotion_result = None

            # Error Handling for body server (graceful degradation)
            if "error" in body_result:
                logger.warning(f"Body server error: {body_result.get('error')}")
                body_result = None

            # Error Handling for gaze server (graceful degradation)
            # if "error" in gaze_result:
            #     logger.warning(f"Gaze server error: {gaze_result.get('error')}")
            #     gaze_result = None

            # Final Orchestration Logic (Rule-based)
            head_distracted = head_result.get("is_distracted", False)
            emotion_distracted = emotion_result.get("is_distracted", False) if emotion_result else False
            body_distracted = body_result.get("is_distracted", False) if body_result else False

            # 셋 중 하나라도 True이면 최종적으로 집중하지 않은 것으로 판단
            is_distracted = head_distracted or emotion_distracted or body_distracted

    ###################################################################################################
            # # Orchestration Logic
            # EMOTION_WEIGHT = 0.3
            # POSITION_WEIGHT = 0.3
            # HEAD_WEIGHT = 0.4

            # MONITOR_DISTANCE_CM = 30
            # MONITOR_WIDTH_CM = 50
            # MONITOR_HEIGHT_CM = 30
            # MONITOR_WIDTH_PX = 1920
            # MONITOR_HEIGHT_PX = 1080

            # emotion_cls = emotion_result.get("emotion", {}).get("emotion", "") if emotion_result else 0
            # emotion_confidence = emotion_result.get("emotion", {}).get("confidence", 0.0) if emotion_result else 0.0
            # if emotion_cls == "Concentrated":
            #     emotion_score = 0.5 + emotion_confidence * 0.5
            # elif emotion_cls == "Distracted":
            #     emotion_score = (1 - emotion_confidence) * 0.5
            # elif emotion_cls == "Sleepy":
            #     emotion_score = 0.0
            # else:
            #     pass

            # position_distance = body_result.get("body_pose", {}).get("distance_cm", 0.0) if body_result else 0.0
            # position_score = max(0, (50 - position_distance) / 50)


            # yaw_rad = math.radians(head_result.get("head_pose", {}).get("yaw", 0.0)) if head_result else 0.0
            # pitch_rad = math.radians(head_result.get("head_pose", {}).get("pitch", 0.0)) if head_result else 0.0

            # gaze_cordinate_x = gaze_result.get("gaze_cordinate", {}).get("x", 0.0) if gaze_result else 0.0
            # gaze_cordinate_y = gaze_result.get("gaze_cordinate", {}).get("y", 0.0) if gaze_result else 0.0
            # head_direction_x = math.sin(yaw_rad) * math.cos(pitch_rad)
            # head_direction_y = -math.sin(pitch_rad)
            # head_direction_z = math.cos(yaw_rad) * math.cos(pitch_rad)

            # if gaze_cordinate_x is not None:
            #     if head_direction_z > 0:
            #         # head 방향 벡터를 30cm 거리의 모니터 평면에 투영 (cm 단위)
            #         t = MONITOR_DISTANCE_CM / head_direction_z
            #         head_point_x_cm = head_direction_x * t
            #         head_point_y_cm = head_direction_y * t
                    
            #         # cm를 픽셀로 변환 (모니터 중심을 원점으로 가정)
            #         head_point_x_px = (head_point_x_cm / MONITOR_WIDTH_CM) * MONITOR_WIDTH_PX + (MONITOR_WIDTH_PX / 2)
            #         head_point_y_px = (head_point_y_cm / MONITOR_HEIGHT_CM) * MONITOR_HEIGHT_PX + (MONITOR_HEIGHT_PX / 2)
            #     else:
            #         head_point_x_px = MONITOR_WIDTH_PX / 2
            #         head_point_y_px = MONITOR_HEIGHT_PX / 2

            #     # gaze_coordinate와 head가 가리키는 좌표 사이의 거리 계산 (픽셀 단위)
            #     dx = gaze_cordinate_x - head_point_x_px
            #     dy = gaze_cordinate_y - head_point_y_px
            #     head_gaze_score = 1 - min(1, math.sqrt(dx * dx + dy * dy) / (MONITOR_HEIGHT_PX / 2))
            # else:
            #     if head_result.get("is_distracted", False):
            #         head_gaze_score = 0.0
            #     else:
            #         head_gaze_score = 1.0

            # is_monitor = 1 if gaze_cordinate_x else 0
            # is_seat = 1 if emotion_result else 0

            # concentration_score = is_monitor * is_seat * (EMOTION_WEIGHT * emotion_score + POSITION_WEIGHT * position_score + HEAD_WEIGHT * head_gaze_score)

    ###################################################################################################

            # 상태 메시지 구성
            if is_distracted:
                messages = []
                if head_distracted:
                    messages.append(head_result.get("status_message", "Head pose issue"))
                if emotion_distracted and emotion_result:
                    messages.append(emotion_result.get("status_message", "Emotion issue"))
                if body_distracted and body_result:
                    messages.append(body_result.get("status_message", "Body posture issue"))
                status_message = " | ".join(messages) if messages else "Distracted"
            else:
                status_message = "Focused"

            return InferenceResponse(
                is_distracted=is_distracted,
                status_message=status_message,
                head_pose=head_result.get("head_pose"),
                emotion=emotion_result.get("emotion") if emotion_result else None,
                body_pose=body_result.get("body_pose") if body_result else None
            )
        except Exception as e:
            logger.exception("Internal Error in AI Interface")
            raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # Changed default port to 8010 to avoid conflict if old orchestrator was on 8000
    # But for now, let's keep it 8000 if that's what the design implied, 
    # OR we can use 8010 and let Operation Server be 8000.
    # Design says Operation Server is the entry point, so Operation Server should be 8000.
    uvicorn.run(app, host="0.0.0.0", port=8010)
