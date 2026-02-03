import os
import cv2
import numpy as np
import base64
import logging
import secrets
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor AI Head Pose")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # 프로덕션/팀 프로젝트에서 키가 없는 상태로 구동되는 것을 방지
    raise RuntimeError("ENVIRONMENT ERROR: API_KEY is not set in .env file.")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Middleware: Limit Request Size (10MB) to prevent DoS
from fastapi import Request

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

# YOLO Model Initialization (YOLOv8 Pose)
model = YOLO('yolov8n-pose.pt')

# Target Tracking State
target_id = None

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

def estimate_pose(keypoints_normalized):
    """
    keypoints_normalized: [17, 2] array of keypoints (x, y) in normalized [0, 1] range
    0: nose, 1: left_eye, 2: right_eye
    """
    nose = keypoints_normalized[0]
    l_eye = keypoints_normalized[1]
    r_eye = keypoints_normalized[2]

    # Check if keypoints are visible
    if np.all(nose == 0) or np.all(l_eye == 0) or np.all(r_eye == 0):
        return 0, 0, 0

    # Yaw (Left/Right): 코가 두 눈 중심에서 얼마나 벗어났는지
    eye_center_x = (l_eye[0] + r_eye[0]) / 2
    eye_width = max(abs(l_eye[0] - r_eye[0]), 0.01)
    yaw = (nose[0] - eye_center_x) / eye_width * 120 

    # Pitch (Up/Down): 코와 눈의 수직 거리 기반
    # 기본적으로 정면을 볼 때 코는 눈보다 약간 아래에 위치함 (Offset 적용)
    eye_center_y = (l_eye[1] + r_eye[1]) / 2
    pitch_raw = (nose[1] - eye_center_y) / eye_width
    neutral_offset = 0.45  # 정면 응시 시 코의 일반적인 수직 위치 오프셋
    pitch = (pitch_raw - neutral_offset) * 120 

    # Roll (Tilt)
    roll = np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]) * 180 / np.pi

    return pitch, yaw, roll

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_head"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    global target_id
    
    if not request.image_base64:
        return InferenceResponse(is_distracted=False, status_message="No image provided")

    try:
        # Decode base64 image
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return InferenceResponse(is_distracted=False, status_message="Invalid base64 encoding")

        # Basic Image Signature Check (Magic Numbers)
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47
        if len(img_data) < 4:
            raise HTTPException(status_code=400, detail="Invalid image data (too short)")
        
        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or  # JPEG
            img_data.startswith(b'\x89PNG') or       # PNG
            img_data.startswith(b'RIFF')             # WebP (simplified)
        )
        
        if not is_valid_image:
            logger.warning(f"Potential malicious upload: invalid image signature")
            raise HTTPException(status_code=400, detail="Unsupported image format")

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # YOLO Tracking
        results = model.track(frame, persist=True, verbose=False)
        
        if not results[0].boxes or results[0].boxes.id is None:
            return InferenceResponse(
                is_distracted=True, 
                status_message="No person detected"
            )

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.xyn.cpu().numpy()

        # Target Selection Logic
        img_h, img_w = frame.shape[:2]
        centers = boxes.xywh.cpu().numpy()[:, :2]
        center_points = np.array([img_w/2, img_h/2])
        distances = np.linalg.norm(centers - center_points, axis=1)

        if target_id is None or target_id not in ids:
            target_idx = np.argmin(distances)
            target_id = int(ids[target_idx])
            logger.info(f"[*] New Target Locked: ID {target_id}")

        try:
            current_idx = list(ids).index(target_id)
        except ValueError:
            return InferenceResponse(is_distracted=True, status_message="Target lost")

        # Pose Estimation
        target_kpts = keypoints_data[current_idx]
        pitch, yaw, roll = estimate_pose(target_kpts)

        # Thresholds (실용적인 범위로 재조정)
        YAW_LIMIT = 30
        PITCH_UP_LIMIT = -25
        PITCH_DOWN_LIMIT = 30
        
        is_distracted = False
        reason = ""

        if pitch == 0 and yaw == 0:
            is_distracted = False
            status_message = "Focused (Searching...)"
        else:
            if abs(yaw) > YAW_LIMIT:
                is_distracted = True
                reason = "Looking Away (Side)"
            elif pitch < PITCH_UP_LIMIT:
                is_distracted = True
                reason = "Looking Up"
            elif pitch > PITCH_DOWN_LIMIT:
                is_distracted = True
                reason = "Looking Down"

            status_message = "Focused" if not is_distracted else f"Distracted ({reason})"

        # Logging for debugging
        log_status = "DISTRACTED" if is_distracted else "FOCUSED"
        logger.info(f"[{log_status}] ID:{target_id} | Pitch:{pitch:4.1f} | Yaw:{yaw:4.1f} | Roll:{roll:4.1f} | {reason}")

        return InferenceResponse(
            is_distracted=is_distracted,
            status_message=status_message,
            head_pose={
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll)
            }
        )

    except Exception as e:
        logger.exception(f"[!] Inference Error: {e}")
        return InferenceResponse(
            is_distracted=False, 
            status_message=f"Server Error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
