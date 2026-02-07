import os
import cv2
import numpy as np
import base64
import logging
import secrets
import math
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor AI Body Posture")

# CORS Setup - More restrictive in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY = os.getenv("API_KEY")
if not API_KEY:
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

# MediaPipe Holistic 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 정자세 기준값 (세션별로 관리하지 않고 전역으로 사용)
baseline_distance = None  # 거북목 측정용
baseline_distance_cm = None  # 앞뒤 이동거리 측정용

# 거북목 판단 임시 기준: posture_percentage가 이 값 미만이면 비집중 (나중에 수치 조정)
POSTURE_DISTRACTED_THRESHOLD = float(os.getenv("POSTURE_DISTRACTED_THRESHOLD", "70"))

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

def process_posture(frame):
    """
    MediaPipe Holistic을 사용하여 자세 분석
    Returns: shoulder_angle, distance_cm, posture_percentage, distance_offset_cm
    """
    global baseline_distance, baseline_distance_cm
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    
    height, width, _ = image.shape
    shoulder_angle = None
    distance_cm = None
    posture_percentage = None
    distance_offset_cm = None
    
    if results.face_landmarks and results.pose_landmarks:
        # 어깨 좌표로 어깨 기울기 계산
        pose_landmarks = results.pose_landmarks.landmark
        left_sh = pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_sh = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        
        lx, ly = left_sh.x * width, left_sh.y * height
        rx, ry = right_sh.x * width, right_sh.y * height
        dx = rx - lx
        dy = ry - ly
        
        # 어깨 각도 계산 (수평선 기준)
        shoulder_angle = math.degrees(math.atan2(dy, dx))
        shoulder_angle = abs(shoulder_angle)
        
        # 어깨 폭(픽셀)으로 카메라와의 거리 근사 계산
        pixel_shoulder_dist = math.hypot(dx, dy)
        REAL_SHOULDER_CM = 40.0  # 실제 어깨너비 (cm)
        FOCAL_PX = width * 1.2  # 근사 초점 거리
        
        if pixel_shoulder_dist > 0:
            distance_cm = (REAL_SHOULDER_CM * FOCAL_PX) / pixel_shoulder_dist
            
            # 정자세 기준값 대비 계산
            if baseline_distance is not None and distance_cm is not None:
                # 거북목 측정: 거리 감소량을 계산
                distance_decrease = max(0, baseline_distance - distance_cm)
                decrease_percentage = (distance_decrease / baseline_distance) * 30
                posture_percentage = 100 - decrease_percentage
                posture_percentage = max(0, min(100, posture_percentage))
            
            # 앞뒤 이동거리 측정
            if baseline_distance_cm is not None:
                distance_offset_cm = distance_cm - baseline_distance_cm
    
    return shoulder_angle, distance_cm, posture_percentage, distance_offset_cm

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_body"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    global baseline_distance, baseline_distance_cm
    
    if not request.image_base64:
        return InferenceResponse(is_distracted=False, status_message="이미지가 제공되지 않았습니다")
    
    try:
        # Decode base64 image
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return InferenceResponse(is_distracted=False, status_message="잘못된 base64 인코딩입니다")
        
        # Basic Image Signature Check
        if len(img_data) < 4:
            raise HTTPException(status_code=400, detail="잘못된 이미지 데이터 (너무 짧음)")
        
        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or  # JPEG
            img_data.startswith(b'\x89PNG') or       # PNG
            img_data.startswith(b'RIFF')             # WebP
        )
        
        if not is_valid_image:
            logger.warning("Potential malicious upload: invalid image signature")
            raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다")
        
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다")
        
        # 자세 분석
        shoulder_angle, distance_cm, posture_percentage, distance_offset_cm = process_posture(frame)
        
        # 거북목 판단: 비집중으로는 하지 않고, posture_alert만 body_pose에 넣어 GUI에서 노란 경고·경고음
        is_distracted = False
        status_message = "정상"

        if distance_cm is None:
            status_message = "사람을 감지할 수 없습니다"

        # body_pose: 현재 프레임 거리 + 정자세 기준값 + 거북목 시 posture_alert
        body_pose = None
        if distance_cm is not None or baseline_distance is not None or (
            posture_percentage is not None and posture_percentage < POSTURE_DISTRACTED_THRESHOLD
        ):
            body_pose = {}
            if distance_cm is not None:
                body_pose["distance_cm"] = float(distance_cm)
            if baseline_distance is not None:
                body_pose["baseline_distance_cm"] = float(baseline_distance)
            if shoulder_angle is not None:
                body_pose["shoulder_angle"] = float(shoulder_angle)
            if posture_percentage is not None:
                body_pose["posture_percentage"] = float(posture_percentage)
            if distance_offset_cm is not None:
                body_pose["distance_offset_cm"] = float(distance_offset_cm)
            if posture_percentage is not None and posture_percentage < POSTURE_DISTRACTED_THRESHOLD:
                body_pose["posture_alert"] = True

        d_str = f"{distance_cm:.1f}" if distance_cm is not None else "None"
        b_str = f"{baseline_distance:.1f}" if baseline_distance is not None else "None"
        alert = body_pose.get("posture_alert", False) if body_pose else False
        logger.info(f"[BODY] distance_cm={d_str} baseline_distance_cm={b_str} posture%={posture_percentage} posture_alert={alert}")

        return InferenceResponse(
            is_distracted=is_distracted,
            status_message=status_message,
            body_pose=body_pose
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[!] Inference Error: {e}")
        return InferenceResponse(
            is_distracted=False,
            status_message=f"서버 오류가 발생했습니다"
        )

@app.post("/set_baseline")
async def set_baseline(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    """
    정자세 기준값 설정
    현재 프레임의 거리를 기준으로 설정합니다.
    """
    global baseline_distance, baseline_distance_cm
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="이미지가 제공되지 않았습니다")
    
    try:
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다")
        
        _, distance_cm, _, _ = process_posture(frame)
        
        if distance_cm is not None:
            baseline_distance = distance_cm
            baseline_distance_cm = distance_cm
            logger.info(f"정자세 기준값 설정: {distance_cm:.1f}cm")
            return {"status": "ok", "baseline_distance_cm": distance_cm}
        else:
            raise HTTPException(status_code=400, detail="거리를 감지할 수 없습니다")
    
    except Exception as e:
        logger.exception(f"Baseline 설정 오류: {e}")
        raise HTTPException(status_code=500, detail="기준값 설정 실패")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
