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

app = FastAPI(title="Focus Monitor AI Emotion Detection")

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

# YOLO Model Initialization (Emotion Detection Model)
# 모델 파일은 apps/ai_emotion/best.pt에 위치해야 함
model = YOLO('best.pt')

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

def encode_frame_to_base64(frame):
    """OpenCV BGR 프레임을 JPEG base64 문자열로 인코딩."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def draw_emotion_boxes(frame, results):
    """YOLO 결과에서 바운딩박스 + 클래스명 + confidence를 프레임에 표시."""
    if not results or len(results) == 0:
        return
    result = results[0]
    if not result.boxes or len(result.boxes) == 0:
        return

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    class_names = result.names

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = confidences[i]
        cls_name = class_names.get(int(classes[i]), f"class_{classes[i]}")

        # 클래스별 색상
        if cls_name.lower() == "concentrated":
            color = (0, 255, 0)      # 초록
        elif cls_name.lower() == "sleepy":
            color = (0, 0, 255)      # 빨강
        else:
            color = (0, 165, 255)    # 주황 (Distracted 등)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def extract_emotion_from_results(results):
    """
    YOLO 모델 결과에서 감정 정보를 추출합니다.
    results: YOLO 모델의 inference 결과
    returns: 감정 정보 딕셔너리 (emotion_name, confidence 등)
    """
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    
    # 박스가 없는 경우
    if not result.boxes or len(result.boxes) == 0:
        return None
    
    # 가장 높은 confidence를 가진 감정 선택
    boxes = result.boxes
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    
    if len(confidences) == 0:
        return None
    
    # 가장 높은 confidence의 감정
    max_idx = np.argmax(confidences)
    emotion_class = int(classes[max_idx])
    confidence = float(confidences[max_idx])
    
    # 클래스 이름 가져오기 (모델의 클래스 이름)
    class_names = result.names
    emotion_name = class_names.get(emotion_class, f"class_{emotion_class}")
    
    return {
        "emotion": emotion_name,
        "confidence": confidence,
        "class_id": emotion_class
    }

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_emotion"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    if not request.image_base64:
        return InferenceResponse(is_distracted=False, status_message="이미지가 제공되지 않았습니다")

    try:
        # Decode base64 image
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return InferenceResponse(is_distracted=False, status_message="잘못된 base64 인코딩입니다")

        # Basic Image Signature Check (Magic Numbers)
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47
        if len(img_data) < 4:
            raise HTTPException(status_code=400, detail="잘못된 이미지 데이터 (너무 짧음)")
        
        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or  # JPEG
            img_data.startswith(b'\x89PNG') or       # PNG
            img_data.startswith(b'RIFF')             # WebP (simplified)
        )
        
        if not is_valid_image:
            logger.warning(f"Potential malicious upload: invalid image signature")
            raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다")

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다")

        # YOLO Emotion Detection (단일 이미지)
        results = model(frame, verbose=False)
        
        # 감정 정보 추출
        emotion_data = extract_emotion_from_results(results)
        
        # 감정 기반 집중도 판단
        is_distracted = False
        status_message = "정상"
        
        if emotion_data:
            emotion_name = emotion_data.get("emotion", "").lower()
            confidence = emotion_data.get("confidence", 0.0)
            
            # 산만한 감정 목록
            distracted_emotions = [ "distracted", "sleepy"]
            
            # 감정명을 status_message에 직접 표시
            status_message = emotion_data['emotion']
            
            # 산만한 감정이 감지되면 집중하지 않은 것으로 판단
            if emotion_name in distracted_emotions:
                is_distracted = True
            
            logger.info(f"[EMOTION] {emotion_data['emotion']} | Confidence: {confidence:.2f} | Distracted: {is_distracted}")
        else:
            status_message = "감정을 감지할 수 없습니다"
            is_distracted = True

        return InferenceResponse(
            is_distracted=is_distracted,
            status_message=status_message,
            emotion=emotion_data
        )

    except Exception as e:
        logger.exception(f"[!] Inference Error: {e}")
        return InferenceResponse(
            is_distracted=False, 
            status_message=f"서버 오류가 발생했습니다"
        )

@app.post("/debug_inference")
async def debug_inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    """디버그용: 추론 결과 + 바운딩박스가 그려진 어노테이션 이미지를 반환."""
    if not request.image_base64:
        return {"data": {"emotion": None, "confidence": 0, "is_distracted": False}, "annotated_image": None}

    try:
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return {"data": {"emotion": None, "confidence": 0, "is_distracted": False}, "annotated_image": None}

        if len(img_data) < 4:
            return {"data": {"emotion": None, "confidence": 0, "is_distracted": False}, "annotated_image": None}

        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or
            img_data.startswith(b'\x89PNG') or
            img_data.startswith(b'RIFF')
        )
        if not is_valid_image:
            return {"data": {"emotion": None, "confidence": 0, "is_distracted": False}, "annotated_image": None}

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"data": {"emotion": None, "confidence": 0, "is_distracted": False}, "annotated_image": None}

        # 어노테이션은 원본 컬러 프레임에 그림
        debug_frame = frame.copy()

        # 추론은 기존과 동일하게 그레이스케일 변환 후 진행
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        results = model(gray_frame, verbose=False)

        # 바운딩박스 + 클래스 어노테이션 그리기
        draw_emotion_boxes(debug_frame, results)

        # 감정 정보 추출
        emotion_data = extract_emotion_from_results(results)
        is_distracted = False
        if emotion_data:
            emotion_name = emotion_data.get("emotion", "").lower()
            distracted_emotions = ["distracted", "sleepy"]
            if emotion_name in distracted_emotions:
                is_distracted = True

            # 상태 텍스트 표시
            color = (0, 0, 255) if is_distracted else (0, 255, 0)
            status_text = f"{emotion_data['emotion']} | {'Distracted' if is_distracted else 'Focused'}"
            cv2.putText(debug_frame, status_text, (10, debug_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(debug_frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return {
            "data": {
                "emotion": emotion_data.get("emotion") if emotion_data else None,
                "confidence": emotion_data.get("confidence", 0) if emotion_data else 0,
                "class_id": emotion_data.get("class_id") if emotion_data else None,
                "is_distracted": is_distracted
            },
            "annotated_image": encode_frame_to_base64(debug_frame)
        }

    except Exception as e:
        logger.exception(f"[!] Debug Inference Error: {e}")
        return {"data": {"emotion": None, "confidence": 0, "is_distracted": False}, "annotated_image": None}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
