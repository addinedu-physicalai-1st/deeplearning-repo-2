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
import mediapipe as mp
from pydantic import BaseModel
from typing import List

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor AI Gaze Tracking")

# CORS Setup
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

# Middleware: Limit Request Size (10MB)
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

app.add_middleware(LimitUploadSize, max_upload_size=10_000_000)  # 10MB

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# ─── MediaPipe Face Mesh (iris 랜드마크 포함) ───
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # iris 랜드마크(468-477) 활성화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─── 캘리브레이션 상태 (전역, 단일 사용자) ───
calibration_model = None
calibration_bounds = None

# ─── Iris 랜드마크 인덱스 ───
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


def extract_iris_position(landmarks, image_width, image_height):
    """
    양쪽 눈의 iris 중심을 eye contour 대비 정규화 좌표(0~1)로 반환.
    0 = iris가 눈의 왼쪽/위쪽 끝, 1 = 오른쪽/아래쪽 끝.
    """
    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * image_width, lm.y * image_height])

    def normalize_iris(iris_indices, inner_idx, outer_idx, top_idx, bottom_idx):
        iris_center = np.mean([get_point(i) for i in iris_indices], axis=0)
        inner = get_point(inner_idx)
        outer = get_point(outer_idx)
        top = get_point(top_idx)
        bottom = get_point(bottom_idx)

        eye_width = np.linalg.norm(outer - inner)
        eye_height = np.linalg.norm(bottom - top)
        if eye_width < 1 or eye_height < 1:
            return 0.5, 0.5

        eye_dir = (outer - inner) / eye_width
        iris_vec = iris_center - inner
        x_ratio = np.dot(iris_vec, eye_dir) / eye_width

        eye_vdir = (bottom - top) / eye_height
        y_ratio = np.dot(iris_center - top, eye_vdir) / eye_height

        return float(np.clip(x_ratio, 0, 1)), float(np.clip(y_ratio, 0, 1))

    lx, ly = normalize_iris(LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
    rx, ry = normalize_iris(RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

    return (lx + rx) / 2, (ly + ry) / 2


# ─── 캘리브레이션 스키마 ───
class CalibrationPoint(BaseModel):
    iris_x: float
    iris_y: float
    screen_x: float
    screen_y: float

class CalibrationRequest(BaseModel):
    points: List[CalibrationPoint]
    screen_width: int
    screen_height: int


@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_gaze"}


@app.post("/set_calibration")
async def set_calibration(request: CalibrationRequest, api_key: str = Depends(get_api_key)):
    """
    9-point 캘리브레이션 데이터를 받아 iris→screen 좌표 매핑 모델 생성.
    외부 WebGazer 코드의 캘리브레이션 개념을 Python으로 구현.
    """
    global calibration_model, calibration_bounds

    if len(request.points) < 9:
        raise HTTPException(status_code=400, detail="최소 9개의 캘리브레이션 포인트가 필요합니다")

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge

    iris_coords = np.array([[p.iris_x, p.iris_y] for p in request.points])
    screen_coords = np.array([[p.screen_x, p.screen_y] for p in request.points])

    poly = PolynomialFeatures(degree=2, include_bias=True)
    iris_features = poly.fit_transform(iris_coords)

    model_x = Ridge(alpha=1.0)
    model_x.fit(iris_features, screen_coords[:, 0])

    model_y = Ridge(alpha=1.0)
    model_y.fit(iris_features, screen_coords[:, 1])

    calibration_model = {
        "poly": poly,
        "model_x": model_x,
        "model_y": model_y,
    }
    calibration_bounds = {
        "left": 0,
        "right": request.screen_width,
        "top": 0,
        "bottom": request.screen_height,
    }

    logger.info(f"캘리브레이션 설정 완료: {len(request.points)}개 포인트, 화면 {request.screen_width}x{request.screen_height}")
    return {"status": "ok", "points_count": len(request.points)}


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    if not request.image_base64:
        return InferenceResponse(is_distracted=False, status_message="이미지가 제공되지 않았습니다")

    try:
        # 이미지 디코딩 (ai_body 패턴)
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return InferenceResponse(is_distracted=False, status_message="잘못된 base64 인코딩입니다")

        if len(img_data) < 4:
            raise HTTPException(status_code=400, detail="잘못된 이미지 데이터 (너무 짧음)")

        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or  # JPEG
            img_data.startswith(b'\x89PNG') or        # PNG
            img_data.startswith(b'RIFF')              # WebP
        )
        if not is_valid_image:
            logger.warning("Potential malicious upload: invalid image signature")
            raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다")

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다")

        # Face Mesh로 iris 위치 추출
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return InferenceResponse(
                is_distracted=True,
                status_message="얼굴을 감지할 수 없습니다",
                gaze_data={"calibrated": calibration_model is not None, "is_on_screen": None}
            )

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        iris_x, iris_y = extract_iris_position(face_landmarks.landmark, w, h)

        gaze_data = {
            "iris_x": round(iris_x, 4),
            "iris_y": round(iris_y, 4),
            "gaze_x": None,
            "gaze_y": None,
            "is_on_screen": None,
            "calibrated": calibration_model is not None,
            "gaze_cordinate": {"x": None, "y": None},
        }

        is_distracted = False
        status_message = "시선 감지됨 (캘리브레이션 필요)"

        if calibration_model is not None:
            # iris 위치 → 화면 좌표 매핑
            poly = calibration_model["poly"]
            iris_input = poly.transform(np.array([[iris_x, iris_y]]))
            gaze_screen_x = float(calibration_model["model_x"].predict(iris_input)[0])
            gaze_screen_y = float(calibration_model["model_y"].predict(iris_input)[0])

            gaze_data["gaze_x"] = round(gaze_screen_x, 1)
            gaze_data["gaze_y"] = round(gaze_screen_y, 1)
            gaze_data["gaze_cordinate"] = {
                "x": round(gaze_screen_x, 1),
                "y": round(gaze_screen_y, 1),
            }

            # 화면 영역 내인지 판정 (외부 코드의 isOut 로직)
            bounds = calibration_bounds
            is_on_screen = (
                bounds["left"] <= gaze_screen_x <= bounds["right"] and
                bounds["top"] <= gaze_screen_y <= bounds["bottom"]
            )
            gaze_data["is_on_screen"] = is_on_screen

            if not is_on_screen:
                is_distracted = True
                # 이탈 방향 분류 (외부 코드의 outDirection 로직)
                if gaze_screen_x < bounds["left"]:
                    status_message = "시선 이탈 (좌측)"
                elif gaze_screen_x > bounds["right"]:
                    status_message = "시선 이탈 (우측)"
                elif gaze_screen_y < bounds["top"]:
                    status_message = "시선 이탈 (상단)"
                elif gaze_screen_y > bounds["bottom"]:
                    status_message = "시선 이탈 (하단)"
                else:
                    status_message = "시선 이탈"
            else:
                status_message = "화면 응시 중"

        logger.info(
            f"[GAZE] iris=({iris_x:.3f},{iris_y:.3f}) "
            f"screen=({gaze_data.get('gaze_x')},{gaze_data.get('gaze_y')}) "
            f"on_screen={gaze_data.get('is_on_screen')}"
        )

        return InferenceResponse(
            is_distracted=is_distracted,
            status_message=status_message,
            gaze_data=gaze_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[!] Inference Error: {e}")
        return InferenceResponse(
            is_distracted=False,
            status_message="서버 오류가 발생했습니다"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)
