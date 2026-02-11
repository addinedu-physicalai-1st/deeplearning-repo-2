import os
import json
import cv2
import numpy as np
import base64
import logging
import secrets
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Body
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

# YOLO Model Initialization (YOLOv8 Pose)
model = YOLO('yolov8n-pose.pt')

# Target Tracking State
target_id = None

# Head Pose Thresholds (캘리브레이션으로 설정 가능, 파일에서 로드)
PITCH_UP_LIMIT = -25
PITCH_DOWN_LIMIT = 30
YAW_LIMIT = 30

# ===== solvePnP 기반 Head Pose 추정을 위한 3D 얼굴 모델 =====
# COCO 얼굴 키포인트의 근사 3D 위치 (mm 단위, 코 끝 기준)
# 좌표계: X=이미지 오른쪽, Y=아래쪽, Z=깊이(카메라에서 멀어지는 방향)
FACE_MODEL_3D = np.array([
    [0.0,    0.0,   0.0],      # 0: Nose tip
    [33.0, -33.0,  27.0],      # 1: Left eye (인물 왼쪽 → 정면 시 이미지 오른쪽)
    [-33.0, -33.0,  27.0],     # 2: Right eye (인물 오른쪽 → 정면 시 이미지 왼쪽)
    [77.0, -15.0,  67.0],      # 3: Left ear
    [-77.0, -15.0,  67.0],     # 4: Right ear
], dtype=np.float64)
FACE_KPT_INDICES = [0, 1, 2, 3, 4]
MIN_KPT_CONF = 0.3
MIN_PNP_POINTS = 4


def _get_camera_matrix(img_w, img_h):
    """이미지 크기로부터 근사 카메라 내부 행렬 생성."""
    focal_length = float(img_w)
    return np.array([
        [focal_length, 0, img_w / 2.0],
        [0, focal_length, img_h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

THRESHOLDS_PATH = Path(__file__).resolve().parent.parent.parent / "thresholds.json"


def _load_thresholds_from_file():
    """저장된 임계값이 있으면 전역 변수를 갱신 (기동 시 호출)."""
    global PITCH_UP_LIMIT, PITCH_DOWN_LIMIT, YAW_LIMIT
    if not THRESHOLDS_PATH.exists():
        return
    try:
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        PITCH_UP_LIMIT = float(data.get("pitch_up_limit", PITCH_UP_LIMIT))
        PITCH_DOWN_LIMIT = float(data.get("pitch_down_limit", PITCH_DOWN_LIMIT))
        YAW_LIMIT = float(data.get("yaw_limit", YAW_LIMIT))
        logger.info(f"임계값 로드: pitch_up={PITCH_UP_LIMIT}, pitch_down={PITCH_DOWN_LIMIT}, yaw={YAW_LIMIT}")
    except (json.JSONDecodeError, TypeError, OSError) as e:
        logger.warning(f"임계값 파일 로드 실패, 기본값 사용: {e}")


def _save_thresholds_to_file():
    """현재 전역 임계값을 파일에 저장."""
    try:
        with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"pitch_up_limit": PITCH_UP_LIMIT, "pitch_down_limit": PITCH_DOWN_LIMIT, "yaw_limit": YAW_LIMIT},
                f,
                indent=2,
            )
    except OSError as e:
        logger.warning(f"임계값 파일 저장 실패: {e}")


_load_thresholds_from_file()

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


def draw_pose_axes(frame, nose_px, pitch, yaw, roll, axis_length=80,
                   rvec=None, tvec=None, camera_matrix=None):
    """코(nose) 위치에서 3D 축을 화살표로 표시."""
    if rvec is not None and tvec is not None and camera_matrix is not None:
        # solvePnP 결과를 이용한 정확한 3D 축 투영
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        axis_len = 60.0  # mm (3D 모델과 동일 단위)
        axis_points = np.array([
            [0, 0, 0],                # Origin (nose)
            [axis_len, 0, 0],          # X axis (red) - 오른쪽
            [0, axis_len, 0],          # Y axis (green) - 아래쪽
            [0, 0, -axis_len],         # Z axis (blue) - 정면 방향
        ], dtype=np.float64)

        projected, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, dist_coeffs
        )

        origin = tuple(projected[0].ravel().astype(int))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(3):
            end = tuple(projected[i + 1].ravel().astype(int))
            cv2.arrowedLine(frame, origin, end, colors[i], 3, tipLength=0.2)
    else:
        # Fallback: pitch/yaw/roll 기반 3D 축 재구성
        import math
        pitch_r = math.radians(-pitch)
        yaw_r = math.radians(yaw)
        roll_r = math.radians(roll)

        cos_y, sin_y = math.cos(yaw_r), math.sin(yaw_r)
        cos_p, sin_p = math.cos(pitch_r), math.sin(pitch_r)
        cos_r, sin_r = math.cos(roll_r), math.sin(roll_r)

        origin = (int(nose_px[0]), int(nose_px[1]))
        x_end = (
            int(nose_px[0] + axis_length * (cos_y * cos_r + sin_y * sin_p * sin_r)),
            int(nose_px[1] + axis_length * (cos_p * sin_r))
        )
        y_end = (
            int(nose_px[0] + axis_length * (-cos_y * sin_r + sin_y * sin_p * cos_r)),
            int(nose_px[1] + axis_length * (cos_p * cos_r))
        )
        z_end = (
            int(nose_px[0] + axis_length * (sin_y * cos_p)),
            int(nose_px[1] - axis_length * sin_p)
        )
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 3, tipLength=0.2)
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 3, tipLength=0.2)
        cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 3, tipLength=0.2)

    # 각도 텍스트 표시
    cv2.putText(frame, f"P:{pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Y:{yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"R:{roll:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def estimate_pose(keypoints_norm, keypoints_conf, img_w, img_h):
    """
    solvePnP 기반 머리 포즈 추정.
    코·눈·귀 키포인트의 3D 모델과 2D 검출 좌표를 매칭하여 정확한 포즈를 계산한다.
    측면(프로필) 뷰에서도 귀 키포인트 덕분에 안정적으로 동작.

    Args:
        keypoints_norm: [17, 2] 정규화된 키포인트 좌표 (0~1)
        keypoints_conf: [17] 키포인트 신뢰도 배열 (또는 None)
        img_w, img_h: 이미지 크기 (픽셀)

    Returns:
        (pitch, yaw, roll, rvec, tvec, camera_matrix)
        추정 실패 시 (0, 0, 0, None, None, None)
    """
    # 신뢰도 높은 키포인트만 선별하여 2D-3D 대응 구축
    pts_2d = []
    pts_3d = []
    for i, kpt_idx in enumerate(FACE_KPT_INDICES):
        conf = float(keypoints_conf[kpt_idx]) if keypoints_conf is not None else 1.0
        if conf < MIN_KPT_CONF:
            continue
        x_n, y_n = keypoints_norm[kpt_idx]
        if x_n == 0 and y_n == 0:
            continue
        pts_2d.append([x_n * img_w, y_n * img_h])
        pts_3d.append(FACE_MODEL_3D[i])

    if len(pts_2d) < MIN_PNP_POINTS:
        return 0, 0, 0, None, None, None

    pts_2d = np.array(pts_2d, dtype=np.float64)
    pts_3d = np.array(pts_3d, dtype=np.float64)

    camera_matrix = _get_camera_matrix(img_w, img_h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP
    )

    if not success:
        return 0, 0, 0, None, None, None

    # 회전 벡터 → 회전 행렬
    R, _ = cv2.Rodrigues(rvec)

    # 모델 좌표계에서 얼굴 정면 방향은 -Z, 위쪽 방향은 -Y
    forward = R @ np.array([0.0, 0.0, -1.0])
    up = R @ np.array([0.0, -1.0, 0.0])

    # Yaw: 좌우 회전 (양수 = 이미지 오른쪽, 음수 = 이미지 왼쪽)
    yaw = np.degrees(np.arctan2(forward[0], -forward[2]))

    # Pitch: 상하 회전 (양수 = 아래쪽, 음수 = 위쪽)
    pitch = np.degrees(np.arctan2(forward[1],
                                   np.sqrt(forward[0]**2 + forward[2]**2)))

    # Roll: 머리 기울기
    roll = np.degrees(np.arctan2(up[0], -up[1]))

    return pitch, yaw, roll, rvec, tvec, camera_matrix

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_head"}

@app.get("/thresholds")
async def get_thresholds(api_key: str = Depends(get_api_key)):
    """현재 설정된 임계값 조회"""
    global PITCH_UP_LIMIT, PITCH_DOWN_LIMIT, YAW_LIMIT
    return {
        "pitch_up_limit": PITCH_UP_LIMIT,
        "pitch_down_limit": PITCH_DOWN_LIMIT,
        "yaw_limit": YAW_LIMIT
    }

@app.post("/set_thresholds")
async def set_thresholds(
    thresholds: dict = Body(...),
    api_key: str = Depends(get_api_key)
):
    """머리 각도 임계값 설정"""
    global PITCH_UP_LIMIT, PITCH_DOWN_LIMIT, YAW_LIMIT

    pitch_up_limit = thresholds.get("pitch_up_limit")
    pitch_down_limit = thresholds.get("pitch_down_limit")
    yaw_limit = thresholds.get("yaw_limit")

    if pitch_up_limit is None or pitch_down_limit is None or yaw_limit is None:
        raise HTTPException(status_code=400, detail="pitch_up_limit, pitch_down_limit, yaw_limit이 모두 필요합니다")
    if pitch_up_limit >= 0 or pitch_down_limit <= 0:
        raise HTTPException(status_code=400, detail="pitch_up_limit은 음수, pitch_down_limit은 양수여야 합니다")
    if yaw_limit <= 0:
        raise HTTPException(status_code=400, detail="yaw_limit은 양수여야 합니다")

    PITCH_UP_LIMIT = float(pitch_up_limit)
    PITCH_DOWN_LIMIT = float(pitch_down_limit)
    YAW_LIMIT = float(yaw_limit)

    _save_thresholds_to_file()
    logger.info(f"임계값 설정 및 저장: pitch_up={PITCH_UP_LIMIT}, pitch_down={PITCH_DOWN_LIMIT}, yaw={YAW_LIMIT}")

    return {
        "status": "ok",
        "pitch_up_limit": PITCH_UP_LIMIT,
        "pitch_down_limit": PITCH_DOWN_LIMIT,
        "yaw_limit": YAW_LIMIT
    }

@app.post("/pose")
async def get_pose_only(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    """캘리브레이션 전용: 집중 판단 없이 pitch, yaw, roll만 반환."""
    global target_id

    if not request.image_base64:
        return {"head_pose": None}

    try:
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return {"head_pose": None}
        if len(img_data) < 4:
            return {"head_pose": None}
        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or
            img_data.startswith(b'\x89PNG') or
            img_data.startswith(b'RIFF')
        )
        if not is_valid_image:
            return {"head_pose": None}

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"head_pose": None}

        results = model.track(frame, persist=True, verbose=False)
        if not results[0].boxes or results[0].boxes.id is None:
            return {"head_pose": None}

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.xyn.cpu().numpy()
        keypoints_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
        img_h, img_w = frame.shape[:2]
        centers = boxes.xywh.cpu().numpy()[:, :2]
        center_points = np.array([img_w / 2, img_h / 2])
        distances = np.linalg.norm(centers - center_points, axis=1)

        if target_id is None or target_id not in ids:
            target_idx = np.argmin(distances)
            target_id = int(ids[target_idx])

        try:
            current_idx = list(ids).index(target_id)
        except ValueError:
            return {"head_pose": None}

        target_kpts = keypoints_data[current_idx]
        target_conf = keypoints_conf[current_idx] if keypoints_conf is not None else None
        pitch, yaw, roll, *_ = estimate_pose(target_kpts, target_conf, img_w, img_h)
        return {
            "head_pose": {
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
            }
        }
    except Exception:
        return {"head_pose": None}

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
        keypoints_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

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
        target_conf = keypoints_conf[current_idx] if keypoints_conf is not None else None
        pitch, yaw, roll, *_ = estimate_pose(target_kpts, target_conf, img_w, img_h)

        # Thresholds (전역 변수 사용)
        global PITCH_UP_LIMIT, PITCH_DOWN_LIMIT, YAW_LIMIT

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

@app.post("/debug_inference")
async def debug_inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    """디버그용: 추론 결과 + 3D 포즈 축이 그려진 어노테이션 이미지를 반환."""
    global target_id

    if not request.image_base64:
        return {"data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": False, "status_message": "No image"}, "annotated_image": None}

    try:
        try:
            img_data = base64.b64decode(request.image_base64)
        except Exception:
            return {"data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": False, "status_message": "Invalid base64"}, "annotated_image": None}

        if len(img_data) < 4:
            return {"data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": False, "status_message": "Invalid image"}, "annotated_image": None}

        is_valid_image = (
            img_data.startswith(b'\xff\xd8\xff') or
            img_data.startswith(b'\x89PNG') or
            img_data.startswith(b'RIFF')
        )
        if not is_valid_image:
            return {"data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": False, "status_message": "Unsupported format"}, "annotated_image": None}

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": False, "status_message": "Decode error"}, "annotated_image": None}

        debug_frame = frame.copy()
        results = model.track(frame, persist=True, verbose=False)

        if not results[0].boxes or results[0].boxes.id is None:
            cv2.putText(debug_frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return {
                "data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": True, "status_message": "No person detected"},
                "annotated_image": encode_frame_to_base64(debug_frame)
            }

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        keypoints_data = results[0].keypoints.xyn.cpu().numpy()
        keypoints_conf = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None
        img_h, img_w = frame.shape[:2]
        centers = boxes.xywh.cpu().numpy()[:, :2]
        center_points = np.array([img_w / 2, img_h / 2])
        distances = np.linalg.norm(centers - center_points, axis=1)

        if target_id is None or target_id not in ids:
            target_idx = np.argmin(distances)
            target_id = int(ids[target_idx])

        try:
            current_idx = list(ids).index(target_id)
        except ValueError:
            cv2.putText(debug_frame, "Target lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return {
                "data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": True, "status_message": "Target lost"},
                "annotated_image": encode_frame_to_base64(debug_frame)
            }

        target_kpts = keypoints_data[current_idx]
        target_conf = keypoints_conf[current_idx] if keypoints_conf is not None else None
        pitch, yaw, roll, rvec, tvec, cam_mtx = estimate_pose(target_kpts, target_conf, img_w, img_h)

        # 코(nose) 픽셀 좌표 계산
        nose_px = (target_kpts[0][0] * img_w, target_kpts[0][1] * img_h)

        # 키포인트 그리기 (코, 눈, 귀)
        kpt_indices = [0, 1, 2, 3, 4, 5, 6]  # nose, eyes, ears, shoulders
        for idx in kpt_indices:
            kx = int(target_kpts[idx][0] * img_w)
            ky = int(target_kpts[idx][1] * img_h)
            if kx > 0 and ky > 0:
                cv2.circle(debug_frame, (kx, ky), 4, (0, 255, 255), -1)

        # 3D 포즈 축 그리기
        if not (pitch == 0 and yaw == 0):
            draw_pose_axes(debug_frame, nose_px, pitch, yaw, roll,
                           rvec=rvec, tvec=tvec, camera_matrix=cam_mtx)

        # 집중 판단
        is_distracted = False
        reason = ""
        if not (pitch == 0 and yaw == 0):
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

        # 상태 텍스트 표시
        color = (0, 0, 255) if is_distracted else (0, 255, 0)
        cv2.putText(debug_frame, status_message, (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return {
            "data": {
                "pitch": float(pitch), "yaw": float(yaw), "roll": float(roll),
                "is_distracted": is_distracted, "status_message": status_message
            },
            "annotated_image": encode_frame_to_base64(debug_frame)
        }

    except Exception as e:
        logger.exception(f"[!] Debug Inference Error: {e}")
        return {"data": {"pitch": 0, "yaw": 0, "roll": 0, "is_distracted": False, "status_message": "Server Error"}, "annotated_image": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
