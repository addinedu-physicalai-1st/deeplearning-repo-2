import os
import httpx
import asyncio
import logging
import secrets
from datetime import datetime
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
AI_GAZE_URL = os.getenv("AI_GAZE_URL", "http://localhost:8005/inference")

# Operation Server (세션/로그 조회) & LLM Server (피드백 생성)
OPERATION_SERVER_URL = os.getenv("OPERATION_SERVER_URL", "http://localhost:8000").rstrip("/")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8004/feedback").rstrip("/")

def _preprocess_session_for_llm(session_summary: dict, logs: list) -> dict:
    """
    클라이언트 load_and_draw_graphs와 동일한 방식으로 연속 집중 구간·sleepy·비집중 시각 계산.
    """
    if not logs:
        start_ts = session_summary.get("start_time") or ""
        end_ts = session_summary.get("end_time") or ""
        start_str = start_ts if isinstance(start_ts, str) else (start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts))
        end_str = end_ts if isinstance(end_ts, str) else (end_ts.isoformat() if hasattr(end_ts, "isoformat") else str(end_ts))
        return {
            "duration": 0,
            # 세션 전체 집중 비율(%) = Operation Server의 focus_ratio
            "focus_ratio": session_summary.get("focus_ratio", 0),
            "distract_cnt": session_summary.get("distraction_count", 0),
            "start_time": start_str,
            "end_time": end_str,
            "longest_focus_seconds": 0,
            "avg_focus_seconds": 0,
            "sleepy_timestamps": [],
            "distracted_timestamps": [],
        }

    # timestamp 기준 정렬 (문자열이면 파싱)
    def _ts(log):
        t = log.get("timestamp")
        if t is None:
            return None
        if isinstance(t, str):
            return datetime.fromisoformat(t.replace("Z", "+00:00").replace("Z", ""))
        return t

    sorted_logs = sorted([l for l in logs if _ts(l) is not None], key=_ts)
    if not sorted_logs:
        start_ts = session_summary.get("start_time") or ""
        end_ts = session_summary.get("end_time") or ""
        start_str = start_ts if isinstance(start_ts, str) else (start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts))
        end_str = end_ts if isinstance(end_ts, str) else (end_ts.isoformat() if hasattr(end_ts, "isoformat") else str(end_ts))
        return {
            "duration": 0,
            # 세션 전체 집중 비율(%) = Operation Server의 focus_ratio
            "focus_ratio": session_summary.get("focus_ratio", 0),
            "distract_cnt": session_summary.get("distraction_count", 0),
            "start_time": start_str,
            "end_time": end_str,
            "longest_focus_seconds": 0,
            "avg_focus_seconds": 0,
            "sleepy_timestamps": [],
            "distracted_timestamps": [],
        }

    timestamps = [_ts(l) for l in sorted_logs]
    is_distracted_list = [bool(l.get("is_distracted")) for l in sorted_logs]
    base = timestamps[0]
    relative_times = [(t - base).total_seconds() for t in timestamps]

    # 연속 집중 구간 길이 (클라이언트와 동일)
    focus_segment_durations = []
    current_segment = 0.0
    for i in range(len(relative_times) - 1):
        time_diff = relative_times[i + 1] - relative_times[i]
        if is_distracted_list[i]:
            if current_segment > 0:
                focus_segment_durations.append(current_segment)
                current_segment = 0.0
        else:
            current_segment += time_diff
    if len(relative_times) > 0 and not is_distracted_list[-1]:
        current_segment += 3.0
    if current_segment > 0:
        focus_segment_durations.append(current_segment)

    longest_focus_seconds = max(focus_segment_durations) if focus_segment_durations else 0
    avg_focus_seconds = (sum(focus_segment_durations) / len(focus_segment_durations)) if focus_segment_durations else 0

    def _ts_str(log):
        t = log.get("timestamp")
        if t is None:
            return ""
        return t.isoformat() if hasattr(t, "isoformat") else str(t)

    sleepy_timestamps = []
    distracted_timestamps = []
    for log in sorted_logs:
        ts = _ts_str(log)
        emotion = (log.get("emotion_data") or {}).get("emotion") or ""
        msg = (log.get("status_message") or "").lower()
        if "sleepy" in emotion.lower() or "sleepy" in msg:
            sleepy_timestamps.append(ts)
        if log.get("is_distracted"):
            distracted_timestamps.append(ts)

    start_ts = session_summary.get("start_time")
    end_ts = session_summary.get("end_time")
    start_str = start_ts if isinstance(start_ts, str) else (start_ts.isoformat() if start_ts and hasattr(start_ts, "isoformat") else str(start_ts or ""))
    end_str = end_ts if isinstance(end_ts, str) else (end_ts.isoformat() if end_ts and hasattr(end_ts, "isoformat") else str(end_ts or ""))

    duration_seconds = 0
    if start_ts and end_ts:
        try:
            if isinstance(start_ts, str):
                start_ts = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
            if isinstance(end_ts, str):
                end_ts = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
            duration_seconds = int((end_ts - start_ts).total_seconds())
        except Exception:
            if len(relative_times) >= 2:
                duration_seconds = int(relative_times[-1] - relative_times[0]) + 3

    return {
        "duration": max(0, duration_seconds),
        # 세션 전체 집중 비율(%) = Operation Server의 focus_ratio
        "focus_ratio": session_summary.get("focus_ratio", 0),
        "distract_cnt": session_summary.get("distraction_count", 0),
        "start_time": start_str,
        "end_time": end_str,
        "longest_focus_seconds": round(longest_focus_seconds, 1),
        "avg_focus_seconds": round(avg_focus_seconds, 1),
        "sleepy_timestamps": sleepy_timestamps,
        "distracted_timestamps": distracted_timestamps,
    }


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


@app.post("/sessions/{session_id}/feedback")
async def session_feedback(session_id: str, api_key: str = Depends(get_api_key)):
    """
    Operation Server가 호출. Operation에서 세션·로그 조회 후 전처리 → LLM 호출 → llm_comment만 반환.
    """
    headers = {API_KEY_NAME: api_key}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            session_resp = await client.get(
                f"{OPERATION_SERVER_URL}/sessions/{session_id}",
                headers=headers
            )
            session_resp.raise_for_status()
            session_summary = session_resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(status_code=502, detail="Operation server session fetch failed")
        except Exception as e:
            logger.error(f"Operation session fetch: {e}")
            raise HTTPException(status_code=502, detail="Operation server unavailable")

        try:
            logs_resp = await client.get(
                f"{OPERATION_SERVER_URL}/sessions/{session_id}/logs",
                headers=headers
            )
            logs_resp.raise_for_status()
            logs_payload = logs_resp.json()
            logs = logs_payload.get("logs", [])
        except Exception as e:
            logger.error(f"Operation logs fetch: {e}")
            logs = []

    session_data = _preprocess_session_for_llm(session_summary, logs)

    try:
        async with httpx.AsyncClient(timeout=60.0) as llm_client:
            llm_resp = await llm_client.post(
                LLM_SERVER_URL,
                json={"session_data": session_data},
                headers=headers
            )
            llm_resp.raise_for_status()
            llm_result = llm_resp.json()
    except Exception as e:
        logger.error(f"LLM feedback request: {e}")
        raise HTTPException(status_code=502, detail="LLM server request failed")

    comment = llm_result.get("comment", "")
    feedback = llm_result.get("feedback", "")
    if comment and feedback:
        llm_comment = f"[격려]\n{comment}\n\n[행동팁]\n{feedback}"
    elif comment:
        llm_comment = f"[격려]\n{comment}"
    elif feedback:
        llm_comment = f"[행동팁]\n{feedback}"
    else:
        llm_comment = "피드백을 생성하지 못했습니다."
    return {"llm_comment": llm_comment}


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    async with httpx.AsyncClient(verify=True) as client:
        try:
            # Orchestrate multiple AI models
            tasks = [
                call_ai_server(client, AI_HEAD_URL, request),
                call_ai_server(client, AI_EMOTION_URL, request),
                call_ai_server(client, AI_BODY_URL, request),
                call_ai_server(client, AI_GAZE_URL, request),
            ]
            
            results = await asyncio.gather(*tasks)
            head_result = results[0]
            emotion_result = results[1]
            body_result = results[2]
            gaze_result = results[3]

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
            if "error" in gaze_result:
                logger.warning(f"Gaze server error: {gaze_result.get('error')}")
                gaze_result = None

            # Final Orchestration Logic (Rule-based)
            head_distracted = head_result.get("is_distracted", False)
            emotion_distracted = emotion_result.get("is_distracted", False) if emotion_result else False
            body_distracted = body_result.get("is_distracted", False) if body_result else False
            gaze_distracted = gaze_result.get("is_distracted", False) if gaze_result else False

            # 하나라도 True이면 최종적으로 집중하지 않은 것으로 판단
            is_distracted = head_distracted or emotion_distracted or body_distracted or gaze_distracted

            # ---- 추가: 가중치 기반 concentration_score 계산 ----
            EMOTION_WEIGHT = 0.3
            POSITION_WEIGHT = 0.3
            HEAD_WEIGHT = 0.4

            # ai_body에서 캘리브레이션한 baseline_distance_cm을 모니터 거리로 사용
            baseline_cm = None
            if body_result:
                baseline_cm = (body_result.get("body_pose") or {}).get("baseline_distance_cm")

            MONITOR_DISTANCE_CM = (
                float(baseline_cm)
                if isinstance(baseline_cm, (int, float)) and baseline_cm > 0
                else 30.0
            )
            MONITOR_WIDTH_CM = 50.0
            MONITOR_HEIGHT_CM = 30.0
            MONITOR_WIDTH_PX = 1920.0
            MONITOR_HEIGHT_PX = 1080.0

            # 1) Emotion score
            emotion_score = 0.5
            if emotion_result:
                emo = emotion_result.get("emotion") or {}
                emotion_cls = emo.get("emotion", "")
                emotion_confidence = float(emo.get("confidence", 0.0))
                if emotion_cls == "Concentrated":
                    emotion_score = 0.5 + emotion_confidence * 0.5
                elif emotion_cls == "Distracted":
                    emotion_score = (1.0 - emotion_confidence) * 0.5
                elif emotion_cls == "Sleepy":
                    emotion_score = 0.0

            # 2) Position (body) score
            position_score = 0.5
            if body_result:
                distance_cm = (body_result.get("body_pose") or {}).get("distance_cm")
                if isinstance(distance_cm, (int, float)) and distance_cm > 0:
                    position_score = 1.0 if distance_cm <= MONITOR_DISTANCE_CM else max(0.0, 1.0 - (distance_cm - MONITOR_DISTANCE_CM) / MONITOR_DISTANCE_CM)

            # 3) Head + gaze score
            head_gaze_score = 0.5
            gaze_cordinate_x = None
            gaze_cordinate_y = None
            if gaze_result:
                gaze_data = gaze_result.get("gaze_data") or {}
                gaze_coord = gaze_data.get("gaze_cordinate") or {}
                gaze_cordinate_x = gaze_coord.get("x")
                gaze_cordinate_y = gaze_coord.get("y")

            yaw_rad = math.radians((head_result.get("head_pose") or {}).get("yaw", 0.0)) if head_result else 0.0
            pitch_rad = math.radians((head_result.get("head_pose") or {}).get("pitch", 0.0)) if head_result else 0.0

            head_direction_x = math.sin(yaw_rad) * math.cos(pitch_rad)
            head_direction_y = -math.sin(pitch_rad)
            head_direction_z = math.cos(yaw_rad) * math.cos(pitch_rad)

            if gaze_cordinate_x is not None and gaze_cordinate_y is not None:
                if head_direction_z > 0:
                    t = MONITOR_DISTANCE_CM / head_direction_z
                    head_point_x_cm = head_direction_x * t
                    head_point_y_cm = head_direction_y * t

                    head_point_x_px = (head_point_x_cm / MONITOR_WIDTH_CM) * MONITOR_WIDTH_PX + (MONITOR_WIDTH_PX / 2.0)
                    head_point_y_px = (head_point_y_cm / MONITOR_HEIGHT_CM) * MONITOR_HEIGHT_PX + (MONITOR_HEIGHT_PX / 2.0)
                else:
                    head_point_x_px = MONITOR_WIDTH_PX / 2.0
                    head_point_y_px = MONITOR_HEIGHT_PX / 2.0

                dx = float(gaze_cordinate_x) - head_point_x_px
                dy = float(gaze_cordinate_y) - head_point_y_px
                dist_norm = min(1.0, math.sqrt(dx * dx + dy * dy) / (MONITOR_HEIGHT_PX / 2.0))
                head_gaze_score = 1.0 - dist_norm
            else:
                # 시선 정보가 없으면, head_result의 is_distracted만 반영
                head_gaze_score = 0.0 if head_result.get("is_distracted", False) else 1.0

            # 신호 유무와 상관없이 항상 점수를 계산하도록 is_monitor, is_seat는 1로 고정
            is_monitor = 1
            is_seat = 1

            concentration_score = (
                EMOTION_WEIGHT * emotion_score
                + POSITION_WEIGHT * position_score
                + HEAD_WEIGHT * head_gaze_score
            )

            logger.info(
                f"[ORCH] concentration_score={concentration_score:.3f} "
                f"(emotion={emotion_score:.2f}, position={position_score:.2f}, head_gaze={head_gaze_score:.2f})"
            )

            # 상태 메시지 구성
            if is_distracted:
                messages = []
                if head_distracted:
                    messages.append(head_result.get("status_message", "Head pose issue"))
                if emotion_distracted and emotion_result:
                    messages.append(emotion_result.get("status_message", "Emotion issue"))
                if body_distracted and body_result:
                    messages.append(body_result.get("status_message", "Body posture issue"))
                if gaze_distracted and gaze_result:
                    messages.append(gaze_result.get("status_message", "Gaze issue"))
                status_message = " | ".join(messages) if messages else "Distracted"
            else:
                status_message = "Focused"

            # concentration_score는 0~1 범위이므로 0~100%로 변환해서 내려줌
            concentration_pct = max(0.0, min(100.0, concentration_score * 100.0))

            return InferenceResponse(
                is_distracted=is_distracted,
                status_message=status_message,
                head_pose=head_result.get("head_pose"),
                emotion=emotion_result.get("emotion") if emotion_result else None,
                body_pose=body_result.get("body_pose") if body_result else None,
                gaze_data=gaze_result.get("gaze_data") if gaze_result else None,
                concentration_score=concentration_pct,
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
