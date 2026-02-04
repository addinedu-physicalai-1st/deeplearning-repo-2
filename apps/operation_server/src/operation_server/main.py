import os
import sys

# Add the 'src' directory to the system path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import httpx
import logging
import secrets
import time
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse, SessionStartResponse, SessionSummary, FeedbackRequest
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime

# Change to absolute imports within the package
from operation_server.database import engine, get_db
from operation_server import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor Operation Server")

# CORS Setup - More restrictive in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("환경 변수 오류: .env 파일에 API_KEY가 설정되지 않았습니다.")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Simple Rate Limiter
rate_limit_store = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_COUNT = 30

# Trusted Proxies (Empty by default, add IPs if using Nginx/Cloudflare)
TRUSTED_PROXIES = set(os.getenv("TRUSTED_PROXIES", "").split(","))

def check_rate_limit(client_ip: str):
    now = time.time()
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_COUNT:
        return False
    rate_limit_store[client_ip].append(now)
    return True

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. IP Determination (More Secure)
        client_ip = request.client.host if request.client else "unknown"
        
        # Only trust X-Forwarded-For if it comes from a trusted proxy
        if client_ip in TRUSTED_PROXIES or not TRUSTED_PROXIES:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # Use the last IP in the chain if trusting the immediate proxy
                client_ip = forwarded.split(",")[-1].strip()

        # 2. Rate Limiting
        if not check_rate_limit(client_ip):
            return Response("Rate limit exceeded", status_code=429)
        
        return await call_next(request)

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

app.add_middleware(SecurityMiddleware)
app.add_middleware(LimitUploadSize, max_upload_size=10_000_000) # 10MB

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Unauthorized")

# Service URLs
AI_INTERFACE_URL = os.getenv("AI_INTERFACE_URL", "http://localhost:8010/inference")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8004/feedback")

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "operation_server"}

@app.post("/sessions/start", response_model=SessionStartResponse)
async def start_session(api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    new_session = models.MonitoringSession()
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return SessionStartResponse(session_id=new_session.id, start_time=new_session.start_time)

@app.post("/sessions/stop/{session_id}", response_model=SessionSummary)
async def stop_session(session_id: str, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.end_time = datetime.utcnow()
    
    # Calculate stats from logs
    logs = db.query(models.FocusLog).filter(models.FocusLog.session_id == session_id).all()
    total_logs = len(logs)
    if total_logs > 0:
        distracted_logs = len([log for log in logs if log.is_distracted])
        focused_logs = total_logs - distracted_logs
        session.focus_ratio = (focused_logs / total_logs) * 100
        session.distraction_count = distracted_logs
    
    # LLM 피드백 코멘트 생성
    session.llm_comment = None
    try:
        # 세션 지속 시간 계산 (초 단위)
        duration_seconds = 0
        if session.end_time and session.start_time:
            duration = session.end_time - session.start_time
            duration_seconds = int(duration.total_seconds())
        
        # 새로운 API 형식에 맞는 세션 데이터 생성
        session_data = {
            'duration': duration_seconds,
            'focus_score': session.focus_ratio,  # 집중 비율을 점수로 사용
            'distract_cnt': session.distraction_count,
            'model_type': 'HEAD'  # 기본 모델 타입
        }
        
        feedback_request = FeedbackRequest(session_data=session_data)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {API_KEY_NAME: API_KEY}
            response = await client.post(
                LLM_SERVER_URL,
                json=feedback_request.dict(),
                headers=headers
            )
            response.raise_for_status()
            llm_result = response.json()
            # comment와 feedback을 결합하여 저장
            comment = llm_result.get("comment", "")
            feedback = llm_result.get("feedback", "")
            if comment and feedback:
                session.llm_comment = f"{comment}\n\n{feedback}"
            elif comment:
                session.llm_comment = comment
            elif feedback:
                session.llm_comment = feedback
            logger.info(f"세션 {session_id}에 대한 LLM 피드백 생성 완료")
    except httpx.HTTPStatusError as e:
        logger.warning(f"LLM 서버 오류: {e.response.status_code}, 코멘트 없이 계속 진행")
    except Exception as e:
        logger.warning(f"LLM 피드백 생성 실패: {e}, 코멘트 없이 계속 진행")
    
    db.commit()
    db.refresh(session)
    
    return SessionSummary(
        session_id=session.id,
        start_time=session.start_time,
        end_time=session.end_time,
        focus_ratio=session.focus_ratio,
        distraction_count=session.distraction_count,
        llm_comment=session.llm_comment
    )

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    # 0. Basic Image Validation
    if request.image_base64:
        try:
            import base64
            img_data = base64.b64decode(request.image_base64)
            if len(img_data) < 4:
                raise HTTPException(status_code=400, detail="Invalid image data")
            
            # Simple magic number check
            if not (img_data.startswith(b'\xff\xd8\xff') or img_data.startswith(b'\x89PNG') or img_data.startswith(b'RIFF')):
                logger.warning("Invalid image format attempt")
                raise HTTPException(status_code=400, detail="Unsupported image format")
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    async with httpx.AsyncClient() as client:
        try:
            # 1. Forward request to AI Interface Server
            headers = {API_KEY_NAME: API_KEY}
            response = await client.post(
                AI_INTERFACE_URL, 
                json=request.dict(), 
                headers=headers, 
                timeout=5.0
            )
            response.raise_for_status()
            ai_result = response.json()
            
            # 2. Save result to SQLite DB
            new_log = models.FocusLog(
                session_id=request.session_id,
                is_distracted=ai_result.get('is_distracted'),
                status_message=ai_result.get('status_message'),
                head_pose_data=ai_result.get('head_pose'),
                emotion_data=ai_result.get('emotion'),
                body_pose_data=ai_result.get('body_pose')
            )
            db.add(new_log)
            db.commit()
            
            logger.info(f"Saved to DB: distracted={ai_result.get('is_distracted')}")
            
            return InferenceResponse(**ai_result)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"AI Interface returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail="AI Interface Error")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Internal Error in Operation Server")
            raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
