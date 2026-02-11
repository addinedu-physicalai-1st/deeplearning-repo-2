import os
import sys

# Add the 'src' directory to the system path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import asyncio
import httpx
import logging
import secrets
import time
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import (InferenceRequest, InferenceResponse, SessionStartResponse, SessionSummary,
                             FeedbackRequest, FeedbackResponse, SessionLogsResponse, FocusLogItem,
                             UserRegisterRequest, UserLoginRequest, UserResponse, LoginResponse, SessionStartRequest)
from typing import List
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Change to absolute imports within the package
from operation_server.database import engine, get_db, SessionLocal
from operation_server import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Add posture_alert column to focus_logs if missing (existing DB migration)
try:
    from sqlalchemy import text
    if "sqlite" in str(engine.url):
        with engine.connect() as conn:
            r = conn.execute(text("PRAGMA table_info(focus_logs)"))
            cols = [row[1] for row in r]
        if "posture_alert" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE focus_logs ADD COLUMN posture_alert BOOLEAN DEFAULT 0"))
except Exception as e:
    logging.warning(f"Migration posture_alert: {e}")

# Add gaze_data column to focus_logs if missing
try:
    if "sqlite" in str(engine.url):
        with engine.connect() as conn:
            r = conn.execute(text("PRAGMA table_info(focus_logs)"))
            cols = [row[1] for row in r]
        if "gaze_data" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE focus_logs ADD COLUMN gaze_data JSON"))
except Exception as e:
    logging.warning(f"Migration gaze_data: {e}")

# Add user_id column to monitoring_sessions if missing
try:
    if "sqlite" in str(engine.url):
        with engine.connect() as conn:
            r = conn.execute(text("PRAGMA table_info(monitoring_sessions)"))
            cols = [row[1] for row in r]
        if "user_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE monitoring_sessions ADD COLUMN user_id INTEGER"))
except Exception as e:
    logging.warning(f"Migration user_id: {e}")

load_dotenv()

# Data retention settings
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor Operation Server")

# CORS Setup
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
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

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="인증 정보를 확인할 수 없습니다"
    )

def cleanup_expired_data():
    """보관 기간이 만료된 세션 및 로그 데이터를 삭제합니다."""
    db = SessionLocal()
    try:
        cutoff = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None) - timedelta(days=DATA_RETENTION_DAYS)

        expired_sessions = db.query(models.MonitoringSession).filter(
            models.MonitoringSession.start_time < cutoff
        ).all()

        if not expired_sessions:
            logger.info("만료된 데이터가 없습니다.")
            return

        session_ids = [s.id for s in expired_sessions]

        deleted_logs = db.query(models.FocusLog).filter(
            models.FocusLog.session_id.in_(session_ids)
        ).delete(synchronize_session=False)

        deleted_sessions = db.query(models.MonitoringSession).filter(
            models.MonitoringSession.id.in_(session_ids)
        ).delete(synchronize_session=False)

        db.commit()
        logger.info(f"데이터 정리 완료: {deleted_sessions}개 세션, {deleted_logs}개 로그 삭제 (보관 기간: {DATA_RETENTION_DAYS}일)")
    except Exception as e:
        db.rollback()
        logger.error(f"데이터 정리 중 오류: {e}")
    finally:
        db.close()

async def _periodic_cleanup():
    """24시간 간격으로 만료 데이터를 정리하는 백그라운드 태스크."""
    while True:
        await asyncio.sleep(86400)
        cleanup_expired_data()

@app.on_event("startup")
async def on_startup():
    cleanup_expired_data()
    asyncio.create_task(_periodic_cleanup())

# Service URLs (AI_INTERFACE_URL is base e.g. http://localhost:8010; inference = base + /inference)
AI_INTERFACE_BASE = (os.getenv("AI_INTERFACE_URL", "http://localhost:8010").rstrip("/").replace("/inference", "") or "http://localhost:8010")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8004/feedback")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "operation_server"}

@app.post("/auth/register", response_model=UserResponse)
async def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.username == request.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="이미 사용 중인 사용자 이름입니다.")
    new_user = models.User(
        username=request.username,
        password_hash=models.hash_password(request.password),
        display_name=request.display_name,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    kst = ZoneInfo("Asia/Seoul")
    return UserResponse(
        user_id=new_user.id,
        username=new_user.username,
        display_name=new_user.display_name,
        created_at=new_user.created_at.replace(tzinfo=kst) if new_user.created_at else None,
    )

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == request.username).first()
    if not user or not models.verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="사용자 이름 또는 비밀번호가 올바르지 않습니다.")
    return LoginResponse(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        message="로그인 성공",
    )

@app.post("/sessions/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest = None, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    new_session = models.MonitoringSession(
        user_id=request.user_id if request else None
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return SessionStartResponse(session_id=new_session.id, start_time=new_session.start_time)

@app.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    """단일 세션 조회 (AI Interface가 세션 요약 조회 시 사용)."""
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    kst = ZoneInfo("Asia/Seoul")
    return SessionSummary(
        session_id=session.id,
        start_time=session.start_time.replace(tzinfo=kst) if session.start_time else None,
        end_time=session.end_time.replace(tzinfo=kst) if session.end_time else None,
        focus_ratio=session.focus_ratio,
        distraction_count=session.distraction_count,
        llm_comment=session.llm_comment
    )

@app.get("/sessions", response_model=List[SessionSummary])
async def get_sessions(user_id: int = None, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    query = db.query(models.MonitoringSession).filter(models.MonitoringSession.end_time != None)
    if user_id is not None:
        query = query.filter(models.MonitoringSession.user_id == user_id)
    sessions = query.order_by(models.MonitoringSession.start_time.desc()).all()
    kst = ZoneInfo("Asia/Seoul")
    return [
        SessionSummary(
            session_id=s.id,
            start_time=s.start_time.replace(tzinfo=kst) if s.start_time else None,
            end_time=s.end_time.replace(tzinfo=kst) if s.end_time else None,
            focus_ratio=s.focus_ratio,
            distraction_count=s.distraction_count,
            llm_comment=s.llm_comment
        ) for s in sessions
    ]

@app.post("/sessions/stop/{session_id}", response_model=SessionSummary)
async def stop_session(session_id: str, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.end_time = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    
    # Calculate stats from logs
    logs = db.query(models.FocusLog).filter(models.FocusLog.session_id == session_id).all()
    total_logs = len(logs)
    if total_logs > 0:
        distracted_logs = len([log for log in logs if log.is_distracted])
        focused_logs = total_logs - distracted_logs
        session.focus_ratio = (focused_logs / total_logs) * 100
        session.distraction_count = distracted_logs
    
    db.commit()
    db.refresh(session)
    
    kst = ZoneInfo("Asia/Seoul")
    return SessionSummary(
        session_id=session.id,
        start_time=session.start_time.replace(tzinfo=kst) if session.start_time else None,
        end_time=session.end_time.replace(tzinfo=kst) if session.end_time else None,
        focus_ratio=session.focus_ratio,
        distraction_count=session.distraction_count,
        llm_comment=session.llm_comment
    )

@app.get("/sessions/{session_id}/logs", response_model=SessionLogsResponse)
async def get_session_logs(session_id: str, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    """세션의 모든 로그 데이터를 반환"""
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logs = db.query(models.FocusLog).filter(models.FocusLog.session_id == session_id).order_by(models.FocusLog.timestamp).all()

    def _normalize_emotion_data(data):
        if data is None:
            return None
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        return data if isinstance(data, dict) else None

    log_items = [
        FocusLogItem(
            timestamp=log.timestamp,
            is_distracted=log.is_distracted,
            status_message=log.status_message,
            emotion_data=_normalize_emotion_data(log.emotion_data),
            gaze_data=log.gaze_data if hasattr(log, 'gaze_data') else None
        ) for log in logs
    ]
    
    return SessionLogsResponse(
        session_id=session_id,
        logs=log_items
    )

@app.post("/sessions/{session_id}/feedback")
async def generate_session_feedback(session_id: str, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    """클라이언트가 호출. AI Interface로 위임 후 llm_comment 저장·반환."""
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    duration_seconds = 0
    if session.end_time and session.start_time:
        duration = session.end_time - session.start_time
        duration_seconds = int(duration.total_seconds())
    
    if duration_seconds < 10:
        session.llm_comment = "모니터링 시간이 너무 짧아 코멘트를 생성하지 못하였습니다."
        db.commit()
        return {"llm_comment": session.llm_comment}

    try:
        feedback_url = f"{AI_INTERFACE_BASE.rstrip('/')}/sessions/{session_id}/feedback"
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {API_KEY_NAME: API_KEY}
            response = await client.post(feedback_url, json={}, headers=headers)
            response.raise_for_status()
            result = response.json()
            llm_comment = result.get("llm_comment", "")
            session.llm_comment = llm_comment
            db.commit()
            return {"llm_comment": session.llm_comment}
    except Exception as e:
        logger.error(f"AI Interface feedback Error: {e}")
        raise HTTPException(status_code=500, detail="LLM 분석 중 오류가 발생했습니다.")

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        try:
            headers = {API_KEY_NAME: API_KEY}
            response = await client.post(f"{AI_INTERFACE_BASE.rstrip('/')}/inference", json=request.dict(), headers=headers, timeout=5.0)
            response.raise_for_status()
            ai_result = response.json()
            
            body_pose = ai_result.get('body_pose') or {}
            new_log = models.FocusLog(
                session_id=request.session_id,
                is_distracted=ai_result.get('is_distracted'),
                status_message=ai_result.get('status_message'),
                head_pose_data=ai_result.get('head_pose'),
                emotion_data=ai_result.get('emotion'),
                body_pose_data=ai_result.get('body_pose'),
                gaze_data=ai_result.get('gaze_data'),
                posture_alert=body_pose.get('posture_alert', False),
            )
            db.add(new_log)
            db.commit()
            return InferenceResponse(**ai_result)
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            raise HTTPException(status_code=500, detail="AI 분석 서버 통신 오류")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
