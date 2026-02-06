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
from shared.schemas import InferenceRequest, InferenceResponse, SessionStartResponse, SessionSummary, FeedbackRequest, FeedbackResponse, SessionLogsResponse, FocusLogItem
from typing import List
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

# Service URLs
AI_INTERFACE_URL = os.getenv("AI_INTERFACE_URL", "http://localhost:8010/inference")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8004/feedback")

@app.get("/health")
async def health_check(api_key: str = Depends(api_key_header)):
    return {"status": "ok", "service": "operation_server"}

@app.post("/sessions/start", response_model=SessionStartResponse)
async def start_session(api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    new_session = models.MonitoringSession()
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return SessionStartResponse(session_id=new_session.id, start_time=new_session.start_time)

@app.get("/sessions", response_model=List[SessionSummary])
async def get_sessions(api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    sessions = db.query(models.MonitoringSession).filter(models.MonitoringSession.end_time != None).order_by(models.MonitoringSession.start_time.desc()).all()
    return [
        SessionSummary(
            session_id=s.id,
            start_time=s.start_time,
            end_time=s.end_time,
            focus_ratio=s.focus_ratio,
            distraction_count=s.distraction_count,
            llm_comment=s.llm_comment
        ) for s in sessions
    ]

@app.post("/sessions/stop/{session_id}", response_model=SessionSummary)
async def stop_session(session_id: str, api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
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

@app.get("/sessions/{session_id}/logs", response_model=SessionLogsResponse)
async def get_session_logs(session_id: str, api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    """세션의 모든 로그 데이터를 반환"""
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logs = db.query(models.FocusLog).filter(models.FocusLog.session_id == session_id).order_by(models.FocusLog.timestamp).all()
    
    log_items = [
        FocusLogItem(
            timestamp=log.timestamp,
            is_distracted=log.is_distracted,
            status_message=log.status_message
        ) for log in logs
    ]
    
    return SessionLogsResponse(
        session_id=session_id,
        logs=log_items
    )

@app.post("/sessions/{session_id}/feedback")
async def generate_session_feedback(session_id: str, api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    session = db.query(models.MonitoringSession).filter(models.MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 세션 지속 시간 계산
    duration_seconds = 0
    if session.end_time and session.start_time:
        duration = session.end_time - session.start_time
        duration_seconds = int(duration.total_seconds())
    
    if duration_seconds < 10: # 테스트 편의를 위해 10초로 하향
        session.llm_comment = "모니터링 시간이 너무 짧아 코멘트를 생성하지 못하였습니다."
        db.commit()
        return {"llm_comment": session.llm_comment}

    try:
        session_data = {
            'duration': duration_seconds,
            'focus_score': session.focus_ratio,
            'distract_cnt': session.distraction_count,
            'model_type': 'HEAD'
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {API_KEY_NAME: API_KEY}
            response = await client.post(LLM_SERVER_URL, json={"session_data": session_data}, headers=headers)
            response.raise_for_status()
            llm_result = response.json()
            comment = llm_result.get("comment", "")
            feedback = llm_result.get("feedback", "")
            session.llm_comment = f"{comment}\n\n{feedback}" if comment and feedback else (comment or feedback)
            db.commit()
            return {"llm_comment": session.llm_comment}
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail="LLM 분석 중 오류가 발생했습니다.")

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        try:
            headers = {API_KEY_NAME: API_KEY}
            response = await client.post(AI_INTERFACE_URL, json=request.dict(), headers=headers, timeout=5.0)
            response.raise_for_status()
            ai_result = response.json()
            
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
            return InferenceResponse(**ai_result)
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            raise HTTPException(status_code=500, detail="AI 분석 서버 통신 오류")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
