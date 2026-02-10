import datetime
import uuid
import hashlib
import hmac
import os as _os
from zoneinfo import ZoneInfo
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

KST = ZoneInfo("Asia/Seoul")

def _now_kst_naive():
    return datetime.datetime.now(KST).replace(tzinfo=None)

def hash_password(password: str) -> str:
    salt = _os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex() + "$" + h.hex()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        salt_hex, hash_hex = hashed_password.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        h = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(h.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=_now_kst_naive)

    sessions = relationship("MonitoringSession", back_populates="user")

class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    start_time = Column(DateTime, default=_now_kst_naive)
    end_time = Column(DateTime, nullable=True)
    focus_ratio = Column(Float, default=0.0)
    distraction_count = Column(Integer, default=0)
    llm_comment = Column(String, nullable=True)

    user = relationship("User", back_populates="sessions")
    # Relationship to logs
    logs = relationship("FocusLog", back_populates="session")

class FocusLog(Base):
    __tablename__ = "focus_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("monitoring_sessions.id"), nullable=True)
    timestamp = Column(DateTime, default=_now_kst_naive)
    is_distracted = Column(Boolean)
    status_message = Column(String)
    head_pose_data = Column(JSON, nullable=True)
    emotion_data = Column(JSON, nullable=True)
    body_pose_data = Column(JSON, nullable=True)
    gaze_data = Column(JSON, nullable=True)
    posture_alert = Column(Boolean, default=False, nullable=True)  # 거북목 경고 (ai_body body_pose.posture_alert)

    # Relationship back to session
    session = relationship("MonitoringSession", back_populates="logs")
