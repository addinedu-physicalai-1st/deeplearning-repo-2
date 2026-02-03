import datetime
import uuid
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    focus_ratio = Column(Float, default=0.0)
    distraction_count = Column(Integer, default=0)
    llm_comment = Column(String, nullable=True)
    
    # Relationship to logs
    logs = relationship("FocusLog", back_populates="session")

class FocusLog(Base):
    __tablename__ = "focus_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("monitoring_sessions.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_distracted = Column(Boolean)
    status_message = Column(String)
    head_pose_data = Column(JSON, nullable=True)
    emotion_data = Column(JSON, nullable=True)
    body_pose_data = Column(JSON, nullable=True)

    # Relationship back to session
    session = relationship("MonitoringSession", back_populates="logs")
