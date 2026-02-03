from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, JSON
from .database import Base
import datetime

class FocusLog(Base):
    __tablename__ = "focus_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_distracted = Column(Boolean)
    status_message = Column(String)
    head_pose_data = Column(JSON, nullable=True)
    emotion_data = Column(JSON, nullable=True)
    body_pose_data = Column(JSON, nullable=True)
