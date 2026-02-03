from pydantic import BaseModel, Field
from typing import Optional, List, Union
from datetime import datetime
from uuid import UUID

class InferenceRequest(BaseModel):
    image_base64: str = Field(..., max_length=10*1024*1024)
    session_id: Optional[Union[int, str, UUID]] = None

class InferenceResponse(BaseModel):
    is_distracted: bool
    status_message: str
    head_pose: Optional[dict] = None
    emotion: Optional[dict] = None
    body_pose: Optional[dict] = None

class SessionStartResponse(BaseModel):
    session_id: Union[str, UUID]
    start_time: datetime

class SessionSummary(BaseModel):
    session_id: Union[str, UUID]
    start_time: datetime
    end_time: datetime
    focus_ratio: float
    distraction_count: int
    llm_comment: Optional[str] = None
