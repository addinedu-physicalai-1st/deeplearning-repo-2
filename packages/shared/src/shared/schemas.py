from pydantic import BaseModel, Field
from typing import Optional, List

class InferenceRequest(BaseModel):
    image_base64: str = Field(..., max_length=10*1024*1024)  # Limit to 10MB to prevent DoS

class InferenceResponse(BaseModel):
    is_distracted: bool
    status_message: str
    head_pose: Optional[dict] = None
    emotion: Optional[dict] = None
    body_pose: Optional[dict] = None

class FeedbackRequest(BaseModel):
    session_data: dict = Field(..., description="Session data with duration, focus_score, distract_cnt, model_type, etc.")
    # Optional: session_id for future DB integration
    # session_id: Optional[int] = None

class FeedbackResponse(BaseModel):
    comment: str = Field(..., description="Encouraging comment")
    feedback: str = Field(..., description="Actionable feedback tips")
