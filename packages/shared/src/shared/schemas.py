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
