import json
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Union, Any
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
    gaze_data: Optional[dict] = None
    concentration_score: Optional[float] = None

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

class FeedbackRequest(BaseModel):
    session_data: dict

class FeedbackResponse(BaseModel):
    comment: str = Field(description="사용자 감정을 고려한 격려나 위로의 말")
    feedback: str = Field(description="데이터에 기반한 구체적이고 실천 가능한 행동 교정 팁")

    @model_validator(mode="wrap")
    @classmethod
    def _validate_raw_llm_output(cls, value: Any, handler):
        if isinstance(value, str):
            content = value.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                return cls(
                    comment="분석 결과를 생성하는 중 오류가 발생했습니다.",
                    feedback="세션 데이터를 다시 확인해주세요.",
                )
        return handler(value)

class FocusLogItem(BaseModel):
    timestamp: datetime
    is_distracted: bool
    status_message: Optional[str] = None
    emotion_data: Optional[dict] = None
    gaze_data: Optional[dict] = None

class SessionLogsResponse(BaseModel):
    session_id: Union[str, UUID]
    logs: List[FocusLogItem]

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=100)

class UserLoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    user_id: int
    username: str
    display_name: str
    created_at: datetime

class LoginResponse(BaseModel):
    user_id: int
    username: str
    display_name: str
    message: str

class SessionStartRequest(BaseModel):
    user_id: Optional[int] = None
