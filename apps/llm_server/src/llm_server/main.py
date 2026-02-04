import os
import logging
import secrets
import ollama
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import FeedbackRequest, FeedbackResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor LLM Server")

# CORS Setup - More restrictive in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("환경 변수 오류: .env 파일에 API_KEY가 설정되지 않았습니다.")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8005")

# Middleware: Limit Request Size (10MB) to prevent DoS
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_upload_size:
                raise HTTPException(status_code=413, detail="요청 크기가 너무 큽니다 (최대 10MB)")
        return await call_next(request)

app.add_middleware(LimitUploadSize, max_upload_size=10_000_000)  # 10MB

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="인증 정보를 확인할 수 없습니다"
    )

def generate_prompt(session_data: dict) -> str:
    """
    세션 데이터를 기반으로 Ollama용 프롬프트를 생성합니다.
    원본 LLM 클라이언트 로직을 기반으로 합니다.
    """
    duration_min = session_data.get('duration', 0) // 60
    score = session_data.get('focus_score', 0)
    distractions = session_data.get('distract_cnt', 0)
    model_type = session_data.get('model_type', 'HEAD')
    
    prompt = (
        f"당신은 업무 생산성 코치입니다. 다음 세션 데이터를 분석해주세요.\n"
        f"- 집중 시간: {duration_min}분\n"
        f"- 집중 점수: {score}점\n"
        f"- 산만 횟수: {distractions}회\n"
        f"- 감지 모델: {model_type}\n\n"
        f"사용자의 감정을 고려한 따뜻한 격려나 위로의 말(comment)과 데이터에 기반한 구체적이고 실천 가능한 행동 교정 팁(feedback)을 제공해주세요."
    )
    return prompt

@app.get("/")
async def root():
    return {
        "message": "Focus Monitor LLM Server",
        "status": "running",
        "model": OLLAMA_MODEL
    }

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "llm_server"}

@app.post("/feedback", response_model=FeedbackResponse)
async def generate_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Ollama를 사용하여 세션 데이터를 기반으로 피드백을 생성합니다.
    """
    try:
        session_data = request.session_data
        
        # 프롬프트 생성
        prompt = generate_prompt(session_data)
        logger.info(f"세션 피드백 생성 중: score={session_data.get('focus_score')}, duration={session_data.get('duration')}")
        
        # 구조화된 출력으로 Ollama 호출
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful coach. Output ONLY valid JSON.'},
                    {'role': 'user', 'content': prompt},
                ],
                format=FeedbackResponse.model_json_schema()
            )
            
            content = response['message']['content']
            logger.info(f"Ollama로부터 응답 수신 (길이: {len(content)})")
            
            # Pydantic 모델이 문자열(JSON 또는 ```json 블록) 및 dict 공통 처리
            try:
                return FeedbackResponse.model_validate(content)
            except Exception as validation_error:
                logger.warning(f"Pydantic 검증 실패, 폴백 반환: {validation_error}")
                return FeedbackResponse(
                    comment="분석 결과가 없습니다.",
                    feedback="피드백이 없습니다."
                )
            
        except Exception as e:
            logger.exception(f"Ollama 호출 중 오류: {e}")
            # 오류 시 폴백 JSON 반환
            return FeedbackResponse(
                comment="분석 중 오류가 발생했습니다.",
                feedback="Ollama 서비스 연결에 실패했습니다. 관리자에게 문의해주세요."
            )
            
    except Exception as e:
        logger.exception(f"피드백 요청 처리 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail="내부 서버 오류가 발생했습니다."
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
