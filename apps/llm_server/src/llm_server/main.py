import os
import json
import logging
import secrets
import ollama
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import FeedbackRequest, FeedbackResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor LLM Server")

# Security
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("ENVIRONMENT ERROR: API_KEY is not set in .env file.")

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
                raise HTTPException(status_code=413, detail="Payload too large (Max 10MB)")
        return await call_next(request)

app.add_middleware(LimitUploadSize, max_upload_size=10_000_000)  # 10MB

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

def generate_prompt(session_data: dict) -> str:
    """
    Generate prompt for Ollama based on session data.
    Based on the original LLM client logic.
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
        f"반드시 아래 JSON 형식으로만 응답하세요. 다른 말은 하지 마세요.\n"
        f"{{\n"
        f"  \"comment\": \"사용자의 감정을 고려한 따뜻한 격려나 위로의 말 (1~2문장)\",\n"
        f"  \"feedback\": \"데이터에 기반한 구체적이고 실천 가능한 행동 교정 팁 (개조식으로 3가지)\"\n"
        f"}}"
    )
    return prompt

def parse_ollama_response(content: str) -> dict:
    """
    Parse Ollama response and extract JSON.
    Handles cases where LLM wraps JSON in ```json blocks.
    """
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from response: {content[:200]}")
        # Return fallback if parsing fails
        return {
            "comment": "분석 결과를 생성하는 중 오류가 발생했습니다.",
            "feedback": "세션 데이터를 다시 확인해주세요."
        }

@app.get("/")
async def root():
    return {
        "message": "Focus Monitor LLM Server",
        "status": "running",
        "model": OLLAMA_MODEL
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/feedback", response_model=FeedbackResponse)
async def generate_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Generate feedback based on session data using Ollama.
    """
    try:
        session_data = request.session_data
        
        # Generate prompt
        prompt = generate_prompt(session_data)
        logger.info(f"Generating feedback for session: score={session_data.get('focus_score')}, duration={session_data.get('duration')}")
        
        # Call Ollama
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful coach. Output ONLY valid JSON.'},
                    {'role': 'user', 'content': prompt},
                ]
            )
            
            content = response['message']['content']
            logger.info(f"Received response from Ollama (length: {len(content)})")
            
            # Parse JSON response
            parsed = parse_ollama_response(content)
            
            return FeedbackResponse(
                comment=parsed.get("comment", "분석 결과가 없습니다."),
                feedback=parsed.get("feedback", "피드백이 없습니다.")
            )
            
        except Exception as e:
            logger.exception(f"Error calling Ollama: {e}")
            # Return fallback JSON on error
            return FeedbackResponse(
                comment="분석 중 오류가 발생했습니다.",
                feedback=f"Ollama 연결 상태를 확인해주세요. ({str(e)})"
            )
            
    except Exception as e:
        logger.exception(f"Error processing feedback request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
