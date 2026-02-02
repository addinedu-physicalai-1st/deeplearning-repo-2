import os
import httpx
import asyncio
import logging
import secrets
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse
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

app = FastAPI(title="Focus Monitor Orchestrator")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("ENVIRONMENT ERROR: API_KEY is not set in .env file.")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Simple Rate Limiter (In-memory)
# In production, use Redis or a dedicated library like slowapi
rate_limit_store = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_COUNT = 30   # requests per window

def check_rate_limit(client_ip: str):
    now = time.time()
    # Remove old requests
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_COUNT:
        return False
    rate_limit_store[client_ip].append(now)
    return True

class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        # 1. Rate Limiting
        client_ip = request.client.host if request.client else "unknown"
        if not check_rate_limit(client_ip):
            return Response("Rate limit exceeded", status_code=429)

        # 2. Request Size Limiting
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_upload_size:
                raise HTTPException(status_code=413, detail="Payload too large")
        
        return await call_next(request)

app.add_middleware(SecurityMiddleware, max_upload_size=15_000_000) # 15MB

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# AI Server URLs
AI_HEAD_URL = os.getenv("AI_HEAD_URL", "http://localhost:8001/inference")
AI_EMOTION_URL = os.getenv("AI_EMOTION_URL", "http://localhost:8002/inference")
AI_BODY_URL = os.getenv("AI_BODY_URL", "http://localhost:8003/inference")
DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8005/logs")

async def call_ai_server(client: httpx.AsyncClient, url: str, request: InferenceRequest):
    try:
        # Pass internal API key to sub-servers
        headers = {API_KEY_NAME: API_KEY}
        response = await client.post(url, json=request.dict(), headers=headers, timeout=2.0)
        if response.status_code == 200:
            return response.json()
        logger.error(f"Server {url} returned {response.status_code}")
        return {"error": "Sub-server error"}
    except Exception as e:
        logger.exception(f"Exception calling {url}")
        return {"error": "Internal connection error"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    async with httpx.AsyncClient() as client:
        # For pipeline validation, we primarily focus on AI Head
        # But we structure it to handle multiple models
        tasks = [
            call_ai_server(client, AI_HEAD_URL, request),
            # placeholders for others
            # call_ai_server(client, AI_EMOTION_URL, request),
            # call_ai_server(client, AI_BODY_URL, request),
        ]
        
        results = await asyncio.gather(*tasks)
        
        head_result = results[0]
        # emotion_result = results[1]
        # body_result = results[2]

        # Simplified Logic for Pipeline Validation:
        # If head pose says distracted, then distracted.
        is_distracted = head_result.get("is_distracted", False)
        status_message = head_result.get("status_message", "Focused")

        # In the future, this will be a more complex Rule-based Aggregator
        # if head_result.get("is_distracted") or emotion_result.get("is_distracted"):
        #     is_distracted = True

        # Send to DB Server (Fire and Forget or Async)
        # For now, just print or log since DB server is not yet built
        logger.info(f"Orchestration Result: distracted={is_distracted}, msg={status_message}")

        return InferenceResponse(
            is_distracted=is_distracted,
            status_message=status_message,
            head_pose=head_result if "error" not in head_result else None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
