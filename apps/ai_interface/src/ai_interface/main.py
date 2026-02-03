import os
import httpx
import asyncio
import logging
import secrets
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Focus Monitor AI Interface")

# CORS Setup - More restrictive in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

app.add_middleware(LimitUploadSize, max_upload_size=10_000_000) # 10MB

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("ENVIRONMENT ERROR: API_KEY is not set in .env file.")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key and secrets.compare_digest(header_api_key, API_KEY):
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

# AI Model Server URLs
AI_HEAD_URL = os.getenv("AI_HEAD_URL", "http://localhost:8001/inference")
AI_EMOTION_URL = os.getenv("AI_EMOTION_URL", "http://localhost:8002/inference")
AI_BODY_URL = os.getenv("AI_BODY_URL", "http://localhost:8003/inference")

async def call_ai_server(client: httpx.AsyncClient, url: str, request: InferenceRequest):
    try:
        headers = {API_KEY_NAME: API_KEY}
        response = await client.post(url, json=request.dict(), headers=headers, timeout=2.0)
        if response.status_code == 200:
            return response.json()
        logger.error(f"Server {url} returned {response.status_code}")
        return {"error": "Sub-server error", "status_code": response.status_code}
    except Exception as e:
        logger.exception(f"Exception calling {url}")
        return {"error": str(e)}

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "ok", "service": "ai_interface"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    async with httpx.AsyncClient() as client:
        try:
            # Orchestrate multiple AI models
            tasks = [
                call_ai_server(client, AI_HEAD_URL, request),
            ]
            
            results = await asyncio.gather(*tasks)
            head_result = results[0]

            # Basic Error Handling for the required model
            if "error" in head_result:
                return InferenceResponse(
                    is_distracted=False,
                    status_message=f"AI Sub-server Error", # Hide internal error
                    head_pose=None
                )

            # Final Orchestration Logic (Rule-based)
            is_distracted = head_result.get("is_distracted", False)
            status_message = head_result.get("status_message", "Focused")

            return InferenceResponse(
                is_distracted=is_distracted,
                status_message=status_message,
                head_pose=head_result.get("head_pose")
            )
        except Exception as e:
            logger.exception("Internal Error in AI Interface")
            raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # Changed default port to 8010 to avoid conflict if old orchestrator was on 8000
    # But for now, let's keep it 8000 if that's what the design implied, 
    # OR we can use 8010 and let Operation Server be 8000.
    # Design says Operation Server is the entry point, so Operation Server should be 8000.
    uvicorn.run(app, host="0.0.0.0", port=8010)
