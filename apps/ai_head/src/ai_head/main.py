import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from shared.schemas import InferenceRequest, InferenceResponse
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN

load_dotenv()

app = FastAPI(title="Focus Monitor AI Head Pose")

API_KEY = os.getenv("API_KEY", "default-secret-key")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_api_key: str = Depends(api_key_header)):
    if header_api_key == API_KEY:
        return header_api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, api_key: str = Depends(get_api_key)):
    # In a real scenario, we would decode base64 image and run MediaPipe/DeepLearning model
    # For pipeline validation, we return a mock result
    
    # Simulate processing (e.g., checking if image is provided)
    if not request.image_base64:
        return InferenceResponse(
            is_distracted=False,
            status_message="No image provided"
        )
    
    # Mock logic: For demonstration, let's say it's always focused for now
    # but we include dummy head pose data
    mock_head_pose = {
        "pitch": 0.5,
        "yaw": -1.2,
        "roll": 0.1
    }
    
    return InferenceResponse(
        is_distracted=False,
        status_message="Head Focused",
        head_pose=mock_head_pose
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
