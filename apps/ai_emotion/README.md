# Monitor AI Emotion
Emotion Detection AI server for the Focus Monitor project.

## 개요
YOLO 모델을 사용하여 이미지에서 감정을 감지하는 FastAPI 서버입니다.

## 요구사항
- Python >= 3.10
- YOLO 모델 파일 (`best.pt`) - `apps/ai_emotion/` 디렉토리에 위치해야 합니다

## 환경 설정
`.env.example` 파일을 참고하여 `.env` 파일을 생성하고 다음 변수들을 설정하세요:
- `PORT`: 서버 포트 (기본값: 8002)
- `API_KEY`: API 인증 키
- `ALLOWED_ORIGINS`: 허용된 CORS 오리진 (쉼표로 구분)

## 설치 및 실행
```bash
# 의존성 설치
uv sync

# 서버 실행
uv run python -m ai_emotion.main
```

## API 엔드포인트

### GET /health
서버 상태 확인

### POST /inference
이미지에서 감정을 감지합니다.

**요청:**
- `image_base64`: Base64로 인코딩된 이미지
- `session_id` (선택사항): 세션 ID

**응답:**
- `is_distracted`: 집중도 저하 여부
- `status_message`: 상태 메시지
- `emotion`: 감지된 감정 정보 (emotion, confidence, class_id)

## 모델 파일
YOLO 모델 파일(`best.pt`)을 `apps/ai_emotion/` 디렉토리에 복사해야 합니다.
코드는 상대 경로 `best.pt`로 모델을 로드합니다.
