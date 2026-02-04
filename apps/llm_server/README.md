# LLM Server

세션 요약 데이터를 기반으로 개인화된 피드백 코멘트를 생성하는 LLM 서버입니다.

## 개요

이 서비스는 세션 집중도 데이터를 분석하여 LLM(Large Language Model)을 사용해 개인화된 피드백 코멘트를 생성합니다. 운영 서버와 통합되어 학습 세션에 대한 AI 기반 인사이트를 제공합니다.

## 기능

- **개인화된 피드백**: 세션 통계를 기반으로 한국어 피드백 생성
- **다중 LLM 백엔드**: Ollama(로컬) 및 OpenAI(클라우드) LLM 제공자 지원
- **보안 API**: API 키 인증 및 CORS 보호
- **FastAPI**: 현대적인 비동기 API 프레임워크

## 엔드포인트

### `GET /health`
헬스 체크 엔드포인트입니다.

**헤더:**
- `X-API-Key`: 인증용 API 키

**응답:**
```json
{
  "status": "ok",
  "service": "llm_server"
}
```

### `POST /generate-comment`
세션 요약을 기반으로 개인화된 피드백 코멘트를 생성합니다.

**헤더:**
- `X-API-Key`: 인증용 API 키

**요청 본문:**
```json
{
  "session_id": "uuid-string",
  "start_time": "2024-01-01T10:00:00",
  "end_time": "2024-01-01T11:00:00",
  "focus_ratio": 85.5,
  "distraction_count": 5,
  "llm_comment": null
}
```

**응답:**
```json
{
  "comment": "생성된 피드백 코멘트..."
}
```

## 설정

환경 변수 (`.env` 파일에 설정):

- `PORT`: 서버 포트 (기본값: 8004)
- `API_KEY`: 인증용 API 키 (필수)
- `ALLOWED_ORIGINS`: 허용된 CORS 출처 목록 (쉼표로 구분, 기본값: "*")
- `LLM_TYPE`: LLM 제공자 타입 - "ollama" 또는 "openai" (기본값: "ollama")

### Ollama 설정
- `OLLAMA_URL`: Ollama API URL (기본값: "http://localhost:11434/api/generate")
- `OLLAMA_MODEL`: 사용할 모델 이름 (기본값: "llama3.2")

### OpenAI 설정
- `OPENAI_API_KEY`: OpenAI API 키 (OpenAI 사용 시 필수)
- `OPENAI_MODEL`: 사용할 모델 이름 (기본값: "gpt-3.5-turbo")

## 설치

1. `.env.example`을 `.env`로 복사하고 설정:
```bash
cp .env.example .env
```

2. 의존성 설치:
```bash
uv sync
```

3. Ollama 설정:
```bash
# Ollama 설치 (아직 설치하지 않은 경우)
# 참고: https://ollama.ai

# 모델 다운로드
ollama pull llama3.2
```

4. 서버 실행:
```bash
uv run python -m llm_server.main
```

또는 uvicorn을 직접 사용:
```bash
uvicorn llm_server.main:app --host 0.0.0.0 --port 8004
```

## 통합

LLM 서버는 세션이 종료될 때 운영 서버에 의해 자동으로 호출됩니다. 운영 서버는 세션 요약을 전송하고 생성된 코멘트를 받아 데이터베이스에 저장합니다.

## 아키텍처

- **FastAPI**: 웹 프레임워크
- **httpx**: LLM API 호출용 비동기 HTTP 클라이언트
- **monitor-shared**: 공유 스키마 및 모델
- **python-dotenv**: 환경 변수 관리

## 에러 처리

- LLM 서비스를 사용할 수 없는 경우, 운영 서버는 코멘트 없이 계속 진행합니다 (우아한 성능 저하)
- API 오류는 로그에 기록되고 적절한 HTTP 상태 코드와 함께 반환됩니다
- LLM의 빈 응답은 오류 응답으로 처리됩니다
