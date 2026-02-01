# Focus Monitor (업무 집중도 모니터링 시스템)

딥러닝 기반의 실시간 업무 집중 모니터링 및 코칭 시스템입니다. 웹캠을 통해 사용자의 상태(고개 각도, 표정, 자세)를 분석하여 집중 상태를 판정하고, 모니터링 종료 후 AI 피드백을 제공합니다.

---

## 🏗 시스템 아키텍처

본 프로젝트는 확장성과 효율성을 위해 **Microservices Architecture (MSA)** 스타일의 구조를 채택하고 있습니다.

```mermaid
graph TD
    Client[Client GUI - PyQt6] <--> Orch[Orchestrator - FastAPI]
    Orch <--> AI_Head[AI Head Pose - FastAPI]
    Orch <--> AI_Emotion[AI Emotion - FastAPI]
    Orch <--> AI_Body[AI Upper Body - FastAPI]
    Orch <--> DB[DB Server - SQLite]
    DB <--> LLM[LLM Server - Feedback AI]
```

1.  **Client (PyQt6)**: 실시간 웹캠 프리뷰, 주기적 이미지 캡처 및 전송, 비집중 알림 시각화.
2.  **Orchestrator (FastAPI)**: 요청 분배, 다중 모델 결과 집계(Aggregation), 최종 비집중 판정 알고리즘 수행.
3.  **AI Models**: 비전 AI 기반의 개별 추론 서버 (고개 각도, 감정, 상체 자세).
4.  **Feedback AI**: 세션 데이터 기반 맞춤형 코멘트 생성 (Ollama/Llama 3.1).

---

## 📌 주요 기능

*   **실시간 집중도 모니터링**: 3~5초 주기로 사용자의 상태를 분석하여 집중 여부 판정.
*   **다중 지표 분석**: 고개 숙임/돌림(Head Pose), 표정(Emotion), 어깨 기울기(Body Pose)를 복합적으로 고려.
*   **즉각적인 피드백**: 비집중 상태 감지 시 화면 오버레이를 통해 주의 환기 유도.
*   **통계 및 AI 코칭**: 세션 종료 후 집중 시간 비율 시각화 및 LLM 기반 행동 교정 팁 제공.

---

## 📂 프로젝트 구조 (uv Workspace)

```text
deeplearning-repo-2/
├── apps/
│   ├── client/           # [UI] PyQt6 데스크탑 애플리케이션
│   ├── orchestrator/     # [Control] 중앙 제어 서버
│   ├── ai_head/          # [Vision] Head Pose 분석 서버 (작업 중)
│   ├── ai_emotion/       # [Vision] 표정 분석 서버 (오픈 소스 예정)
│   ├── ai_body/          # [Vision] 상체 자세 분석 서버 (오픈 소스 예정)
│   ├── db_server/        # [Data] SQLite 데이터 관리 서버
│   └── llm_server/       # [AI] 피드백 생성 서버
├── packages/
│   └── shared/           # [Common] 공통 스키마 및 유틸리티
├── pyproject.toml        # uv 워크스페이스 설정
└── README.md
```

---

## 🛠 기술 스택

*   **Language**: Python 3.10+
*   **Environment**: `uv` (Fast Python package manager)
*   **GUI**: PyQt6, OpenCV
*   **Backend**: FastAPI, Uvicorn, HTTPX (Async communication)
*   **AI/Vision**: MediaPipe, NumPy, Pydantic (Data validation)
*   **Database**: SQLite

---

## 📋 설치 및 실행 방법

### 1. 필수 요구사항
*   Python 3.10 이상
*   `uv` 설치: `pip install uv`

### 2. 프로젝트 초기화
```bash
# 의존성 설치 및 가상환경 구축
uv sync
```

### 3. 서버 실행 (각각의 터미널에서 실행)

**1) AI Head 서버 (Port 8001)**
```bash
cd apps/ai_head
uv run python src/ai_head/main.py
```

**2) 오케스트레이션 서버 (Port 8000)**
```bash
cd apps/orchestrator
uv run python src/orchestrator/main.py
```

**3) 클라이언트 프로그램**
```bash
cd apps/client
uv run python src/client/main.py
```

---

## 🔒 보안 설정
본 시스템은 서버 간 통신 보호를 위해 `X-API-Key` 헤더를 통한 간단한 인증을 수행합니다.
각 앱의 `.env` 파일에 동일한 `API_KEY`가 설정되어 있어야 정상적으로 통신이 가능합니다.

---

## 👥 팀 정보
*   **딥러닝 프로젝트 2조**

---
© 2026 Focus Monitor Project
