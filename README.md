# Focus Monitor (업무 집중도 모니터링 시스템)

딥러닝 기반의 실시간 업무 집중 모니터링 및 코칭 시스템입니다. 웹캠을 통해 사용자의 상태(고개 각도, 표정, 자세)를 분석하여 집중 상태를 판정하고, 모니터링 종료 후 AI 피드백을 제공합니다.

---

## 🏗 시스템 아키텍처

본 프로젝트는 확장성과 효율성을 위해 **Microservices Architecture (MSA)** 스타일의 구조를 채택하고 있습니다.

```mermaid
graph TD
    Client[Client GUI - PyQt6] <--> Orch[Orchestrator - FastAPI]
    Orch <--> AI_Head[AI Head Pose - YOLOv8 Pose]
    Orch <--> AI_Emotion[AI Emotion - FastAPI]
    Orch <--> AI_Body[AI Upper Body - FastAPI]
    Orch <--> DB[DB Server - SQLite]
    DB <--> LLM[LLM Server - Feedback AI]
```

1.  **Client (PyQt6)**: 실시간 웹캠 프리뷰, 주기적 이미지 캡처 및 전송, 비집중 알림 시각화.
2.  **Orchestrator (FastAPI)**: 요청 분배, 다중 모델 결과 집계(Aggregation), 최종 비집중 판정 알고리즘 수행.
3.  **AI Head (YOLOv8 Pose)**: YOLOv8 Pose 모델을 이용한 실시간 고개 각도 추정 및 사용자 추적(Tracking).
4.  **Feedback AI**: 세션 데이터 기반 맞춤형 코멘트 생성 (Ollama/Llama 3.1).

---

## 📌 주요 기능

*   **실시간 집중도 모니터링**: 3초 주기로 사용자의 상태를 분석하여 집중 여부 판정.
*   **고도화된 Head Pose 분석 (YOLOv8)**:
    *   **Pitch, Yaw, Roll 추정**: 안면 키포인트를 활용한 정밀한 각도 계산.
    *   **타겟 잠금 및 추적 (Target Locking)**: 여러 사람이 포착되어도 모니터링 대상을 고정하여 지속적으로 추적.
    *   **이탈/산만함 감지**: 고개 숙임, 옆보기 등 설정된 임계값을 벗어난 동작 실시간 감지.
*   **즉각적인 피드백**: 비집중 상태 감지 시 화면 오버레이를 통해 주의 환기 유도.
*   **보안 강화**: API Key 인증 및 요청 데이터 크기 제한을 통한 안정적인 서버 운영.

---

## 📂 프로젝트 구조 (uv Workspace)

```text
deeplearning-repo-2/
├── apps/
│   ├── client/           # [UI] PyQt6 데스크탑 애플리케이션
│   ├── orchestrator/     # [Control] 중앙 제어 서버
│   ├── ai_head/          # [Vision] YOLOv8 기반 Head Pose 분석 서버
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
*   **AI/Vision**: **YOLOv8 Pose**, Ultralytics, NumPy, Pydantic
*   **Security**: API Key Header Auth, Request Size Limit Middleware

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

### 3. 서버 실행 (단일 컴퓨터 / 여러 터미널 테스트)

한 대의 컴퓨터에서 여러 개의 터미널을 열어 테스트할 경우의 명령어입니다.

**1) AI Head 서버 실행 (Port 8001)**
```bash
cd apps/ai_head
uv run python src/ai_head/main.py
```

**2) 오케스트레이션 서버 실행 (Port 8000)**
```bash
cd apps/orchestrator
uv run python src/orchestrator/main.py
```

**3) 클라이언트 프로그램 실행**
```bash
cd apps/client
uv run python src/client/main.py
```

### 4. 분산 환경 설정 (여러 대의 컴퓨터에서 실행 시)

각 서비스를 서로 다른 컴퓨터에서 실행할 경우, 네트워크 연결을 위해 다음 설정을 수행해야 합니다.

#### 1) 공통 준비 사항
*   모든 컴퓨터는 **동일한 네트워크(WiFi/유선 LAN)**에 연결되어 있어야 합니다.
*   각 컴퓨터의 **내부 IP 주소**를 확인합니다. (Windows: `ipconfig`, Linux/Mac: `ifconfig` 또는 `ip addr`)
*   방화벽에서 해당 포트(8000, 8001 등)를 개방해야 합니다.
    *   **Ubuntu 24.04 (UFW) 설정 방법**:
        1.  방화벽 상태 확인: `sudo ufw status`
        2.  필요한 포트 허용:
            *   `sudo ufw allow 8000/tcp` (Orchestrator용)
            *   `sudo ufw allow 8001/tcp` (AI Head용)
        3.  방화벽 활성화 (꺼져 있는 경우): `sudo ufw enable`
        4.  적용된 규칙 확인: `sudo ufw status numbered`

#### 2) 서비스별 `.env` 설정 예시

상황에 따라 각 서비스의 `.env` 파일을 다음과 같이 수정합니다.

**[시나리오 1: 모든 서비스를 서로 다른 컴퓨터에서 실행 (A, B, C)]**

*   **컴퓨터 A (AI Head 서버, IP: 192.168.0.10)**: 기본 실행 (Port 8001)
*   **컴퓨터 B (오케스트레이션 서버, IP: 192.168.0.20)**: `apps/orchestrator/.env` 수정
    ```env
    AI_HEAD_URL=http://192.168.0.10:8001/inference
    API_KEY=your-secret-key
    ```
*   **컴퓨터 C (클라이언트 GUI)**: `apps/client/.env` 수정
    ```env
    ORCHESTRATOR_URL=http://192.168.0.20:8000/inference
    API_KEY=your-secret-key
    ```

**[시나리오 2: 서버들은 한 대에, 클라이언트는 다른 한 대에 실행 (A, B)]**

가장 권장되는 구성입니다. (서버 사양이 충분할 경우)

*   **컴퓨터 A (서버 통합, IP: 192.168.0.10)**
    *   **AI Head**: 기본 실행 (Port 8001)
    *   **Orchestrator**: `apps/orchestrator/.env` 수정 (같은 컴퓨터이므로 `localhost` 사용)
        ```env
        AI_HEAD_URL=http://localhost:8001/inference
        API_KEY=your-secret-key
        ```
*   **컴퓨터 B (클라이언트 GUI)**: `apps/client/.env` 수정
    ```env
    ORCHESTRATOR_URL=http://192.168.0.10:8000/inference
    API_KEY=your-secret-key
    ```

#### 3) 실행 순서
1.  **AI Head 서버** 실행
2.  **오케스트레이션 서버** 실행
3.  **클라이언트 GUI** 실행 (서버가 모두 뜬 후 실행)

---

## 🔒 보안 및 환경 설정
*   **API 인증**: 모든 서버 간 통신은 `X-API-Key` 헤더 인증을 거칩니다.
*   **환경 변수**: 각 앱의 `.env` 파일에 `API_KEY`가 반드시 설정되어 있어야 합니다.
*   **데이터 제한**: 서버 안정성을 위해 10MB 이상의 이미지 데이터 전송은 차단됩니다.

---

## 👥 팀 정보
*   **딥러닝 프로젝트 2조**

---
© 2026 Focus Monitor Project
