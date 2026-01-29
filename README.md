# Focus Monitor

딥러닝 기반의 실시간 업무 집중 모니터링 및 코칭 시스템입니다. 웹캠을 통해 사용자의 고개 각도(Head Pose)를 분석하여 집중 상태를 판단하고, 세션 종료 후 LLM을 통해 개인화된 피드백을 제공합니다.

## 📌 주요 기능

*   **실시간 집중도 모니터링**: 웹캠을 사용하여 사용자의 얼굴 각도(Pitch, Yaw, Roll)를 실시간으로 추정합니다.
*   **산만함 감지**: 설정된 임계값을 벗어나는 움직임이 감지되면 산만함(Distraction)으로 기록합니다.
*   **집중도 점수 계산**: 전체 세션 시간 대비 집중 시간과 산만함 횟수를 기반으로 집중 점수를 산출합니다.
*   **AI 맞춤형 피드백**: 세션 종료 후, 수집된 데이터를 바탕으로 LLM(Ollama - Llama 3.1등)이 격려 멘트와 구체적인 행동 교정 팁을 제공합니다.
*   **세션 기록 관리**: 과거 집중 세션의 기록을 데이터베이스에 저장하고 조회할 수 있습니다.
*   **직관적인 UI**: PyQt6 기반의 데스크탑 애플리케이션으로 제공됩니다.

## 🛠 기술 스택

*   **Language**: Python 3.10+
*   **GUI**: PyQt6
*   **Computer Vision**: OpenCV, MediaPipe, NumPy
*   **LLM Integration**: Ollama (Local LLM), Llama 3.1
*   **Database**: SQLite

## 📋 설치 및 실행 방법

### 1. 필수 요구사항
*   Python 3.10 이상
*   웹캠 (노트북 내장 카메라 또는 USB 카메라)
*   **Ollama**: 로컬 LLM 구동을 위해 [Ollama](https://ollama.com/)가 설치되어 있어야 합니다.

### 2. 프로젝트 설치

```bash
# 저장소 복제
git clone <repository-url>
cd deeplearning-repo-2
```

**방법 1: `uv` 사용 (권장)**
`uv`는 기존 `pip`보다 훨씬 빠른 Python 패키지 관리 도구입니다.

```bash
# uv 설치 (이미 설치된 경우 생략)
pip install uv

# 초기화 및 의존성 동기화
uv init
uv sync
```

**방법 2: `pip` 사용 (기본)**
`uv`를 사용하지 않을 경우 기존 방식을 사용할 수 있습니다.

```bash
pip install -r requirements.txt
# 또는
pip install .
```
*`requirements.txt`가 없는 경우 `app/pyproject.toml`을 참고하여 다음 패키지들을 설치하세요:*
```bash
uv add matplotlib mediapipe numpy ollama opencv-python pyqt6 requests
```

### 3. LLM 모델 준비
이 프로젝트는 `llama3.1:8b` 모델을 기본으로 사용합니다. 터미널에서 다음 명령어로 모델을 다운로드하세요.

```bash
ollama pull llama3.1:8b
```
*실행 전 `ollama serve` 등으로 Ollama 서비스가 백그라운드에서 실행 중이어야 합니다.*

### 4. 실행

```bash
# app/src 폴더 내의 main.py 실행
python app/src/main.py
```

## 📂 프로젝트 구조

```
deeplearning-repo-2/
├── README.md               # 프로젝트 설명 파일
├── app/
│   ├── pyproject.toml      # 프로젝트 의존성 및 설정
│   └── src/
│       ├── main.py         # 애플리케이션 진입점 (GUI 실행)
│       ├── core/           # 핵심 로직 (세션 관리, 비디오 스레드 등)
│       ├── database/       # 데이터베이스 처리
│       ├── llm/            # LLM 클라이언트 (Ollama 연동)
│       ├── ui/             # UI 구성 요소 및 페이지
│       └── vision/         # 컴퓨터 비전/딥러닝 모델 (MediaPipe FaceMesh)
└── training/               # 모델 학습 관련 리소스
```

## ⚠️ 문제 해결

*   **카메라가 작동하지 않는 경우**: 다른 프로그램에서 카메라를 사용 중인지 확인하거나, `opencv` 관련 권한 설정을 확인하세요.
*   **LLM 응답이 없는 경우**: 터미널에서 `ollama list`를 입력하여 `llama3.1:8b` 모델이 있는지 확인하고, Ollama 서버가 실행 중인지 확인하세요.

## 👥 팀 정보
*   딥러닝 프로젝트 2조

---
© 2026 Focus Monitor Project