# AI Gaze Tracking Server

MediaPipe Face Mesh의 iris 랜드마크를 활용한 시선 추적 AI 서버입니다.

## 📌 개요

이 서버는 웹캠 이미지에서 사용자의 홍채(iris) 위치를 추출하고, 9-point 캘리브레이션을 통해 시선이 화면의 어느 위치를 응시하고 있는지 추정합니다. 화면 영역을 벗어나면 비집중 상태로 판정하며, 이탈 방향(좌/우/상/하)을 분류합니다.

## 🛠️ 기술 스택

- **FastAPI**: REST API 서버
- **MediaPipe Face Mesh**: 얼굴 랜드마크 + iris 추적 (468-477번 랜드마크)
- **scikit-learn**: 다항 회귀를 통한 iris → 화면 좌표 매핑
- **OpenCV**: 이미지 처리
- **NumPy**: 수치 연산

## 🎯 핵심 기능

### 1. Iris 위치 추출
- MediaPipe Face Mesh의 `refine_landmarks=True` 옵션으로 iris 랜드마크(468-477) 활성화
- 양쪽 눈의 iris 중심을 eye contour 대비 정규화 좌표(0~1)로 변환
- 양안 평균으로 안정성 향상

### 2. 9-Point 캘리브레이션
- 화면 9개 지점 `[(0.05,0.05), (0.5,0.05), (0.95,0.05), ...]`에서 각 5회 클릭 수집
- 총 45개 데이터포인트를 사용한 2차 다항 회귀(Polynomial Regression, degree=2)
- Ridge 회귀로 과적합 방지 (alpha=1.0)
- iris 좌표 → 화면 픽셀 좌표 매핑 모델 생성

### 3. 화면 이탈 감지
- 캘리브레이션된 화면 영역(left/right/top/bottom) 설정
- 추정된 시선 좌표가 영역 밖이면 `is_distracted=True`
- 이탈 방향 자동 분류:
  - 좌측: `gaze_x < calibLeft`
  - 우측: `gaze_x > calibRight`
  - 상단: `gaze_y < calibTop`
  - 하단: `gaze_y > calibBottom`

## 📡 API 엔드포인트

### `GET /health`
서버 헬스체크

**Response:**
```json
{
  "status": "ok",
  "service": "ai_gaze"
}
```

### `POST /set_calibration`
9-point 캘리브레이션 데이터 설정

**Request:**
```json
{
  "points": [
    {
      "iris_x": 0.45,
      "iris_y": 0.52,
      "screen_x": 96.0,
      "screen_y": 54.0
    },
    ...
  ],
  "screen_width": 1920,
  "screen_height": 1080
}
```

**Response:**
```json
{
  "status": "ok",
  "points_count": 45
}
```

### `POST /inference`
시선 추적 추론

**Request:** (Shared Schema의 `InferenceRequest`)
```json
{
  "image_base64": "iVBORw0KGgoAAAANS...",
  "session_id": "optional-session-id"
}
```

**Response:** (Shared Schema의 `InferenceResponse`)
```json
{
  "is_distracted": false,
  "status_message": "화면 응시 중",
  "gaze_data": {
    "iris_x": 0.48,
    "iris_y": 0.51,
    "gaze_x": 920.3,
    "gaze_y": 540.7,
    "is_on_screen": true,
    "calibrated": true,
    "gaze_cordinate": {
      "x": 920.3,
      "y": 540.7
    }
  }
}
```

**상태 메시지:**
- `"화면 응시 중"`: 시선이 화면 내부
- `"시선 이탈 (좌측)"`: 화면 왼쪽으로 이탈
- `"시선 이탈 (우측)"`: 화면 오른쪽으로 이탈
- `"시선 이탈 (상단)"`: 화면 위쪽으로 이탈
- `"시선 이탈 (하단)"`: 화면 아래쪽으로 이탈
- `"시선 감지됨 (캘리브레이션 필요)"`: 캘리브레이션 미완료
- `"얼굴을 감지할 수 없습니다"`: 얼굴 미검출

## 🔧 환경 변수 (.env)

```env
PORT=8005
API_KEY=your-secure-api-key-here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

## 🚀 실행 방법

### 독립 실행
```bash
cd apps/ai_gaze
uv run python src/ai_gaze/main.py
```

### 통합 실행 (루트에서)
```bash
python scripts/dev.py
```

## 📊 데이터 흐름

```
Client → POST /set_calibration → 캘리브레이션 모델 생성
Client → POST /inference → Iris 추출 → 모델 추론 → 화면 좌표 → 이탈 판정
```

## 🧩 AI Interface 통합

이 서버는 AI Interface에서 4번째 AI 모델로 통합되어 head/emotion/body와 함께 병렬로 호출됩니다.

**통합 로직:**
```python
is_distracted = (
    head_distracted OR
    emotion_distracted OR
    body_distracted OR
    gaze_distracted  # NEW
)
```

## 📝 참고 사항

- **캘리브레이션 저장**: 현재 서버 재시작 시 캘리브레이션이 초기화됩니다. 영구 저장이 필요하면 `joblib`로 모델 저장 기능 추가 가능.
- **정확도**: MediaPipe iris tracking은 전문 eye tracker 대비 정확도가 낮지만, 상대적 시선 이탈 감지에는 충분합니다.
- **호환성 유지**: `gaze_cordinate` 필드명의 오타("coordinate" → "cordinate")는 AI Interface의 주석 코드와의 호환성을 위해 의도적으로 유지됩니다.

## 🔗 관련 문서

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [외부 참고 코드](../../외부코드_gaze/README.md) - WebGazer 기반 원본 개념
