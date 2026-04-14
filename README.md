# Like Bach: Polyphonic Generative Engine v2.6

바흐 스타일의 2성부 인벤션 및 대위법 선율을 실시간으로 생성하고 연주하는 인터랙티브 작곡 엔진입니다.

## 🎻 Key Features (Pro v2.6)

- **Interactive Subject Input**: 브라우저 오선보 위에 직접 음표를 찍어 나만의 바흐풍 테마를 생성할 수 있습니다. 4/4박자 마디 구분을 완벽 지원합니다.
- **2-Part Counterpoint AI**: 입력된 주제에 맞추어 리듬적으로 독립적이고 화성적으로 풍부한 대위선율(Countersubject)을 실시간으로 작곡합니다.
- **Perfect Sync Visualization**: 픽셀 기반 추적 알고리즘을 통해 재생 바와 연주되는 음표가 1:1로 완벽하게 일치합니다.
- **Flowing Score Experience**: 고정된 재생 바와 부드럽게 가로로 흐르는 악보(Scrolling Score)를 통해 리듬 게임이나 전문 사보 프로그램과 같은 사용자 경험을 제공합니다.
- **Zero-Error Calibration**: 사용자 디스플레이의 해상도를 감지하여 입력 좌표를 자동으로 교정하는 자가 측정 시스템이 탑재되어 있습니다.

## 🛠 Tech Stack

- **Frontend**: Vanilla JS (UI), VexFlow 4.2 (Notation Rendering), Tone.js (Audio Synthesis)
- **Backend**: FastAPI (Python), music21 (Music Analysis & Composition Engine)
- **Design**: Dark Mode with Premium Gold Aesthetics

## 🚀 Getting Started

1. **Start Backend**:
   ```bash
   uvicorn src.main:app --reload
   ```
2. **Open Frontend**:
   브라우저에서 `ui/index.html` 파일을 엽니다.

3. **Compose**:
   오선보를 클릭하여 주제를 입력하고 `[대위법적 전곡 작곡 및 연주]` 버튼을 누르세요.

## 📝 Development Log
상세한 개발 히스토리는 `development_log.txt`에서 확인할 수 있습니다.
