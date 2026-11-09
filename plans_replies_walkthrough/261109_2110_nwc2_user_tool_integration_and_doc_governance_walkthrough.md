# [ 261109 2110 메인 워크스테이션 ] Like_Bach NWC 2.75 부가기능(User Tool) 통합 및 SG 표준 문서 거버넌스 전환 Walkthrough

- [문서 식별자]: `261109_2110_nwc2_user_tool_integration_and_doc_governance_walkthrough`
- [작성 시각]: 2026-11-09 21:10 (KST)
- [작성 및 수행 주체]: `메인 워크스테이션` (`RTX5080Desktop-HC`)
- [대상 저장소]: `HyunchanAn/Like_Bach` (`feature/nwc-user-tool` 브랜치)

---

## 1. 작업 배경 및 목표

1. [NoteWorthy Composer 2 (NWC 2.75) 부가기능 연동]
   - 로컬 워크스테이션(`C:\Program Files (x86)\Noteworthy Software\NoteWorthy Composer 2\NWC2.exe`)에 설치된 NWC 2.75 환경에서 사용자가 소프라노 단선율 주제만 입력한 뒤 단축키(`Alt+F8` / `Ctrl+F8`)만으로 4성부(Soprano, Alto, Tenor, Bass) 코랄 및 푸가 악보를 즉시 완성할 수 있는 양방향 User Tool 브리지를 구축함.

2. [구버전 SDAD 프로토콜 제거 및 맥북 `SG_` 시리즈 표준 문서 체계 일원화]
   - 저장소 내에 잔존하던 `SDAD (SPEC-Driven AI Development)` 제어 파일(`GEMINI.md`, `sdad-state.yaml`, `review-findings.md`, `docs/`, `SPEC/`)을 전수 제거함.
   - 맥북 `SG_` 시리즈와 동일한 4대 표준 문서 디렉토리(`plans_replies_walkthrough/`, `plans_replies_walkthrough_archive/`, `reports/`, `reports_archive/`) 및 최상위 단일 개발일지(`development_log.txt`) 체제로 전환하고, 과거 커밋(`f51361e`)에서 덮어쓰여 유실되었던 `2026-04-12` ~ `2026-06-07` 개발일지 내역을 복원·통합함.

---

## 2. 세부 엔지니어링 구현 내역

### 2.1 NWC2 Text (`NWCTXT`) 양방향 파서 및 4성부 총보 빌더 (`src/nwc/nwctxt_bridge.py`)
- [지원 포맷]:
  - 전체 악보 모드: `!NoteWorthyComposer(2.75)` (`FileText`)
  - 선택 영역 클립 모드: `!NoteWorthyComposerClip(2.75,Single)` (`ClipText`)
- [음악 문법 및 오선 좌표 변환]:
  - 높은음자리표(`Treble`), 낮은음자리표(`Bass`), 가온음자리표(`Alto`), 테너음자리표(`Tenor`) 및 옥타브 변형(`Octave Up/Down`)의 오선지 상대 좌표(`Pos`)와 MIDI 피치 간 양방향 정밀 매핑을 구현함.
  - 조표(`|Key|Signature:...`) 및 마디 내 임시표(`#`, `b`, `n`, `x`, `v`) 상태를 추적하여 정확한 반음계 피치를 복원하고 생성함.
  - 마디 경계를 넘어가는 음표를 자동으로 분할하여 붙임줄(`^`)을 부여하고, 성부별 빈 박자 구간을 정규 쉼표로 자동 패딩하며, 8분/16분음표 단위의 꼬리 묶기(`Beam=First`, `Beam`, `Beam=End`)를 자동 산출함.

### 2.2 NWC2 User Tool CLI 엔트리포인트 및 자동 설치기 (`src/nwc/nwc_user_tool.py`, `scripts/install_nwc_tool.py`)
- [표준 입출력 격리]:
  - NWC2 User Tool은 `stdout` 전체를 악보 데이터로 파싱하므로, 내부 엔진(`HybridFugueEngine`, `NeuralBachEngine`, `music21`)의 콘솔 출력이 파이프를 오염시키지 않도록 `stdout`을 `stderr`로 리다이렉트한 뒤 최종 `NWCTXT`만 `stdout`으로 배출함.
- [3단계 추론/생성 파이프라인]:
  - `127.0.0.1:8000` FastAPI 서버 가동 시 REST 엔드포인트 연동 -> 오프라인 시 로컬 PyTorch 가중치 직접 추론 -> 가중치 파일 부재 시 조성·화성·대위법 규칙 기반 4성부 코랄 및 푸가(Tonal/Real Answer 제시부 및 대주제) 즉시 합성.
- [로컬 NWC2 환경 자동 등록]:
  - `C:\Users\sg\AppData\Local\Noteworthy Software\nwc2\nwc2UserTools.ini` 및 Windows 레지스트리(`HKCU\Software\Noteworthy Software\NoteWorthy Composer 2\Options`)에 4종의 도구를 자동 등록함:
    1. `Like Bach: 4-Voice Chorale (Full Score)`
    2. `Like Bach: 4-Voice Fugue (Full Score)`
    3. `Like Bach: Custom Generator (Prompt)`
    4. `Like Bach: Harmonize Selected Clip`
