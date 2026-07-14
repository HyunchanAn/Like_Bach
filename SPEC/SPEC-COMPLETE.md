# Like_Bach Specification

이 문서는 프로젝트의 단일 진실 공급원(SSOT) 역할을 수행하는 명세서입니다. 기존 `development_log.txt` 및 과거 기획안 파일들을 융합하여 작성되었습니다.

## 1. 개요 및 목적
Like_Bach 프로젝트(Bach-Style Polyphony Generative Engine, BPGE)는 바흐(J.S. Bach) 스타일의 수직적 화성(Chorale) 및 수평적 대위법(Fugue, Invention)을 자동 생성하고 렌더링하는 AI 작곡 엔진 및 웹 애플리케이션 스튜디오입니다. 

## 2. 시스템 아키텍처 (Dual-Engine)
1. **코랄(Chorale) 엔진 (`bach_model.pt`)**
   - 수직적 화성 진행(Vertical Harmony) 학습 중심.
2. **푸가(Fugue) 엔진 (`fugue_model.pt` / V5)**
   - 수평적 대위법 및 모방 구조(Imitative Counterpoint) 학습 중심.
   - 25M 파라미터 수준 유지, 컨텍스트 길이(Block Size) 4096+ 확장(또는 Flash Attention/RoPE 적용).
   - 데이터 증강: 12키 이조(Transposition) 적용 및 MAESTRO, GiantMIDI, Lakh MIDI 바로크 서브셋 결합.

## 3. 핵심 규칙 및 요구사항

### 3.1 AI 엔진 생성 규칙
- **하이브리드 추론 (Hybrid Inference)**:
  - 4성부 푸가 제시부(Exposition) 뼈대는 결정론적 스케줄링 적용.
  - V1(소프라노): 0박자 진입
  - V2(알토): +N박자 5도 하강 응답(Answer) 진입
  - V3(테너): +2N박자 옥타브 하강 주제 진입
  - V4(베이스): +3N박자 12도 하강 응답 진입
- **음악적 문법(Grammar)**:
  - `[ROMAN_I]`, `[ROMAN_V7]` 등 기능 화성 토큰 적용.
  - 마디 카운트다운 `[REMAIN_N]` 및 박자표 `[TS_X]` 적용.
- **사후 검증(Evaluation) 루프**:
  - 병행 5도/8도, 장7도/단2도 불협화음 점수 감점 (기준: 평균 90점 이상).
  - 최대 3회(또는 5회) 재시도 후 Fail-safe 발동 시 3도/6도 협화음으로 강제 변환.
  - **유니즌 방지**: 동일 시점(Offset)에 서로 다른 성부가 동일 음정(Pitch)을 소리 내지 않도록 마스킹.
  - **침묵(Silence) 제한**: 곡 전개 중 모든 4성부가 동시에 쉬는 구간(Gap) 원천 방지 (직전 음표 Sustain).
  - **종결부(Cadence) 특수 규칙**: 종결을 알리는 `[FINAL]` 토큰이 나타난 가장 마지막 마디에만 온음표(Whole Note) 생성을 허용함.

### 3.2 프론트엔드 UI/UX (Like Bach Studio)
- **스택**: Vite + React + TypeScript + VexFlow + Tone.js.
- **렌더링**:
  - `VexFlow getLineForY(y)` 기반 정밀 피치(Pitch) 매핑 및 툴팁 호버 지원.
  - `joinVoices([Soprano, Alto])` 및 `joinVoices([Tenor, Bass])`로 분리 포맷팅하여 가로 엇갈림(Collision Shift) 현상 방지.
  - 4/4 박자 초과 방지 및 남은 박자 계산 커서 자동 렌더링.
- **단축키 및 입력 (NWC2 스타일)**:
  - 1~6(음표 길이), 방향키(음정), 엔터(입력), 백스페이스(삭제).
- **스트리밍 아키텍처 (SSE)**:
  - 단일 음표가 아닌 **마디(Bar)** 단위 청크로 패킹하여 프론트엔드로 스트리밍하여 레이턴시를 감추고 화면이 밀리는 현상 방지.
