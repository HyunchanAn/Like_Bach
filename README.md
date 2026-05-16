# Like Bach: Harmonic Generative Engine v4.5 (Refinement Era)

바흐의 4성부 화성 체계를 완벽히 학습하고, `DeepBach` 및 `BachBot`의 선진 알고리즘을 이식하여 음악적 논리성과 구조적 완결성을 극대화한 통합 작곡 엔진입니다.

## Key Features (Master Bach v4.5)

- **4-Voice Integrated Harmony**: 소프라노 주제 입력 시 알토, 테너, 베이스를 유기적으로 동시 생성합니다. 
- **Functional Harmony Awareness**: 로마자 화성 기호(`[ROMAN_I]`, `[ROMAN_V7]` 등)를 명시적으로 학습하여 화성적 개연성과 음악적 정통성을 확보했습니다.
- **Structural Measure Control**: 마디 카운트다운(`[REMAIN_N]`) 및 종지(`[FINAL]`) 토큰을 통해 사용자가 원하는 길이를 제어하고 자연스러운 마무리가 가능합니다.
- **Iterative Refinement (Gibbs Sampling)**: 생성된 결과를 모델이 다시 훑으며 화성을 교정하는 알고리즘을 도입하여 수직적 화성의 정밀도를 높였습니다.
- **Precision Normalization**: 모든 시간 데이터를 0.125 단위로 정률화하고 페르마타(`[FERM]`), 박자표(`[TS_X]`) 정보를 통합했습니다.
- **Hardware Optimization**: RTX 5080 환경에서 VRAM-Direct 및 Vectorized Indexing 기술을 통해 초고속 학습과 실시간 생성을 구현했습니다.

## Technical Architecture

- **Neural Engine**: Transformer-based Generative Model (25M Parameters) with Iterative Refinement
- **Sampling Strategy**: Guided Generation + Gibbs-style Refinement Loop
- **Data Engineering**: music21 기반의 고도화된 로마자 화성 분석 및 인터리빙 토큰 파이프라인
- **Legacy Support**: v3.1, v4.0(구버전)을 별도 보존하며 v4.5 고도화 엔진으로 세대 교체 완료

## Getting Started

1. **Environment**:
   Python 3.10+ 및 CUDA 지원 GPU(RTX 4080/5080 권장) 필요

2. **Run Advanced Preprocessing**:
   ```bash
   python src/v4/preprocess.py
   ```

3. **Run Training**:
   ```bash
   python src/v4/train.py
   ```

4. **Inference (Neural Composition)**:
   ```bash
   python src/v4/neural_engine.py
   ```

## Development History

상세한 기술 개발 과정 및 최적화 이력은 development_log.txt에서 확인할 수 있습니다.
최신 고도화 알고리즘 연구 내용은 feature/algorithm-enhancement 브랜치에 구현되어 있습니다.
