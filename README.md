# Like Bach: Harmonic Generative Engine v4.0 (Master Bach)

바흐의 4성부 화성 체계를 완벽히 학습하여 주제 선율에 대해 정밀한 알토, 테너, 베이스 성부를 동시에 생성하는 4성부 통합 작곡 엔진입니다.

## Key Features (Master Bach v4.0)

- **4-Voice Integrated Harmony**: 소프라노 주제 입력 시 알토, 테너, 베이스를 유기적으로 동시 생성합니다. 
- **Time-Interleaved Learning**: [TIME] [V1] [V2] [V3] [V4] 구조의 인터리빙 토큰 학습을 통해 성부 간의 수평적 선율과 수직적 화성을 동시에 확보했습니다.
- **Precision Normalization**: 모든 시간 데이터를 0.125 단위로 정률화하여 데이터 노이즈를 제거하고 음악적 완결성을 높였습니다.
- **Hardware Optimization**: RTX 5080 환경에서 VRAM-Direct 및 Vectorized Indexing 기술을 적용하여 학습 효율을 극대화했습니다.
- **Advanced Training**: CosineAnnealingLR 스케줄러와 AdamW 옵티마이저를 통해 0.068 수준의 극도로 낮은 손실률(Loss)을 달성했습니다.

## Technical Architecture

- **Neural Engine**: Transformer-based Generative Model (25M Parameters)
- **Data Engineering**: MusicXML to Quantized Interleaved Token Pipeline
- **Legacy Support**: V3.1(2성부)과 V4.0(4성부)을 디렉토리 레벨에서 분리하여 독자적인 운용이 가능합니다.
- **Logic**: guided generation 방식을 통한 주제 기반 3성부 자동 작배(Realization)

## Getting Started

1. **Environment**:
   Python 3.10+ 및 CUDA 지원 GPU(RTX 4080/5080 권장) 필요

2. **Run Training**:
   ```bash
   python src/v4/train.py
   ```

3. **Inference (Neural Composition)**:
   ```bash
   python src/v4/neural_engine.py
   ```

## Development History

상세한 기술 개발 과정 및 최적화 이력은 development_log.txt에서 확인할 수 있습니다.
V3.1 이전의 레거시 코드는 src/v3 폴더에서 별도로 보존 중입니다.
