# [ 261109 2110 메인 워크스테이션 ] NWC 2.75 부가기능 검증 결과 및 코랄·푸가 작곡 엔진 아키텍처 진단 리포트

- [문서 식별자]: `261109_2110_nwc2_addon_verification_and_engine_audit_report`
- [작성 시각]: 2026-11-09 21:10 (KST)
- [작성 및 수행 주체]: `메인 워크스테이션` (`RTX5080Desktop-HC`)
- [대상 저장소]: `HyunchanAn/Like_Bach` (`feature/nwc-user-tool` 브랜치)

---

## 1. NWC 2.75 부가기능(User Tool) 검증 결과

1. [단위 및 통합 테스트 (`tests/test_nwc_tool.py`)]
   - `pytest -v` 실행 결과 8개 테스트 항목 전체 통과 (`8 passed`).
   - 검증 범위: 오선 좌표(`Pos`) <-> MIDI 피치 양방향 변환, `NWCTXT` 클립/전체 악보 파싱, 마디 경계 음표 분할 및 붙임줄(`^`) 처리, 4성부 코랄 확장(`FileText`), 4성부 푸가 확장(`FileText`), 선택 영역 클립 변환(`ClipText`), CLI `stdin`/`stdout` 무결성.

2. [NWC2 공식 변환기(`nwc-conv.exe`) 호환성 검증]
   - 생성된 4성부 코랄 및 푸가 `NWCTXT` 출력물을 `"C:\Program Files (x86)\Noteworthy Software\NoteWorthy Composer 2\nwc-conv.exe" NWCTXT INFO`에 파이핑하여 문법 오류 없이 `exit code 0`으로 정상 파싱됨을 확인함.

3. [정적 분석 (`ruff`)]
   - `python -m ruff check .` 실행 결과 경고 및 오류 0건 (`All checks passed!`).

---

## 2. 코랄(`v4`) 및 푸가(`v5`) 엔진 구조 진단 및 개선 과제

### 2.1 가중치 학습/추론 간 모델 스케일 불일치 결함
- [현상]: `src/v5/train_fugue.py`는 `BachTransformerConfig`의 기본값인 `model_scale='v5_pro'`(16레이어, 은닉 차원 768)로 모델을 초기화하여 `fugue_model_v5.pt`를 저장하는 반면, 추론 엔진인 `src/v5/neural_engine.py`의 `HybridFugueEngine.__init__`은 파일명이 `fugue_model_v5.pt`일 때 `scale = "v5_base"`(12레이어, 은닉 차원 512)로 초기화함.
- [조치 계획]: 학습 스크립트와 추론 엔진의 스케일 설정을 일치시키고 체크포인트 내에 `config` 메타데이터를 함께 저장·참조하도록 개선하여 차원 불일치 오류를 원천 차단함.

### 2.2 코랄(`v4`) 및 푸가(`v5`) 음악적 완성도 한계 진단
- [코랄 엔진 (`v4`)]:
  - 자유 연장 구간(Continuation)에서 소프라노 주제의 동기(Motif) 재현 및 프레이즈 호흡(반종지 -> 정격종지) 계획이 부재하여 후반부 선율이 산만해지는 경향이 있음.
  - 마디 내 강박/약박(Beat Position) 인식이 약해 비화성음(경과음, 보조음, 계류음)과 화성음의 구분이 모호함.
- [푸가 엔진 (`v5`)]:
  - 제시부(Exposition) 이후 에피소드(Episode) 구간에서 대주제(Countersubject) 보존 및 동형진행(Sequence) 가이드가 끊기면서 화성적 추진력이 약화됨.
  - 후반부에서 주제 재진입(Middle Entry / Stretto)이나 딸림음 페달 포인트(Dominant Pedal) 없이 갑자기 마지막 마디에서 종지하여 푸가 특유의 구조적 긴장감이 부족함.
  - 실시간 마스킹 필터가 직전 시점의 병행 5도/8도와 단순 불협화음만 억제하고, 이끈음(Leading Tone) 해결이나 7음 하행 해결 등 성부 진행(Voice Leading) 유도 로짓 보정이 부재함.

---

# [ 261109 2302 메인 워크스테이션 ] 코랄(V4)·푸가(V5) VRAM 누수 해결, 가중치 재학습 및 Hugging Face Hub / ONNX 배포 리포트

- [작성 시각]: 2026-11-09 23:02 (KST)
- [작성 및 수행 주체]: `메인 워크스테이션` (`RTX5080Desktop-HC`)
- [원격 모델 저장소]: `https://huggingface.co/chemahc94/like-bach-models`

## 1. 어텐션 병목 및 12.9GB VRAM 누수 해결

- [원인 규명]:
  1. `src/v5/models.py`의 `Head.__init__` 내부에서 실제로는 사용되지 않는 `tril` 버퍼(`4096 x 4096` FP32 = 67.1MB)가 `16레이어 x 12헤드 = 192개` 생성되어 모델 초기화 즉시 `12,885 MB (12.9GB)` VRAM을 점유함.
  2. `MultiHeadAttention.forward`가 헤드별 3차원 텐서 루프로 `F.scaled_dot_product_attention`을 호출하고 `RoPE` 캐시(`float32`) 곱셈으로 인해 `bfloat16`이 `float32`로 자동 승격되면서 `Math Attention` 폴백 및 Windows WDDM 공유 메모리 스와핑이 발생함.
- [해결 및 벤치마크]:
  - 미사용 `tril` 버퍼 제거로 모델 초기화 VRAM을 `13,320 MB -> 435.7 MB`로 절감함.
  - 4차원 텐서(`(B, n_head, T, head_size)`) 배치 연산, `RoPE` dtype 정렬, `cuDNN SDPA` 호환 드롭아웃(`0.0625 = 1/16`) 및 `bfloat16` autocast를 적용하여 스텝당 연산 시간을 `10,679.4 ms -> 292.7 ms` (36.5배 가속)로 단축함.

## 2. 재학습, ONNX 변환 및 Hugging Face Hub 업로드 결과

- [푸가 V5 (`data/processed/v5/`)]:
  - 10 Epochs 완주: `Average Train Loss = 0.8310`, `Val Loss = 0.8761`
  - ONNX Runtime 정합성 검증(`T=16, 64`): `max_abs_diff = 8.583e-06`
- [코랄 V4 (`data/processed/v4/`)]:
  - 3,000 Steps 완주 (`1,820,280 tokens` VRAM 상주): `Loss = 0.4585`
  - ONNX Runtime 정합성 검증(`T=16, 64`): `max_abs_diff = 2.098e-05`
- [Hugging Face Hub (`chemahc94/like-bach-models`) 업로드 아티팩트]:
  - `v4/bach_model.pt` (`sha256: 95c49b033d45...`, 389,345,895 bytes)
  - `v4/bach_model.safetensors` (`sha256: 03fcefa738a1...`, 389,249,464 bytes)
  - `v4/bach_model.onnx` (`sha256: 9a62f3c68cd1...`, 120,957,635 bytes)
  - `v4/tokenizer.pkl` (`sha256: 8e5f39c02ee3...`, 140,217 bytes)
  - `v5/fugue_model_v5.pt` (`sha256: dc4403109bf0...`, 457,208,896 bytes)
  - `v5/fugue_model_v5.safetensors` (`sha256: 27a725481284...`, 456,983,764 bytes)
  - `v5/fugue_model_v5.onnx` (`sha256: 2059e8c9bc86...`, 457,525,483 bytes)
  - `v5/fugue_vocab_v5.pkl` (`sha256: 467d84144b19...`, 3,179 bytes)
  - `release_manifest.json` 및 `README.md`

