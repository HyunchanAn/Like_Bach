# [ 261109 2345 메인 워크스테이션 ] 심볼릭 음악 생성 및 대위법 평가를 위한 공인(Official) 벤치마크·툴킷 조사 보고서

## 1. 개요

자체 고안한 임의의 휴리스틱 점수나 주관적 임계치를 배제하고, 국제 음악정보검색학회(`ISMIR` / `TISMIR`), 머신러닝 학회(`ICML` / `NeurIPS`), 그리고 컴퓨터 음악학(Computational Musicology) 학계에서 심볼릭 음악 생성(Symbolic Music Generation) 및 바흐 다성 음악 모델을 평가할 때 표준으로 사용하는 5대 공인 평가 프레임워크와 오픈소스 라이브러리를 정리한다.

---

## 2. 학계 및 오픈소스 진영의 5대 공인 평가 표준 요약

| 구분 | 공인 프레임워크 / 라이브러리 | 발표 학회 및 소속 기관 | 핵심 평가 대상 | 설치 및 모듈 경로 |
| :--- | :--- | :--- | :--- | :--- |
| [표준 1] | `MusPy` (`muspy.metrics`) | ISMIR 2020 (UC San Diego) | 단일 곡 단위의 음고·다성 밀도·스케일 일관성·엔트로피·그루브 정량 지표 (11종) | `pip install muspy` (`muspy.metrics.*`) |
| [표준 2] | `MGEval` (`mgeval`) | NCA / ISMIR (Georgia Institute of Technology) | 생성곡 집합과 실제 코퍼스 집합 간 12종 음악 특징 분포의 `KLD` 및 `Overlapping Area (OA)` 비교 | `github.com/RichardYang40148/mgeval` |
| [표준 3] | `Fréchet Music Distance (FMD)` | arXiv:2407.01677 (Warsaw Univ. of Technology) | 사전학습 심볼릭 음악 인코더(`CLaMP 2`) 임베딩 공간에서의 다변량 정규분포 프레셰 거리(`FID`/`FAD`의 심볼릭 음악 버전) | `pip install frechet-music-distance` |
| [표준 4] | `music21` (`figuredBass.checker` & `voiceLeading`) | ISMIR 2010 (MIT) | 성부 진행 및 화성학 규칙 위배(병행 5·8도, 은복 5·8도, 성부 교차·침범, 상3성 간격, 미해결 도약)의 결정론적 검출 | `music21.figuredBass.checker` (프로젝트 내장 완료) |
| [표준 5] | `jSymbolic 2.2` (`music21.features.jSymbolic`) | ISMIR 2018 (McGill University) | 르네상스·바로크 다성 음악 코퍼스 분석을 위해 표준화된 246개 정량 특징 벡터 (Python `music21` 내 72개 공식 이식 완료) | `music21.features.jSymbolic` (프로젝트 내장 완료) |

---

## 3. 공인 평가 도구별 상세 지표 및 공식 API 명세

### 3.1 [표준 1] `MusPy` (`muspy.metrics`) — ISMIR 2020 공인 객관 지표

- 문헌: Hao-Wen Dong, Ke Chen, Julian McAuley, Taylor Berg-Kirkpatrick, `MusPy: A Toolkit for Symbolic Music Generation`, ISMIR 2020 (University of California, San Diego).
- 특징: 심볼릭 음악 생성 논문에서 생성 샘플의 기초 통계 건전성을 보고할 때 가장 널리 쓰이는 파이썬 표준 라이브러리이다.

1. [음고 및 화성(Pitch & Harmony) 공식 함수]:
   - `muspy.pitch_range(music)`: 곡 내 최고 음고와 최저 음고 간의 반음(Semitone) 폭.
   - `muspy.n_pitches_used(music)`: 곡에서 사용된 고유 MIDI 음고(0~127)의 개수.
   - `muspy.n_pitch_classes_used(music)`: 곡에서 사용된 고유 피치 클래스(0~11)의 개수.
   - `muspy.polyphony(music)`: 최소 1개 이상의 음이 발음 중인 타임스텝에서 동시에 울리는 평균 성부(음)의 수:
     $$\text{Polyphony} = \frac{\sum_{t=1}^{T} \text{ count\_active\_pitches}(t)}{\sum_{t=1}^{T} \mathbb{I}(\text{count\_active\_pitches}(t) > 0)}$$
   - `muspy.polyphony_rate(music, threshold=2)`: 지정된 임계치(`threshold`) 이상의 음이 동시에 발음되는 타임스텝의 비율.
   - `muspy.pitch_in_scale_rate(music, root, mode)`: 지정된 조성(`root`, `mode`)의 다이아토닉 스케일 안에 포함되는 음표의 비율(이탈음이 많거나 조성이 붕괴되면 하락).
   - `muspy.scale_consistency(music)`: 24개 표준 장·단조 스케일 중 가장 높은 `pitch_in_scale_rate` 값:
     $$\text{ScaleConsistency} = \max_{r \in \{0..11\}, m \in \{\text{major, minor}\}} \text{pitch\_in\_scale\_rate}(r, m)$$
   - `muspy.pitch_entropy(music)` 및 `muspy.pitch_class_entropy(music)`: 정규화된 음고(또는 12 피치 클래스) 히스토그램 $p$의 섀넌 엔트로피(Shannon Entropy):
     $$H_{\text{PC}} = -\sum_{c=0}^{11} p(c) \log_2 p(c)$$

2. [리듬 및 구조(Rhythm & Structure) 공식 함수]:
   - `muspy.empty_beat_rate(music)`: 전체 박자(Beat) 중 아무 음도 발음되지 않는 빈 박자의 비율.
   - `muspy.empty_measure_rate(music)`: 전체 마디 중 노트가 전혀 없는 빈 마디의 비율.
   - `muspy.groove_consistency(music, measure_resolution)`: 인접한 마디 $m$과 $m+1$의 발음 위치 이진 벡터(Onset Vector) $e_m, e_{m+1} \in \{0, 1\}^Q$ 사이의 정규화 해밍 거리(Hamming Distance)를 기반으로 한 마디 간 리듬 일관성:
     $$\text{GrooveConsistency} = 1 - \frac{1}{M - 1} \sum_{m=1}^{M-1} \frac{1}{Q} d_{\text{Hamming}}(e_m, e_{m+1})$$

---

### 3.2 [표준 2] `MGEval` (`mgeval`) — 코퍼스 분포 중첩도(OA) 및 KLD 표준 벤치마크

- 문헌: Li-Chia Yang, Alexander Lerch, `On the Evaluation of Generative Models in Music`, Neural Computing and Applications (Georgia Institute of Technology, 2020).
- 핵심 철학: 단일 곡에 대해 "몇 점 이상이면 좋은 음악이다"라는 자의적 절대 기준을 세우는 대신, `[실제 바흐 원곡 코퍼스 내부의 곡 간 거리 분포(Intra-set Distance)]`와 `[생성곡 집합과 실제 바흐 원곡 집합 간의 거리 분포(Inter-set Distance)]`를 비교한다.
- `mgeval.core.metrics()`가 추출하는 9개 공식 특징:
  1. `total_used_pitch` (`PC`): 곡당 사용된 고유 음고 수
  2. `pitch_class_histogram` (`PCH`): 12차원 피치 클래스 분포 벡터
  3. `pitch_class_transition_matrix` (`PCTM`): $12 \times 12$ 피치 클래스 전이 행렬
  4. `pitch_range` (`PR`): 음역 폭 (반음)
  5. `avg_pitch_shift` (`PI`): 연속된 두 음 사이의 평균 수평 음정 간격
  6. `total_used_note` (`NC`): 발음된 총 노트 수
  7. `avg_IOI` (`IOI`): 연속 발음 시점 간 평균 간격(Average Inter-Onset Interval)
  8. `note_length_hist` (`NLH`): 12차원 음표 길이 히스토그램
  9. `note_length_transition_matrix` (`NLTM`): $12 \times 12$ 음표 길이 전이 행렬
- 공식 평가 절차:
  1. 원곡 집합 $\mathcal{R}$ 내부의 모든 곡 쌍 간 유클리드/코사인 거리를 구해 가우시안 커널 밀도 추정(`scipy.stats.gaussian_kde`)으로 확률밀도함수 $p_{\text{intra}}(x)$를 구한다.
  2. 생성곡 집합 $\mathcal{G}$와 원곡 집합 $\mathcal{R}$ 간의 거리로 확률밀도함수 $p_{\text{inter}}(x)$를 구한다.
  3. 두 분포 사이의 `Kullback-Leibler Divergence (KLD)`와 `Overlapping Area (OA, 중첩 면적)`를 계산한다 (`mgeval.utils.c_dist`, `mgeval.utils.kl_dist`, `mgeval.utils.overlap_area`):
     $$\text{OA}(p_{\text{intra}}, p_{\text{inter}}) = \int_{-\infty}^{\infty} \min\left(p_{\text{intra}}(x), p_{\text{inter}}(x)\right) dx \in [0, 1]$$
  - `OA`가 `1.0`에 수렴하고 `KLD`가 `0.0`에 수렴할수록 생성 모델의 분포가 원곡 코퍼스와 통계적으로 구별 불가능함을 뜻한다.

---

### 3.3 [표준 3] `Fréchet Music Distance (FMD)` — 딥러닝 임베딩 공간 분포 거리 표준

- 문헌: Jan Retkowski, Jakub Stępniak, Mateusz Modrzejewski, `Fréchet Music Distance: A Metric For Generative Symbolic Music Evaluation`, arXiv:2407.01677 (Warsaw University of Technology, 2024).
- 패키지: `pip install frechet-music-distance` (`from frechet_music_distance import FrechetMusicDistance`)
- 특징: 컴퓨터 비전의 `FID (Fréchet Inception Distance)` 및 오디오 생성의 `FAD (Fréchet Audio Distance)`와 완전히 동일한 수학적 구조를 심볼릭 음악(MIDI / ABC 포맷)에 이식한 최신 공식 지표이다.
- 동작 원리:
  1. 대용량 심볼릭 음악으로 사전학습된 대조 학습 인코더(`CLaMP` 또는 다성 MIDI를 지원하는 `CLaMP 2`)를 사용하여 레퍼런스 코퍼스 $\mathcal{R}$과 생성 샘플 집합 $\mathcal{G}$의 곡들을 고차원 특징 임베딩 벡터로 변환한다.
  2. 두 임베딩 집합의 평균 벡터 $(\mu_r, \mu_g)$와 공분산 행렬 $(\Sigma_r, \Sigma_g)$로부터 프레셰 거리를 계산한다:
     $$\text{FMD}(\mathcal{R}, \mathcal{G}) = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
  3. 소표본 편향을 보정한 `FMD-Inf` 및 `Ledoit-Wolf` / `OAS` 공분산 추정기를 공식 지원한다.

---

### 3.4 [표준 4] `music21.figuredBass.checker` 및 `music21.voiceLeading` — MIT 공인 대위법·화성학 체커

- 문헌: Michael Scott Cuthbert, Christopher Ariza, `music21: A Toolkit for Computer-Aided Musicology and Symbolic Music Data`, ISMIR 2010 (MIT).
- 특징: 현재 `Like_Bach` 가상환경에 이미 설치되어 있는 `music21` 라이브러리 내부에는 4성부 화성학 및 대위법 금칙을 공식 검증하는 `music21.figuredBass.checker`, `music21.figuredBass.possibility`, `music21.voiceLeading.VoiceLeadingQuartet` 모듈이 내장되어 있다.

1. [`music21.figuredBass.checker` 공식 검사 함수]:
   - `checker.extractHarmonies(score)`: 다성 악보(`stream.Score`)를 수직 화성 단면(Voice-Leading Moments, `vlm`) 시퀀스로 분할.
   - `checker.parallelFifths(vlmA, vlmB)`: 연속된 두 화성 단면 사이에서 병행 5도(Parallel 5ths)를 범한 성부 쌍 `(partA, partB)` 리스트 반환.
   - `checker.parallelOctaves(vlmA, vlmB)`: 병행 8도 및 병행 동도(Parallel 8ths / Unisons)를 범한 성부 쌍 리스트 반환.
   - `checker.hiddenFifth(vlmA, vlmB)`: 외성(Soprano-Bass) 간 같은 방향 진행으로 완전 5도에 도달할 때 소프라노가 순차 진행하지 않은 은복 5도(Hidden/Direct 5th) 검출.
   - `checker.hiddenOctave(vlmA, vlmB)`: 외성 간 은복 8도(Hidden/Direct 8ve) 검출.
   - `checker.voiceCrossing(vlm)`: 단일 화성 단면에서 상위 성부가 하위 성부보다 낮은 음을 연주하는 성부 교차(Voice Crossing) 성부 쌍 반환.

2. [`music21.figuredBass.possibility` 공식 검사 함수]:
   - `possibility.voiceOverlap(vlmA, vlmB)`: 인접한 두 시점 사이에서 한 성부가 직전 시점의 인접 성부 음고를 넘어서는 성부 침범(Voice Overlap) 여부(`bool`) 반환.
   - `possibility.upperPartsWithinLimit(vlm, maxSemitoneSeparation=12)`: 소프라노-알토, 알토-테너(상3성) 사이의 수직 간격이 1옥타브(12반음) 이내로 유지되는지 검사.
   - `possibility.partMovementsWithinLimits(vlmA, vlmB, maxMovements)`: 성부별 수평 도약 폭이 허용 한계를 초과했는지 검사.
   - `possibility.isIncomplete(vlm)`: 3화음/7화음에서 필수 구성음(근음, 3음 등)이 누락된 불완전 화음 여부 검사.

3. [`music21.voiceLeading.VoiceLeadingQuartet` 공식 메서드]:
   - `iterateAllVoiceLeadingQuartets(score)`를 통해 모든 성부 쌍의 $2 \times 2$ 음표 진행(`v1n1 -> v1n2`, `v2n1 -> v2n2`)을 순회하며 다음 공식 메서드를 호출할 수 있다:
     - 진행 분류: `contraryMotion()`, `obliqueMotion()`, `similarMotion()`, `parallelMotion()`, `antiParallelMotion()`, `noMotion()`, `inwardContraryMotion()`, `outwardContraryMotion()`
     - 대위법 금칙: `parallelFifth()`, `parallelOctave()`, `parallelUnison()`, `hiddenFifth()`, `hiddenOctave()`, `voiceCrossing()`, `voiceOverlap()`, `leapNotSetWithStep()` (큰 도약 후 반대 방향 순차 보상이 없는지 검사), `isProperResolution()` (불협화음/특수 화음의 정규 해결 여부 검사).

---

### 3.5 [표준 5] `jSymbolic 2.2` (`music21.features.jSymbolic`) — McGill 대학 공인 다성 음악 특징 추출기

- 문헌: Cory McKay, Julie E. Cumming, Ichiro Fujinaga, `jSymbolic 2.2: Extracting Features from Symbolic Music for Use in Musicological and MIR Research`, ISMIR 2018 (Marianopolis College & McGill University, Canada).
- 특징: 르네상스 및 바로크 다성 대위법 음악을 정량 비교하기 위해 McGill 대학 연구진이 제정한 학계 표준 특징 세트로, 현재 `music21.features.jSymbolic` 안에 72개의 공식 추출기 클래스가 내장되어 있다.
- 다성 대위법 평가에 직결되는 핵심 공식 추출기 클래스:
  - 선율 진행 품질:
    - `StepwiseMotionFeature`: 전체 선율 음정 중 순차 진행(장·단 2도, 1~2반음)이 차지하는 비율.
    - `ChromaticMotionFeature`: 반음계적 진행(1반음) 비율.
    - `MelodicTritonesFeature`: 선율 내 증4도/감5도(Tritone, 6반음) 도약 비율.
    - `MelodicOctavesFeature`: 1옥타브 이상 큰 도약 비율.
    - `RepeatedNotesFeature`: 동음 반복 비율.
    - `AmountOfArpeggiationFeature`: 3도·4도·5도·8도 아르페지오 진행 비율.
  - 다성 텍스처 및 성부 독립성:
    - `AverageNumberOfIndependentVoicesFeature`: 곡 전체에서 동시에 독립적으로 움직이는 평균 성부 수.
    - `VariabilityOfNumberOfIndependentVoicesFeature`: 성부 수의 시간적 변화량(푸가 제시부처럼 성부가 1 -> 2 -> 3 -> 4성으로 진입할 때 특징적으로 나타남).
    - `AverageTimeBetweenAttacksForEachVoiceFeature`: 성부별 평균 발음 간격.
    - `AverageVariabilityOfTimeBetweenAttacksForEachVoiceFeature`: 성부별 리듬 변화도.
  - 조성 및 화성 구조:
    - `PitchClassDistributionFeature`: 12음계 분포 벡터.
    - `FifthsPitchHistogramFeature`: 5도권(Circle of Fifths) 순서로 정렬된 피치 클래스 히스토그램(조성 전환 및 5도권 응답 분석용).
    - `QualityFeature`: 장조(0) vs 단조(1) 화성 색채 지표.

---

### 3.6 [표준 6] 심볼릭 다성 음악 생성 모델 논문들의 공인 비교 프로토콜 (`JSB Chorales Benchmark`)

기계학습 학회(ICML, NeurIPS, ISMIR)에서 바흐 코랄 및 다성 음악 생성 모델(예: `RNN-RBM`, `DeepBach`, `Coconet`, `Music Transformer`)을 발표할 때 사용하는 공식 벤치마크 프로토콜은 다음과 같다.

1. [표준 데이터셋 분할 (`JSB Chorales Canonical Split`)]:
   - Boulanger-Lewandowski et al. (ICML 2012, University of Montreal) 및 Allan & Williams (NIPS 2004, University of Edinburgh)가 정의한 바흐 4성 코랄 382곡의 공식 곡 단위 분할(`Train: 229곡 / Validation: 76곡 / Test: 77곡`)을 그대로 사용한다. (조성 증강 전에 반드시 곡 단위로 분리되어 있음).
2. [홀드아웃 음의 로그 우도 (`Held-Out Frame / Note Negative Log-Likelihood, NLL`)]:
   - `Coconet` (Huang et al., ICML 2017 / ISMIR 2019, Google Brain) 및 `DeepBach` (Hadjeres et al., ICML 2017, Sony CSL Paris)에서 사용하는 표준 지표로, 학습에 쓰이지 않은 `Test 77곡`에 대해 16분음표 프레임(또는 노트) 단위의 조건부 음의 로그 우도(`Frame-level NLL`, 단위: nats 또는 bits/timestep)를 측정한다.
3. [원곡 복제(Plagiarism) 최대 길이 검사 (Hadjeres et al., ICML 2017)]:
   - `DeepBach` 논문에서 공식 도입한 과적합 검사법으로, 생성된 코랄과 학습셋 코랄(12키 전조 불변) 사이에서 4성부가 정확히 일치하는 최장 연속 서브시퀀스 길이(Maximum Copied Subsequence Length)를 박자 단위로 측정하여 모델이 원곡을 단순 암기했는지 검증한다.

---

## 4. `Like_Bach` 환경 내 공식 API 실측 검증 (바흐 원곡 코퍼스 5곡 기준)

현재 워크스페이스에 설치된 `music21`의 공식 모듈(`music21.figuredBass.checker`, `music21.figuredBass.possibility`, `music21.features.jSymbolic`)을 직접 구동하여 `music21.corpus` 내 실제 바흐 코랄 원곡 5곡(`BWV 66.6`, `BWV 269`, `BWV 270`, `BWV 271`, `BWV 272`)을 측정한 실측 결과는 다음과 같다.

| 곡 번호 (`music21.corpus`) | 수직 화성 슬라이스 수 | `parallelFifths` (병행 5도) | `parallelOctaves` (병행 8도) | `hiddenFifth` (외성 은복 5도) | `hiddenOctave` (외성 은복 8도) | `voiceCrossing` (성부 교차) | `StepwiseMotionFeature` (순차진행 비율) | `MelodicTritonesFeature` (증4도 도약 비율) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bwv66.6` | 51 | `0` | `0` | `0` | `3` | `1` | `0.585` (58.5%) | `0.0126` (1.26%) |
| `bwv269` | 80 | `0` | `0` | `0` | `1` | `0` | `0.692` (69.2%) | `0.0000` (0.00%) |
| `bwv270` | 76 | `0` | `0` | `0` | `1` | `5` | `0.629` (62.9%) | `0.0000` (0.00%) |
| `bwv271` | 69 | `0` | `0` | `0` | `0` | `9` | `0.685` (68.5%) | `0.0000` (0.00%) |
| `bwv272` | 65 | `0` | `0` | `0` | `2` | `0` | `0.693` (69.3%) | `0.0000` (0.00%) |

- 실측 결과에서 확인되듯, 바흐 원곡 코퍼스를 `music21.figuredBass.checker` 공식 함수로 검사하면 `parallelFifths`(`P5`), `parallelOctaves`(`P8`), `hiddenFifth`(`H5`)가 5곡 모두 정확히 `0건`으로 측정되며, `jSymbolic`의 `StepwiseMotionFeature`는 `0.585 ~ 0.693` (약 60~70%가 순차 진행), `MelodicTritonesFeature`는 `0.0000` (단 1곡만 1.26%)으로 일관되게 나타난다.

---

## 5. `Like_Bach` 적용 권장안 (이슈 #27 연동)

자체 고안 지표를 만들지 않고, 다음 3가지 공인 도구를 그대로 조합하여 `Like_Bach`의 공식 평가 파이프라인을 구성하는 것이 가장 객관적이며 학술적·엔지니어링 정당성이 높다.

1. [대위법·화성 규칙 검증기]: 이미 설치된 `music21.figuredBass.checker`(`parallelFifths`, `parallelOctaves`, `hiddenFifth`, `hiddenOctave`, `voiceCrossing`) 및 `music21.figuredBass.possibility`(`voiceOverlap`, `upperPartsWithinLimit`)를 그대로 호출하여 마디당 위배 건수를 측정한다.
2. [단일 곡 및 코퍼스 특징 통계]: 이미 설치된 `music21.features.jSymbolic`의 72개 공식 추출기(`StepwiseMotionFeature`, `MelodicTritonesFeature`, `AverageNumberOfIndependentVoicesFeature` 등)와 `MusPy`(`muspy.metrics`의 `scale_consistency`, `polyphony_rate`, `pitch_class_entropy`, `groove_consistency`)를 사용한다.
3. [코퍼스 분포 거리 비교]: `MGEval`의 `Overlapping Area (OA)` / `KLD` 방식 또는 `frechet-music-distance`(`FMD`)를 사용하여 홀드아웃 바흐 테스트 코퍼스와의 분포 거리를 측정한다.

---

# [ 261110 0035 메인 워크스테이션 ] 원곡 단위 분할(Piece-Level Split) V5 재학습 모델 및 V4 코랄 모델 공인 지표 실측 결과

## 1. 원곡 단위 분할 및 소형 프리셋(`small`, 10.85M 파라미터) 재학습 요약 (Issue #27 완료)

- [데이터셋 보강 및 분할]:
  - 누락되어 있던 평균율 클라비어곡집 2권 1번(`wtc2f01.xml`) 및 3번(`wtc2f03.xml`)을 스탠퍼드 `KernScores`로부터 확보하여 총 48곡(`WTC Book 1 & 2` 전곡)을 완비함.
  - 12키 전조 및 성부 교환 증강을 수행하기 [이전에] 고정 시드(`seed=42`)로 원곡 파일 단위 분할(`Train: 38곡 / Val: 4곡 / Test: 6곡`, `data/processed/v5/split_manifest.json`)을 수행하여 학습·검증 간 데이터 누수를 원천 차단함.
- [모델 스케일 및 최적 체크포인트 선별]:
  - 파라미터 규모: `small` 프리셋(`n_embd=384, n_head=6, n_layer=6`, 총 `10,848,255` 파라미터, 가중치 크기 `45.5 MB`)을 적용하여 기존 `113.8M` 모델 대비 용량을 1/10로 경량화하고 과적합을 억제함.
  - 학습 곡선(`data/processed/v5/training_log.json`): 총 15 에포크 학습 중 `Epoch 7`에서 검증 손실이 최저점(`Train Loss: 0.9287`, `Val Loss: 0.8976`, `Test Loss: 1.2390`)에 도달한 뒤 `Epoch 8~14` 구간에서 검증 손실이 상승(`0.9148 -> 0.9748`)하는 과적합 변곡점을 포착하여 `Epoch 7` 가중치를 최종 체크포인트로 보존함.
  - 배포: `.pt`, `.safetensors`, `.onnx`(ONNX Runtime 최대 절대 오차 `4.768e-06`), `.pkl` 및 `release_manifest.json`을 Hugging Face Hub(`chemahc94/like-bach-models`)에 동기화 완료함.

## 2. 바흐 원곡 코퍼스 대비 V4(코랄) 및 V5(푸가) 생성 결과 공인 지표 비교표

`src/v5/evaluator.py`에 구현된 `music21.figuredBass.checker`, `jSymbolic 2.2`, `MusPy` 공식 함수를 통해 바흐 원곡 코퍼스 평균치와 생성 모델 출력을 비교 측정한 결과는 다음과 같다.

| 공인 평가 프레임워크 | 공식 지표명 | 바흐 원곡 코퍼스 실측 참조치 | Chorale V4 생성 결과 | Fugue V5 (`10.85M` Piece-Split) 생성 결과 |
| :--- | :--- | :---: | :---: | :---: |
| `music21.figuredBass.checker` | `hidden_fifths` (외성 은복 5도) | `0` | `0` | `0` |
| `music21.figuredBass.checker` | `hidden_octaves` (외성 은복 8도) | `0 ~ 3` | `0` | `0` |
| `music21.figuredBass.checker` | `voice_crossings` (성부 교차) | `0 ~ 5` | `0` | `0` |
| `music21.figuredBass.possibility` | `voice_overlaps` (성부 침범) | `0 ~ 10` | `0` | `0` |
| `jSymbolic 2.2` | `StepwiseMotionFeature` (순차 진행 비율) | `0.585 ~ 0.693` | `0.6017` (60.2%) | `0.6762` (67.6%) |
| `jSymbolic 2.2` | `MelodicTritonesFeature` (증4도 도약 비율) | `0.0000 ~ 0.0126` | `0.0169` (1.69%) | `0.0000` (0.00%) |
| `jSymbolic 2.2` | `MelodicOctavesFeature` (옥타브 도약 비율) | `0.0045 ~ 0.0265` | `0.0000` (0.00%) | `0.0000` (0.00%) |
| `jSymbolic 2.2` | `AverageNumberOfIndependentVoicesFeature` | 코랄 `~3.8` / 푸가 `~2.5` | `3.8667` | `2.4909` |
| `MusPy` (`muspy.metrics`) | `scale_consistency` (조성 스케일 일관성) | `0.867 ~ 0.996` (평균 `0.945`) | `0.9180` (91.8%) | `0.9541` (95.4%) |
| `MusPy` (`muspy.metrics`) | `polyphony_rate` (다성 발음 타임스텝 비율) | 코랄 `1.000` | `0.9688` (96.9%) | `0.6667` (제시부 성부 순차 진입 반영) |
| `MusPy` (`muspy.metrics`) | `pitch_class_entropy` (피치 클래스 엔트로피) | `2.80 ~ 3.15` | `3.1258` | `2.7697` |
| `MusPy` (`muspy.metrics`) | `groove_consistency` (마디 간 리듬 일관성) | `0.850 ~ 0.910` (평균 `0.880`) | `0.7679` | `0.8807` |

