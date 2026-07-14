# Implementation Notes

사소한(Spec-unstated) 구현 의사결정 및 가정을 기록하는 노트입니다.

## UI / 렌더링
- **쉼표 시인성 개선**: 쉼표 자동 패딩 시 라이트 모드에서는 어두운 반투명(`rgba(30, 41, 59, 0.45)`), 다크 모드에서는 밝은 반투명(`rgba(255, 255, 255, 0.45)`)을 적용하여 가시성 확보.
- **기둥 방향(Stem Direction)**: VexFlow 옵션 무시 버그 대응을 위해 명시적으로 `.setStemDirection()`을 호출 (Soprano/Tenor 위, Alto/Bass 아래).
- **렌더링 오프셋 교정**: 캔버스 상단 마진 10px 오차를 교정하기 위해 매핑 공식을 `(y - 110)/5` 형태로 보정.

## 오디오 재생 (Tone.js)
- **동음(Same Pitch) 타이브 현상 방지**: Tone.js 로 연달아 같은 음을 재생할 때 어택(Attack)이 유실되는 현상을 막기 위해, 실제 박자 초(Sec)의 `0.85배`만 울리게 하여 아티큘레이션(Articulation) 공간을 확보함.

## AI 최적화
- **메모리 I/O**: `temp_input.mid`, `temp_output.mid` 파일 대신 `io.BytesIO`를 통한 메모리 스트리밍 파이프라인으로 전환하여 I/O 병목 및 스트리밍 유실 해소.
