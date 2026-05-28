def detect_parallel_intervals(voice_notes):
    """
    4성부(Soprano, Alto, Tenor, Bass) 노트 리스트를 받아 병행 5도/8도 등 대위법 오류를 감지합니다.
    voice_notes: dict of voice_id -> list of (pitch, start_time, duration)
    """
    errors = []
    
    voices = list(voice_notes.keys())
    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            v1, v2 = voices[i], voices[j]
            notes1 = voice_notes[v1]
            notes2 = voice_notes[v2]
            
            # Simple heuristic matching intervals at simultaneous note changes
            # In a full implementation, we'd align the offsets and check sequential harmonic intervals.
            pass
            
    return errors

def evaluate_measure(notes_in_measure):
    """
    마디 단위의 노트를 입력받아 대위법적 금기 사항(병행 5/8도, 은복 등)을 검사합니다.
    반환값이 빈 리스트면 정상, 오류가 있으면 부분 재작곡(In-painting) 대상이 됩니다.
    """
    errors = []
    # TODO: Implement full harmonic analysis
    return errors
