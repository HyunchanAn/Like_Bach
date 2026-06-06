import sys
import os
import base64
from music21 import stream, note, tempo, meter, key, midi

sys.path.append(os.getcwd())
from src.v5.neural_engine import HybridFugueEngine

def notes_to_midi_file(notes_list, filename):
    s = stream.Score()
    s.insert(0, tempo.MetronomeMark(number=120))
    s.insert(0, meter.TimeSignature('4/4'))
    s.insert(0, key.Key('C'))
    
    parts = {1: stream.Part(), 2: stream.Part(), 3: stream.Part(), 4: stream.Part()}
    for v in parts.values():
        s.insert(0, v)
        
    for n in notes_list:
        m21_note = note.Note(n['pitch'])
        m21_note.quarterLength = n['duration']
        parts[n['voice']].insert(n['offset'], m21_note)
        
    try:
        mf = midi.translate.music21ObjectToMidiFile(s)
        mf.open(filename, 'wb')
        mf.write()
        mf.close()
        print(f"[+] MIDI 파일 생성 완료: {filename}")
        return True
    except Exception as e:
        print(f"[-] MIDI 변환 오류: {e}")
        return False

def verify_pipeline():
    print("[1] V5 Hybrid Fugue Engine 로드 중...")
    try:
        engine = HybridFugueEngine()
    except Exception as e:
        print(f"[-] 엔진 로드 실패: {e}")
        return
        
    # 간단한 1마디 주제 선율 (C 메이저: 도-레-미-파)
    subject_notes = [
        {"pitch": 60, "duration": 1.0, "offset": 0.0, "voice": 1},
        {"pitch": 62, "duration": 1.0, "offset": 1.0, "voice": 1},
        {"pitch": 64, "duration": 1.0, "offset": 2.0, "voice": 1},
        {"pitch": 65, "duration": 1.0, "offset": 3.0, "voice": 1},
    ]
    
    target_measures = 16  # 검증을 위해 16마디로 단축
    print(f"[2] 푸가 생성 시작 (목표 마디: {target_measures})...")
    try:
        final_notes = engine.generate_fugue(
            subject_notes=subject_notes,
            target_measures=target_measures,
            temperature=0.55,
            refine_iters=3,
            stream_queue=None
        )
        print(f"[+] 생성 완료! 총 {len(final_notes)}개의 노트가 생성되었습니다.")
    except Exception as e:
        print(f"[-] 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    print("[3] MIDI 파일 추출 및 검사...")
    output_file = "test_fugue_output.mid"
    success = notes_to_midi_file(final_notes, output_file)
    
    if success:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"[+] 파이프라인 정상 작동 확인! (파일 크기: {os.path.getsize(output_file)} 바이트)")
        else:
            print("[-] MIDI 파일이 비어있거나 생성되지 않았습니다.")
    
if __name__ == '__main__':
    verify_pipeline()
