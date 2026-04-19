import os
import pickle
from music21 import converter, note, chord, stream, interval
from tqdm import tqdm

import itertools

from music21 import converter, note, chord, stream, interval, roman, key, expressions
from tqdm import tqdm

import itertools

def tokenize_piece(file_path):
    """
    MusicXML 파일을 고도화된 시간 교차형(Time-Interleaved) 토큰으로 변환합니다.
    형식: [KEY_X] [TS_X] [REMAIN_N] [ROMAN_X] [TIME_O] [V1] P_ D_ ... [FERM] [FINAL]
    """
    try:
        score = converter.parse(file_path)
    except Exception as e:
        return None

    # 1. 조성 및 박자 분석
    try:
        source_key = score.analyze('key')
    except:
        source_key = key.Key('C')
    
    ts_list = score.recurse().getElementsByClass('TimeSignature')
    ts_token = f"[TS_{ts_list[0].ratioString}]" if ts_list else "[TS_4/4]"

    all_sequences = []
    
    # 2. 데이터 증강 (12개 조)
    for i in range(-6, 6):
        transposed_score = score.transpose(interval.Interval(i))
        current_key = source_key.transpose(interval.Interval(i))
        key_token = f"[KEY_{current_key.tonic.name}{'m' if current_key.mode == 'minor' else ''}]"
        
        parts = transposed_score.parts
        if not parts: parts = [transposed_score]
        num_voices = min(len(parts), 4)
        
        # 성부 데이터 수집
        time_map = {} # offset -> {voice_idx: note_obj}
        fermata_map = {} # offset -> bool
        
        for v_idx in range(num_voices):
            v_num = v_idx + 1
            for n in transposed_score.parts[v_idx].recurse().notes:
                off = round(float(n.offset), 3)
                if off not in time_map: time_map[off] = {}
                time_map[off][v_num] = n
                # 페르마타 확인
                if any(isinstance(expr, expressions.Fermata) for expr in n.expressions):
                    fermata_map[off] = True

        # 마디 정보 및 전체 길이 계산
        # 4/4 박자 기준 마디 수 추정 (더 정밀한 계산은 music21.measure.Measure 사용)
        sorted_offsets = sorted(time_map.keys())
        if not sorted_offsets: continue
        total_length = max(sorted_offsets)
        total_measures = int(total_length // 4) + 1
        
        seq = [key_token, ts_token]
        last_measure_idx = -1
        
        for off in sorted_offsets:
            # 3. 마디 카운트다운 (새로운 마디 시작 시 주입)
            current_measure_idx = int(off // 4)
            if current_measure_idx > last_measure_idx:
                remain = max(0, total_measures - current_measure_idx)
                seq.append(f"[REMAIN_{remain}]")
                
                # 4. 화성 분석 (로마자 화성 기호 추출)
                current_notes = []
                for v in range(1, 5):
                    if v in time_map[off]:
                        n = time_map[off][v]
                        if n.isNote: current_notes.append(n)
                        elif n.isChord: current_notes.extend(n.pitches)
                
                if current_notes:
                    try:
                        c = chord.Chord(current_notes)
                        rn = roman.romanNumeralFromChord(c, current_key)
                        seq.append(f"[ROMAN_{rn.figure}]")
                    except:
                        pass
                
                last_measure_idx = current_measure_idx

            # 5. 시간 및 성부 토큰
            seq.append(f"[TIME_{off}]")
            if off in fermata_map:
                seq.append("[FERM]")
            
            entries = time_map[off]
            for v in range(1, 5):
                if v in entries:
                    n = entries[v]
                    pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                    dur = round(float(n.duration.quarterLength), 3)
                    seq.append(f"[V{v}] P{pitch} D{dur}")
        
        # 6. 종지 태그
        seq.append("[FINAL]")
        seq.append("[EOS]")
        
        if len(seq) > 20:
            all_sequences.append(seq)
    
    return all_sequences

def preprocess_all(input_dirs, output_file):
    """
    수집된 바흐 작품들을 고도화된 토큰으로 변환하여 저장합니다.
    """
    all_tokens_dataset = []
    
    files_to_process = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            continue
        for filename in os.listdir(input_dir):
            if filename.endswith('.xml') or filename.endswith('.mxl') or filename.endswith('.krn'):
                files_to_process.append(os.path.join(input_dir, filename))
                
    print(f"Found {len(files_to_process)} pieces. Starting Advanced Harmonic Preprocessing (V4.5)...")
    
    for file_path in tqdm(files_to_process):
        try:
            augmented_sequences = tokenize_piece(file_path)
            if augmented_sequences:
                all_tokens_dataset.extend(augmented_sequences)
        except Exception as e:
            continue
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_tokens_dataset, f)
    
    print(f"\nSuccess: Total {len(all_tokens_dataset)} sequences (Advanced) saved to {output_file}")

if __name__ == "__main__":
    # 전체 전처리 실행
    raw_dirs = ['data/raw/bach'] 
    processed_file = 'data/processed/v4/bach_tokens_advanced.pkl'
    preprocess_all(raw_dirs, processed_file)
