import os
import pickle
from music21 import converter, note, chord, stream, interval
from tqdm import tqdm

import itertools

def tokenize_piece(file_path):
    """
    MusicXML 파일을 조성 기반 시간 교차형(Time-Interleaved) 토큰으로 변환합니다.
    형식: [KEY_X] [TIME_O] [V1] P_ D_ [V2] P_ D_ ...
    """
    try:
        score = converter.parse(file_path)
    except Exception as e:
        return None

    # 1. 조성 분석
    try:
        source_key = score.analyze('key')
    except:
        source_key = music21.key.Key('C') # Fallback

    all_sequences = []
    
    # 2. 데이터 증강 (12개 조)
    for i in range(-6, 6):
        transposed_score = score.transpose(interval.Interval(i))
        current_key = source_key.transpose(interval.Interval(i))
        key_token = f"[KEY_{current_key.tonic.name}{'m' if current_key.mode == 'minor' else ''}]"
        
        # 성부 추출 (최대 4성부 가정)
        parts = transposed_score.parts
        if not parts:
            parts = [transposed_score]
            
        # 4성부 이상의 경우, 사용자 피드백 반영하여 성부 쌍(Pair) 추출
        if len(parts) >= 2:
            # 가능한 모든 2성부 조합 (SA, SB, TB 등)
            combos = list(itertools.combinations(range(len(parts)), 2))
            # 너무 많으면 일부만 선택 (최대 3개 쌍: 바깥 성부 및 주요 성부)
            if len(combos) > 3:
                combos = [combos[0], combos[-1], (0, len(parts)-1)] # Sop-Alt, Ten-Bas, Sop-Bas 등 (인덱스 추정)
            
            for p1_idx, p2_idx in combos:
                seq = [key_token]
                # 시간순 정렬을 위한 딕셔너리 {offset: {v1: note, v2: note}}
                time_map = {}
                
                # Part 1
                for n in parts[p1_idx].recurse().notes:
                    off = float(n.offset)
                    if off not in time_map: time_map[off] = {}
                    time_map[off][1] = n
                # Part 2
                for n in parts[p2_idx].recurse().notes:
                    off = float(n.offset)
                    if off not in time_map: time_map[off] = {}
                    time_map[off][2] = n
                
                # 시간 순서대로 토큰 생성
                for off in sorted(time_map.keys()):
                    seq.append(f"[TIME_{off}]")
                    entries = time_map[off]
                    if 1 in entries:
                        n = entries[1]
                        pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                        seq.append(f"[V1] P{pitch} D{float(n.duration.quarterLength)}")
                    if 2 in entries:
                        n = entries[2]
                        pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                        seq.append(f"[V2] P{pitch} D{float(n.duration.quarterLength)}")
                
                if len(seq) > 5: # 유의미한 길이
                    all_sequences.append(seq)
        else:
            # 단선율의 경우 (학습 효율은 낮으나 포함)
            seq = [key_token]
            for n in parts[0].recurse().notes:
                pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                seq.append(f"[TIME_{float(n.offset)}] [V1] P{pitch} D{float(n.duration.quarterLength)}")
            if len(seq) > 5:
                all_sequences.append(seq)
    
    return all_sequences

def preprocess_all(input_dirs, output_file):
    """
    수집된 바흐 작품들을 조성 기반 시간 교차형 토큰으로 변환하여 저장합니다.
    """
    all_tokens_dataset = []
    
    files_to_process = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            continue
        for filename in os.listdir(input_dir):
            if filename.endswith('.xml') or filename.endswith('.mxl'):
                files_to_process.append(os.path.join(input_dir, filename))
                
    print(f"Found {len(files_to_process)} pieces. Starting Harmonic Preprocessing (V3.1)...")
    
    for file_path in tqdm(files_to_process):
        try:
            # 개별 곡 전처리 (데이터 증강 포함)
            augmented_sequences = tokenize_piece(file_path)
            if augmented_sequences:
                all_tokens_dataset.extend(augmented_sequences)
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            continue
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_tokens_dataset, f)
    
    print(f"\nSuccess: Total {len(all_tokens_dataset)} sequences (Time-Interleaved) saved to {output_file}")

if __name__ == "__main__":
    # 신규 데이터 경로 반영
    raw_dirs = ['data/raw/bach'] 
    processed_file = 'data/processed/bach_tokens.pkl'
    
    preprocess_all(raw_dirs, processed_file)
