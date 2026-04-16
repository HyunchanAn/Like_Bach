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
        
        # 성부 추출 (4성부 코랄 최적화)
        parts = transposed_score.parts
        if not parts:
            parts = [transposed_score]
            
        # 4성부 마스터 인터리빙 (최대 4성부까지 모두 추출)
        num_voices = min(len(parts), 4)
        seq = [key_token]
        time_map = {}
        
        for v_idx in range(num_voices):
            voice_num = v_idx + 1 # [V1]~[V4]
            for n in parts[v_idx].recurse().notes:
                # 시간 정규화 (Quantization): 소수점 3자리 반올림으로 데이터 노이즈 제거
                off = round(float(n.offset), 3)
                if off not in time_map: time_map[off] = {}
                time_map[off][voice_num] = n
        
        # 시간 순서대로 4성부 인터리빙 토큰 생성
        for off in sorted(time_map.keys()):
            seq.append(f"[TIME_{off}]")
            entries = time_map[off]
            # 각 시점마다 존재하는 성부들을 순서대로 배치
            for v in range(1, 5):
                if v in entries:
                    n = entries[v]
                    pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                    dur = round(float(n.duration.quarterLength), 3)
                    seq.append(f"[V{v}] P{pitch} D{dur}")
        
        if len(seq) > 10: # 유의미한 길이
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
    processed_file = 'data/processed/v4/bach_tokens.pkl'
    
    preprocess_all(raw_dirs, processed_file)
