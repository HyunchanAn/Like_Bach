import os
import pickle
from music21 import converter, note, chord, stream, interval
from tqdm import tqdm

import itertools

def tokenize_piece(file_path):
    """
    MusicXML ?뚯씪??議곗꽦 湲곕컲 ?쒓컙 援먯감??Time-Interleaved) ?좏겙?쇰줈 蹂?섑빀?덈떎.
    ?뺤떇: [KEY_X] [TIME_O] [V1] P_ D_ [V2] P_ D_ ...
    """
    try:
        score = converter.parse(file_path)
    except Exception as e:
        return None

    # 1. 議곗꽦 遺꾩꽍
    try:
        source_key = score.analyze('key')
    except:
        source_key = music21.key.Key('C') # Fallback

    all_sequences = []
    
    # 2. ?곗씠??利앷컯 (12媛?議?
    for i in range(-6, 6):
        transposed_score = score.transpose(interval.Interval(i))
        current_key = source_key.transpose(interval.Interval(i))
        key_token = f"[KEY_{current_key.tonic.name}{'m' if current_key.mode == 'minor' else ''}]"
        
        # ?깅? 異붿텧 (理쒕? 4?깅? 媛??
        parts = transposed_score.parts
        if not parts:
            parts = [transposed_score]
            
        # 4?깅? ?댁긽??寃쎌슦, ?ъ슜???쇰뱶諛?諛섏쁺?섏뿬 ?깅? ??Pair) 異붿텧
        if len(parts) >= 2:
            # 媛?ν븳 紐⑤뱺 2?깅? 議고빀 (SA, SB, TB ??
            combos = list(itertools.combinations(range(len(parts)), 2))
            # ?덈Т 留롮쑝硫??쇰?留??좏깮 (理쒕? 3媛??? 諛붽묑 ?깅? 諛?二쇱슂 ?깅?)
            if len(combos) > 3:
                combos = [combos[0], combos[-1], (0, len(parts)-1)] # Sop-Alt, Ten-Bas, Sop-Bas ??(?몃뜳??異붿젙)
            
            for p1_idx, p2_idx in combos:
                seq = [key_token]
                # ?쒓컙???뺣젹???꾪븳 ?뺤뀛?덈━ {offset: {v1: note, v2: note}}
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
                
                # ?쒓컙 ?쒖꽌?濡??좏겙 ?앹꽦
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
                
                if len(seq) > 5: # ?좎쓽誘명븳 湲몄씠
                    all_sequences.append(seq)
        else:
            # ?⑥꽑?⑥쓽 寃쎌슦 (?숈뒿 ?⑥쑉? ??쑝???ы븿)
            seq = [key_token]
            for n in parts[0].recurse().notes:
                pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                seq.append(f"[TIME_{float(n.offset)}] [V1] P{pitch} D{float(n.duration.quarterLength)}")
            if len(seq) > 5:
                all_sequences.append(seq)
    
    return all_sequences

def preprocess_all(input_dirs, output_file):
    """
    ?섏쭛??諛뷀쓲 ?묓뭹?ㅼ쓣 議곗꽦 湲곕컲 ?쒓컙 援먯감???좏겙?쇰줈 蹂?섑븯????ν빀?덈떎.
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
            # 媛쒕퀎 怨??꾩쿂由?(?곗씠??利앷컯 ?ы븿)
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
    # ?좉퇋 ?곗씠??寃쎈줈 諛섏쁺
    raw_dirs = ['data/raw/bach'] 
    processed_file = 'data/processed/v3/bach_tokens.pkl'
    
    preprocess_all(raw_dirs, processed_file)
