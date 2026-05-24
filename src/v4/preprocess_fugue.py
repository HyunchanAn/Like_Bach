import os
import pickle
from music21 import converter, note, chord, stream, interval, expressions
from tqdm import tqdm

def tokenize_fugue(file_path):
    """
    MusicXML 파일을 푸가/인벤션 특화 토큰으로 변환.
    형식: [KEY_X] [TS_X] [SUBJECT] [TIME_O] [V1] P_ D_ ... [ANSWER] ... [EPISODE] ... [FINAL]
    """
    try:
        score = converter.parse(file_path)
    except Exception:
        return None

    try:
        source_key = score.analyze('key')
    except:
        return None
    
    ts_list = score.recurse().getElementsByClass('TimeSignature')
    ts_token = f"[TS_{ts_list[0].ratioString}]" if ts_list else "[TS_4/4]"

    all_sequences = []
    
    # 12 keys augmentation
    for i in range(-6, 6):
        transposed_score = score.transpose(interval.Interval(i))
        current_key = source_key.transpose(interval.Interval(i))
        key_token = f"[KEY_{current_key.tonic.name}{'m' if current_key.mode == 'minor' else ''}]"
        
        parts = transposed_score.parts
        if not parts: parts = [transposed_score]
        num_voices = min(len(parts), 4)
        
        time_map = {}
        part_starts = []
        
        for v_idx in range(num_voices):
            v_num = v_idx + 1
            first_offset = -1
            for n in transposed_score.parts[v_idx].flatten().notes:
                off = round(float(n.offset), 3)
                if first_offset == -1: first_offset = off
                if off not in time_map: time_map[off] = {}
                time_map[off][v_num] = n
            
            if first_offset != -1:
                part_starts.append((v_num, first_offset))
                
        if not time_map: continue
        
        # Sort voices by entry time
        part_starts.sort(key=lambda x: x[1])
        if len(part_starts) == 0: continue
        
        subject_time = part_starts[0][1]
        answer_time = part_starts[1][1] if len(part_starts) > 1 else subject_time + 8.0
        
        # Approximate episode start as answer_time + (answer_time - subject_time)
        subject_len = max(4.0, answer_time - subject_time)
        episode_time = answer_time + subject_len
        
        sorted_offsets = sorted(time_map.keys())
        total_length = max(sorted_offsets)
        total_measures = int(total_length // 4) + 1
        
        seq = [key_token, ts_token]
        last_measure_idx = -1
        
        # Track structural state
        current_struct = None
        
        for off in sorted_offsets:
            # Structural tagging
            new_struct = None
            if off >= episode_time:
                new_struct = "[EPISODE]"
            elif off >= answer_time:
                new_struct = "[ANSWER]"
            elif off >= subject_time:
                new_struct = "[SUBJECT]"
                
            if new_struct and current_struct != new_struct:
                seq.append(new_struct)
                current_struct = new_struct

            # Measure countdown
            current_measure_idx = int(off // 4)
            if current_measure_idx > last_measure_idx:
                remain = max(0, total_measures - current_measure_idx)
                seq.append(f"[REMAIN_{remain}]")
                last_measure_idx = current_measure_idx

            # Time and Notes
            seq.append(f"[TIME_{off}]")
            entries = time_map[off]
            for v in range(1, 5):
                if v in entries:
                    n = entries[v]
                    pitch = n.pitch.midi if hasattr(n, 'pitch') else n.pitches[0].midi
                    dur = round(float(n.duration.quarterLength), 3)
                    seq.append(f"[V{v}] P{pitch} D{dur}")
        
        seq.append("[FINAL]")
        seq.append("[EOS]")
        
        if len(seq) > 20:
            all_sequences.append(seq)
    
    return all_sequences

def preprocess_fugues(input_dirs, output_file):
    all_tokens_dataset = []
    
    files_to_process = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist.")
            continue
        for f in os.listdir(input_dir):
            if f.endswith('.xml') or f.endswith('.mxl') or f.endswith('.krn'):
                files_to_process.append(os.path.join(input_dir, f))
                
    print(f"Found {len(files_to_process)} pieces. Starting Fugue Structural Preprocessing...")
    
    for file_path in tqdm(files_to_process):
        try:
            sequences = tokenize_fugue(file_path)
            if sequences:
                all_tokens_dataset.extend(sequences)
        except Exception:
            continue
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_tokens_dataset, f)
    
    print(f"\nSuccess: Total {len(all_tokens_dataset)} Fugue sequences saved to {output_file}")

if __name__ == "__main__":
    dirs = ['data/raw/# Fugue', 'data/raw/# Fugue_Corpus', 'data/raw/inventions', 'data/raw/wtc']
    preprocess_fugues(dirs, 'data/processed/v4/fugue_tokens.pkl')
