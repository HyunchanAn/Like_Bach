import os
import pickle
from music21 import converter, interval
import glob

def extract_fugue_voices_to_bars(score):
    """
    MusicXML/MIDI 점수를 마디(Bar) 및 성부(Voice) 단위로 재편성합니다.
    동적 패딩(Dynamic Padding) 지원: 2~5성부에 유연하게 대응합니다.
    """
    parts = score.parts
    if not parts:
        parts = [score]
    num_voices = min(len(parts), 5) # 최대 5성부 지원
    
    bars_dict = {}
    
    for v_idx in range(num_voices):
        v_num = v_idx + 1
        part = parts[v_idx]
        
        if not part.getElementsByClass('Measure'):
            part = part.makeMeasures()
            
        for m in part.getElementsByClass('Measure'):
            bar_num = m.number
            if bar_num not in bars_dict:
                bars_dict[bar_num] = {}
            if v_num not in bars_dict[bar_num]:
                bars_dict[bar_num][v_num] = []
                
            for elem in m.flatten().notesAndRests:
                off_in_bar = round(float(elem.offset), 3)
                dur = round(float(elem.quarterLength), 3)
                if elem.isNote:
                    bars_dict[bar_num][v_num].append((off_in_bar, int(elem.pitch.midi), dur, "NOTE"))
                elif elem.isChord:
                    highest = max(elem.pitches).midi
                    bars_dict[bar_num][v_num].append((off_in_bar, highest, dur, "NOTE"))
                elif elem.isRest:
                    bars_dict[bar_num][v_num].append((off_in_bar, 0, dur, "REST"))

    return bars_dict, num_voices

def get_voice_tessitura_limit(v_num, total_voices=4):
    """
    성부별 전통적 음역대. 동적 성부에 맞춰 조절 가능.
    """
    if total_voices <= 2:
        return {1: (55, 84), 2: (36, 65)}.get(v_num, (0, 127))
    elif total_voices == 3:
        return {1: (60, 84), 2: (48, 72), 3: (36, 60)}.get(v_num, (0, 127))
    else:
        return {
            1: (60, 84), # Soprano
            2: (55, 72), # Alto
            3: (48, 67), # Tenor
            4: (36, 60), # Bass
            5: (36, 60)  # Bass 2
        }.get(v_num, (0, 127))

def is_valid_tessitura(bars_dict, num_voices):
    tolerance = 4
    for bar_num, voices in bars_dict.items():
        for v_num, elements in voices.items():
            min_p, max_p = get_voice_tessitura_limit(v_num, num_voices)
            for off, pitch, dur, type_ in elements:
                if type_ == "NOTE":
                    if pitch < min_p - tolerance or pitch > max_p + tolerance:
                        return False
    return True

def swap_voices(bars_dict, vA, vB):
    import copy
    new_bars = copy.deepcopy(bars_dict)
    for bar_num in new_bars:
        temp = new_bars[bar_num].get(vA, [])
        new_bars[bar_num][vA] = new_bars[bar_num].get(vB, [])
        new_bars[bar_num][vB] = temp
    return new_bars

def inject_structural_tags(bars_dict, num_voices):
    import copy
    tagged_bars = copy.deepcopy(bars_dict)
    
    voice_entry = {}
    for bar_num in sorted(tagged_bars.keys()):
        for v_num, elements in tagged_bars[bar_num].items():
            has_notes = any(e[3] == "NOTE" for e in elements)
            if has_notes and v_num not in voice_entry:
                voice_entry[v_num] = bar_num
                
    sorted_voices = sorted(voice_entry.items(), key=lambda x: x[1])
    if not sorted_voices:
        return tagged_bars
        
    entry_order = [v for v, bar in sorted_voices]
    
    for i, v_num in enumerate(entry_order):
        start_bar = voice_entry[v_num]
        is_subject = (i % 2 == 0)
        
        if start_bar in tagged_bars and v_num in tagged_bars[start_bar]:
            start_tag = "SUBJECT_START" if is_subject else "ANSWER_START"
            if len(tagged_bars[start_bar][v_num]) > 0:
                first_elem = tagged_bars[start_bar][v_num][0]
                tagged_bars[start_bar][v_num].insert(0, (first_elem[0], 0, 0, start_tag))
                
            for prev_v in entry_order[:i]:
                if start_bar in tagged_bars and prev_v in tagged_bars[start_bar]:
                    if len(tagged_bars[start_bar][prev_v]) > 0:
                        first_elem = tagged_bars[start_bar][prev_v][0]
                        tagged_bars[start_bar][prev_v].insert(0, (first_elem[0], 0, 0, "COUNTERSUBJECT_START"))
                    
    return tagged_bars

def tokenize_fugue_piece(file_path):
    try:
        score = converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

    all_sequences = []
    
    # Data Augmentation: 12 Key Transposition
    for i in range(-6, 6):
        try:
            transposed_score = score.transpose(interval.Interval(i))
            bars_dict, num_voices = extract_fugue_voices_to_bars(transposed_score)
            
            bars_dict = inject_structural_tags(bars_dict, num_voices)
            
            augmentations = [bars_dict]
            
            # Voice Permutation (Soprano <-> Alto)
            if num_voices >= 2:
                swapped = swap_voices(bars_dict, 1, 2)
                augmentations.append(swapped)
                
            for aug_bars in augmentations:
                if not is_valid_tessitura(aug_bars, num_voices):
                    continue
                
                seq = []
                sorted_bars = sorted(aug_bars.keys())
                for bar_num in sorted_bars:
                    seq.append(f"[BAR_{bar_num}]")
                    voices = aug_bars[bar_num]
                    # 동적 패딩: 파일에 존재하는 num_voices 기준으로 모두 순회
                    for v_num in range(1, num_voices + 1):
                        seq.append(f"[VOICE_{v_num}]")
                        elements = sorted(voices.get(v_num, []), key=lambda x: x[0])
                        if not elements:
                            seq.extend(["[REST]", "D4.0"])
                            continue
                            
                        for off, pitch, dur, type_ in elements:
                            if type_ == "REST":
                                seq.extend(["[REST]", f"D{dur}"])
                            elif type_ in ["SUBJECT_START", "ANSWER_START", "COUNTERSUBJECT_START"]:
                                seq.append(f"[{type_}]")
                            else:
                                # [PITCH]와 [DURATION] 분리
                                seq.extend([f"P{pitch}", f"D{dur}"])
                
                if seq:
                    seq.append("[FINAL]")
                    all_sequences.append(seq)
        except Exception as e:
            continue
            
    return all_sequences

class FugueTokenizerV5:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        
    def build_vocab(self, sequences):
        vocab = set(["[PAD]", "[UNK]"])
        for seq in sequences:
            for token in seq:
                vocab.add(token)
                
        vocab.update(["[SUBJECT_START]", "[SUBJECT_END]", "[ANSWER_START]", "[ANSWER_END]", "[COUNTERSUBJECT_START]", "[COUNTERSUBJECT_END]"])
        
        self.itos = {i: t for i, t in enumerate(sorted(list(vocab)))}
        self.stoi = {t: i for i, t in self.itos.items()}
        self.vocab_size = len(self.itos)
        
    def encode(self, sequence):
        return [self.stoi.get(t, self.stoi["[UNK]"]) for t in sequence]
        
    def decode(self, indices):
        return [self.itos.get(i, "[UNK]") for i in indices]
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'stoi': self.stoi, 'itos': self.itos}, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.stoi = data['stoi']
            self.itos = data['itos']
            self.vocab_size = len(self.itos)

if __name__ == "__main__":
    os.makedirs('data/processed/v5', exist_ok=True)
    
    xml_files = glob.glob('data/raw/wtc/*.xml') + glob.glob('data/raw/# Fugue/*.xml')
    print(f"Found {len(xml_files)} files.")
    
    all_seqs = []
    for f in xml_files:
        print(f"Processing {os.path.basename(f)}...")
        seqs = tokenize_fugue_piece(f)
        all_seqs.extend(seqs)
        
    print(f"Generated {len(all_seqs)} transposed sequences.")
    
    tokenizer = FugueTokenizerV5()
    tokenizer.build_vocab(all_seqs)
    print(f"Vocab size: {tokenizer.vocab_size}")
    tokenizer.save('data/processed/v5/fugue_vocab_v5.pkl')
    
    with open('data/processed/v5/fugue_dataset_v5.pkl', 'wb') as f:
        pickle.dump(all_seqs, f)
    print("Saved dataset to data/processed/v5/fugue_dataset_v5.pkl")
