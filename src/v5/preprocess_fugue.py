import copy
import glob
import json
import os
import pickle
import random

from music21 import converter, interval


def extract_fugue_voices_to_bars(score):
    """Reorganize MusicXML/MIDI score into Measure and Voice dicts (2-5 voices)."""
    parts = score.parts
    if not parts:
        parts = [score]
    num_voices = min(len(parts), 5)

    bars_dict = {}

    for v_idx in range(num_voices):
        v_num = v_idx + 1
        part = parts[v_idx]

        if not part.getElementsByClass("Measure"):
            part = part.makeMeasures()

        for m in part.getElementsByClass("Measure"):
            bar_num = m.number
            if bar_num not in bars_dict:
                bars_dict[bar_num] = {}
            if v_num not in bars_dict[bar_num]:
                bars_dict[bar_num][v_num] = []

            for elem in m.flatten().notesAndRests:
                off_in_bar = round(float(elem.offset), 3)
                dur = round(float(elem.quarterLength), 3)
                if elem.isNote:
                    bars_dict[bar_num][v_num].append(
                        (off_in_bar, int(elem.pitch.midi), dur, "NOTE")
                    )
                elif elem.isChord:
                    highest = max(elem.pitches).midi
                    bars_dict[bar_num][v_num].append(
                        (off_in_bar, highest, dur, "NOTE")
                    )
                elif elem.isRest:
                    bars_dict[bar_num][v_num].append((off_in_bar, 0, dur, "REST"))

    return bars_dict, num_voices


def get_voice_tessitura_limit(v_num, total_voices=4):
    if total_voices <= 2:
        return {1: (55, 84), 2: (36, 65)}.get(v_num, (0, 127))
    elif total_voices == 3:
        return {1: (60, 84), 2: (48, 72), 3: (36, 60)}.get(v_num, (0, 127))
    else:
        return {
            1: (60, 84),
            2: (55, 72),
            3: (48, 67),
            4: (36, 60),
            5: (36, 60),
        }.get(v_num, (0, 127))


def is_valid_tessitura(bars_dict, num_voices):
    tolerance = 4
    for _bar_num, voices in bars_dict.items():
        for v_num, elements in voices.items():
            min_p, max_p = get_voice_tessitura_limit(v_num, num_voices)
            for _off, pitch_val, _dur, type_ in elements:
                if type_ == "NOTE":
                    if pitch_val < min_p - tolerance or pitch_val > max_p + tolerance:
                        return False
    return True


def swap_voices(bars_dict, vA, vB):
    new_bars = copy.deepcopy(bars_dict)
    for bar_num in new_bars:
        temp = new_bars[bar_num].get(vA, [])
        new_bars[bar_num][vA] = new_bars[bar_num].get(vB, [])
        new_bars[bar_num][vB] = temp
    return new_bars


def inject_structural_tags(bars_dict, num_voices):
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

    entry_order = [v for v, _bar in sorted_voices]

    for i, v_num in enumerate(entry_order):
        start_bar = voice_entry[v_num]
        is_subject = i % 2 == 0

        if start_bar in tagged_bars and v_num in tagged_bars[start_bar]:
            start_tag = "SUBJECT_START" if is_subject else "ANSWER_START"
            if len(tagged_bars[start_bar][v_num]) > 0:
                first_elem = tagged_bars[start_bar][v_num][0]
                tagged_bars[start_bar][v_num].insert(
                    0, (first_elem[0], 0, 0, start_tag)
                )

            for prev_v in entry_order[:i]:
                if start_bar in tagged_bars and prev_v in tagged_bars[start_bar]:
                    if len(tagged_bars[start_bar][prev_v]) > 0:
                        first_elem = tagged_bars[start_bar][prev_v][0]
                        tagged_bars[start_bar][prev_v].insert(
                            0, (first_elem[0], 0, 0, "COUNTERSUBJECT_START")
                        )

    return tagged_bars


def tokenize_fugue_piece(file_path, augment=True):
    """Tokenize a single fugue file.

    When augment=True (training set), performs 12-key transposition and voice permutation.
    When augment=False (validation/test set), keeps only the original key (transposition=0)
    so evaluation never leaks or distorts held-out pieces.
    """
    try:
        score = converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

    all_sequences = []
    transpositions = range(-6, 6) if augment else [0]

    for semitones in transpositions:
        try:
            transposed_score = (
                score.transpose(interval.Interval(semitones))
                if semitones != 0
                else score
            )
            bars_dict, num_voices = extract_fugue_voices_to_bars(transposed_score)
            bars_dict = inject_structural_tags(bars_dict, num_voices)

            augmentations = [bars_dict]
            if augment and num_voices >= 2:
                swapped = swap_voices(bars_dict, 1, 2)
                augmentations.append(swapped)

            for aug_bars in augmentations:
                if augment and not is_valid_tessitura(aug_bars, num_voices):
                    continue

                seq = []
                sorted_bars = sorted(aug_bars.keys())
                for bar_num in sorted_bars:
                    seq.append(f"[BAR_{bar_num}]")
                    voices = aug_bars[bar_num]
                    for v_num in range(1, num_voices + 1):
                        seq.append(f"[VOICE_{v_num}]")
                        elements = sorted(voices.get(v_num, []), key=lambda x: x[0])
                        if not elements:
                            seq.extend(["[REST]", "D4.0"])
                            continue

                        for _off, pitch_val, dur, type_ in elements:
                            if type_ == "REST":
                                seq.extend(["[REST]", f"D{dur}"])
                            elif type_ in [
                                "SUBJECT_START",
                                "ANSWER_START",
                                "COUNTERSUBJECT_START",
                            ]:
                                seq.append(f"[{type_}]")
                            else:
                                seq.extend([f"P{pitch_val}", f"D{dur}"])

                if seq:
                    seq.append("[FINAL]")
                    all_sequences.append(seq)
        except Exception:
            continue

    return all_sequences


class FugueTokenizerV5:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build_vocab(self, sequences):
        vocab = set(["[PAD]", "[UNK]", "[REST]", "[FINAL]"])
        for seq in sequences:
            for token in seq:
                vocab.add(token)

        # Ensure full pitch and bar coverage so held-out pieces never hit [UNK]
        for p in range(24, 97):
            vocab.add(f"P{p}")
        for b in range(1, 150):
            vocab.add(f"[BAR_{b}]")
        for v in range(1, 6):
            vocab.add(f"[VOICE_{v}]")

        vocab.update(
            [
                "[SUBJECT_START]",
                "[SUBJECT_END]",
                "[ANSWER_START]",
                "[ANSWER_END]",
                "[COUNTERSUBJECT_START]",
                "[COUNTERSUBJECT_END]",
                "[EPISODE]",
                "[EPISODE_MODULATION]",
            ]
        )

        self.itos = {i: t for i, t in enumerate(sorted(list(vocab)))}
        self.stoi = {t: i for i, t in self.itos.items()}
        self.vocab_size = len(self.itos)

    def encode(self, sequence):
        return [self.stoi.get(t, self.stoi["[UNK]"]) for t in sequence]

    def decode(self, indices):
        return [self.itos.get(i, "[UNK]") for i in indices]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.stoi = data["stoi"]
            self.itos = data["itos"]
            self.vocab_size = len(self.itos)


if __name__ == "__main__":
    os.makedirs("data/processed/v5", exist_ok=True)

    score_files = sorted(
        glob.glob("data/raw/wtc/*.xml")
        + glob.glob("data/raw/# Fugue/*.xml")
        + glob.glob("data/raw/bach-wtc/kern/*f*.krn")
    )
    print(f"Found {len(score_files)} piece files.")

    # Piece-level split BEFORE 12-key transposition augmentation (Issue #27)
    SEED = 42
    rng = random.Random(SEED)
    shuffled_files = list(score_files)
    rng.shuffle(shuffled_files)

    n_total = len(shuffled_files)
    n_train = int(n_total * 0.8)
    n_val = max(1, int(n_total * 0.1))

    train_files = sorted(shuffled_files[:n_train])
    val_files = sorted(shuffled_files[n_train : n_train + n_val])
    test_files = sorted(shuffled_files[n_train + n_val :])

    split_manifest = {
        "seed": SEED,
        "total_pieces": n_total,
        "train_pieces_count": len(train_files),
        "val_pieces_count": len(val_files),
        "test_pieces_count": len(test_files),
        "train_files": [os.path.basename(f) for f in train_files],
        "val_files": [os.path.basename(f) for f in val_files],
        "test_files": [os.path.basename(f) for f in test_files],
    }
    manifest_path = "data/processed/v5/split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(split_manifest, mf, indent=2, ensure_ascii=False)
    print(
        f"Saved piece-level split manifest to {manifest_path}: "
        f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )

    train_seqs = []
    for f in train_files:
        seqs = tokenize_fugue_piece(f, augment=True)
        train_seqs.extend(seqs)

    val_seqs = []
    for f in val_files:
        seqs = tokenize_fugue_piece(f, augment=False)
        val_seqs.extend(seqs)

    test_seqs = []
    for f in test_files:
        seqs = tokenize_fugue_piece(f, augment=False)
        test_seqs.extend(seqs)

    print(
        f"Piece-split sequences -> Train (augmented): {len(train_seqs)}, "
        f"Val (original key): {len(val_seqs)}, Test (original key): {len(test_seqs)}"
    )

    tokenizer = FugueTokenizerV5()
    tokenizer.build_vocab(train_seqs + val_seqs + test_seqs)
    print(f"Vocab size: {tokenizer.vocab_size}")
    tokenizer.save("data/processed/v5/fugue_vocab_v5.pkl")

    dataset_payload = {
        "train": train_seqs,
        "val": val_seqs,
        "test": test_seqs,
        "split_manifest": split_manifest,
    }
    with open("data/processed/v5/fugue_dataset_v5.pkl", "wb") as f:
        pickle.dump(dataset_payload, f)
    print("Saved piece-split dataset to data/processed/v5/fugue_dataset_v5.pkl")
