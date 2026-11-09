"""NWC2 User Tool CLI entry point for Like_Bach."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import urllib.error
import urllib.request
from typing import Any

# Ensure project root is in sys.path and set as working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nwc.nwctxt_bridge import (
    build_clip_nwctxt,
    build_full_score_nwctxt,
    parse_nwctxt,
    parse_time_signature,
)


def ensure_default_v5_vocab(vocab_path: str) -> str:
    """Ensure a valid V5 vocabulary pickle exists so HybridFugueEngine initializes cleanly."""
    abs_path = os.path.join(PROJECT_ROOT, vocab_path)
    if os.path.exists(abs_path):
        return abs_path

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tokens: list[str] = ["[PAD]", "[UNK]", "[REST]", "[FINAL]", "[EPISODE_MODULATION]"]
    tokens.extend([
        "[SUBJECT_START]",
        "[SUBJECT_END]",
        "[ANSWER_START]",
        "[ANSWER_END]",
        "[COUNTERSUBJECT_START]",
        "[COUNTERSUBJECT_END]",
    ])
    for b in range(1, 129):
        tokens.append(f"[BAR_{b}]")
    for v in range(1, 6):
        tokens.append(f"[VOICE_{v}]")
    for p in range(24, 97):
        tokens.append(f"P{p}")
    for d in [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        tokens.append(f"D{d}")

    unique_tokens = sorted(set(tokens))
    itos = {i: tok for i, tok in enumerate(unique_tokens)}
    stoi = {tok: i for i, tok in itos.items()}
    with open(abs_path, "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos}, f)
    return abs_path


def _has_trained_weights(mode: str) -> bool:
    """Check whether trained PyTorch checkpoint weights exist on disk."""
    if mode == "fugue":
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "processed", "v5", "fugue_model_v5.pt"),
            os.path.join(PROJECT_ROOT, "models", "v4", "fugue_model.pt"),
        ]
    else:
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "processed", "v4", "bach_model.pt"),
            os.path.join(PROJECT_ROOT, "models", "v4", "fugue_model.pt"),
        ]
    return any(os.path.exists(p) for p in candidates)


def _try_api_generate(
    subject_notes: list[dict[str, Any]],
    mode: str,
    target_measures: int,
    temperature: float,
    api_url: str = "http://127.0.0.1:8000",
) -> list[dict[str, Any]] | None:
    """Try calling the local FastAPI server if it is running."""
    try:
        req = urllib.request.Request(f"{api_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=0.35) as resp:
            if resp.status != 200:
                return None
    except Exception:
        return None

    try:
        payload = json.dumps({
            "subject_notes": subject_notes,
            "target_measures": target_measures,
            "temperature": temperature,
            "refine_iters": 1,
        }).encode("utf-8")

        if mode == "chorale":
            req = urllib.request.Request(
                f"{api_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("success") and data.get("results"):
                    return data["results"]
    except Exception:
        return None
    return None


def _detect_scale_and_tonic(subject_notes: list[dict[str, Any]]) -> tuple[int, list[int]]:
    """Infer tonic pitch class (0..11) and scale pitch classes from subject notes."""
    if not subject_notes:
        return 0, [0, 2, 4, 5, 7, 9, 11]

    first_pc = int(subject_notes[0]["pitch"]) % 12
    last_pc = int(subject_notes[-1]["pitch"]) % 12
    pc_weights: dict[int, float] = {i: 0.0 for i in range(12)}
    for n in subject_notes:
        pc = int(n["pitch"]) % 12
        pc_weights[pc] += float(n["duration"])

    best_tonic = first_pc
    best_mode_intervals = [0, 2, 4, 5, 7, 9, 11]
    best_score = -1e9

    major_template = [0, 2, 4, 5, 7, 9, 11]
    minor_template = [0, 2, 3, 5, 7, 8, 11]

    for tonic in range(12):
        for template in (major_template, minor_template):
            scale_pcs = {(tonic + iv) % 12 for iv in template}
            score = sum(w for pc, w in pc_weights.items() if pc in scale_pcs)
            if tonic == first_pc:
                score += 2.0
            if tonic == last_pc or (tonic + 7) % 12 == last_pc:
                score += 1.5
            if score > best_score:
                best_score = score
                best_tonic = tonic
                best_mode_intervals = template

    scale_pcs_list = [(best_tonic + iv) % 12 for iv in best_mode_intervals]
    return best_tonic, scale_pcs_list


def _snap_to_scale(pitch: int, scale_pcs: list[int], low: int, high: int) -> int:
    """Snap a MIDI pitch to the nearest pitch in scale_pcs within [low, high]."""
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12

    if (pitch % 12) in scale_pcs:
        return max(low, min(high, pitch))

    for delta in (1, -1, 2, -2):
        cand = pitch + delta
        if low <= cand <= high and (cand % 12) in scale_pcs:
            return cand
    return max(low, min(high, pitch))


def generate_baroque_chorale_harmony(
    subject_notes: list[dict[str, Any]],
    target_measures: int = 8,
    beats_per_bar: float = 4.0,
) -> list[dict[str, Any]]:
    """
    Generate 4-voice Bach-style chorale harmonization (SATB) with functional harmony,
    contrary motion, and authentic cadence when neural weights are unavailable or need completion.
    """
    if not subject_notes:
        return []

    tonic_pc, scale_pcs = _detect_scale_and_tonic(subject_notes)
    is_minor = ((tonic_pc + 3) % 12) in scale_pcs

    # Extend soprano if subject is shorter than target_measures
    subject_end = max(float(n["offset"]) + float(n["duration"]) for n in subject_notes)
    total_beats = max(subject_end, target_measures * beats_per_bar)

    v1_notes: list[dict[str, Any]] = [
        {
            "pitch": int(n["pitch"]),
            "duration": round(float(n["duration"]), 4),
            "offset": round(float(n["offset"]), 4),
            "voice": 1,
        }
        for n in sorted(subject_notes, key=lambda x: float(x["offset"]))
    ]

    # If subject is shorter than target_measures, develop subject motifs to the final cadence
    cursor = round(subject_end, 4)
    motif_len = max(beats_per_bar, round(int((subject_end + beats_per_bar - 0.05) // beats_per_bar) * beats_per_bar, 4))
    step_idx = 0
    while cursor < total_beats - 1e-3:
        rem_beats = round(total_beats - cursor, 4)
        if rem_beats <= beats_per_bar + 1e-3:
            # Final cadence measure in Soprano: degree 2 -> degree 1 (Whole or Half+Half)
            tonic_soprano = _snap_to_scale(60 + tonic_pc + 12, scale_pcs, 60, 79)
            supertonic_soprano = _snap_to_scale(tonic_soprano + 2, scale_pcs, 60, 81)
            if rem_beats >= 4.0 - 1e-3:
                v1_notes.append({
                    "pitch": supertonic_soprano,
                    "duration": 2.0,
                    "offset": cursor,
                    "voice": 1,
                })
                v1_notes.append({
                    "pitch": tonic_soprano,
                    "duration": round(rem_beats - 2.0, 4),
                    "offset": round(cursor + 2.0, 4),
                    "voice": 1,
                })
            else:
                v1_notes.append({
                    "pitch": tonic_soprano,
                    "duration": rem_beats,
                    "offset": cursor,
                    "voice": 1,
                })
            cursor = total_beats
            break

        src_note = subject_notes[step_idx % len(subject_notes)]
        dur = min(float(src_note["duration"]), rem_beats - beats_per_bar)
        if dur < 0.25:
            dur = min(1.0, rem_beats - beats_per_bar)
        # Sequential diatonic transposition (-2 or +2 semitones)
        shift = 2 if (int(cursor // beats_per_bar) % 2 == 1) else -2
        new_p = _snap_to_scale(int(src_note["pitch"]) + shift, scale_pcs, 60, 81)
        v1_notes.append({
            "pitch": new_p,
            "duration": round(dur, 4),
            "offset": cursor,
            "voice": 1,
        })
        cursor = round(cursor + dur, 4)
        step_idx += 1

    all_notes: list[dict[str, Any]] = list(v1_notes)
    prev_bass = 48 + tonic_pc
    if prev_bass > 57:
        prev_bass -= 12

    for idx, sn in enumerate(v1_notes):
        sp = int(sn["pitch"])
        off = float(sn["offset"])
        dur = float(sn["duration"])
        is_last = idx == len(v1_notes) - 1
        is_penultimate = idx == len(v1_notes) - 2

        if is_last:
            # Final Tonic chord (Picardy third in minor or standard triad)
            root_pc = tonic_pc
            third_pc = (tonic_pc + (4 if not is_minor else 3)) % 12
            fifth_pc = (tonic_pc + 7) % 12
            bass_p = 48 + root_pc
            if bass_p > 55:
                bass_p -= 12
            tenor_p = 48 + fifth_pc
            while tenor_p <= bass_p + 2:
                tenor_p += 12
            while tenor_p >= sp - 3:
                tenor_p -= 12
            alto_p = 60 + third_pc
            while alto_p <= tenor_p + 2:
                alto_p += 12
            while alto_p >= sp - 1:
                alto_p -= 12
        elif is_penultimate:
            # Dominant (V) chord at cadence
            dom_pc = (tonic_pc + 7) % 12
            lead_pc = (tonic_pc + 11) % 12
            re_pc = (tonic_pc + 2) % 12
            bass_p = 48 + dom_pc
            if bass_p > 55:
                bass_p -= 12
            tenor_p = 48 + re_pc
            while tenor_p <= bass_p + 2:
                tenor_p += 12
            alto_p = 60 + lead_pc
            while alto_p >= sp - 1:
                alto_p -= 12
            while alto_p <= tenor_p:
                alto_p += 12
        else:
            # Functional triad containing or harmonizing with soprano pitch
            # Choose bass in contrary motion to soprano
            s_pc = sp % 12
            if s_pc in scale_pcs:
                deg = scale_pcs.index(s_pc)
                # Build triad where soprano is root, 3rd, or 5th
                triad_choice = (idx % 3)
                if triad_choice == 0:
                    root_deg = (deg - 2) % 7
                elif triad_choice == 1:
                    root_deg = (deg - 4) % 7
                else:
                    root_deg = deg
                root_pc = scale_pcs[root_deg]
                third_pc = scale_pcs[(root_deg + 2) % 7]
                fifth_pc = scale_pcs[(root_deg + 4) % 7]
            else:
                root_pc = (sp - 4) % 12
                third_pc = sp % 12
                fifth_pc = (root_pc + 7) % 12

            bass_p = 48 + root_pc
            if abs((bass_p - 12) - prev_bass) < abs(bass_p - prev_bass) and (bass_p - 12) >= 40:
                bass_p -= 12
            elif bass_p > 57:
                bass_p -= 12
            # Avoid parallel fifth/octave with previous bass
            if bass_p == prev_bass and root_pc != third_pc:
                bass_p = 48 + third_pc
                if bass_p > 57:
                    bass_p -= 12
            prev_bass = bass_p

            alto_target = sp - (4 if (sp - third_pc) % 12 == 0 else 5)
            alto_p = _snap_to_scale(alto_target, [root_pc, third_pc, fifth_pc], 53, min(74, sp - 2))
            tenor_target = alto_p - 5
            tenor_p = _snap_to_scale(tenor_target, [root_pc, third_pc, fifth_pc], max(45, bass_p + 3), min(67, alto_p - 2))

        all_notes.append({"pitch": int(alto_p), "duration": dur, "offset": off, "voice": 2})
        all_notes.append({"pitch": int(tenor_p), "duration": dur, "offset": off, "voice": 3})
        all_notes.append({"pitch": int(bass_p), "duration": dur, "offset": off, "voice": 4})

    return all_notes


def generate_baroque_fugue_polyphony(
    subject_notes: list[dict[str, Any]],
    target_measures: int = 8,
    beats_per_bar: float = 4.0,
) -> list[dict[str, Any]]:
    """
    Generate a 4-voice Bach-style Fugue (Exposition with Subject/Answer entries,
    Countersubjects in rhythmic/contrary counterpoint, Episodes, and Final Cadence).
    """
    if not subject_notes:
        return []

    tonic_pc, scale_pcs = _detect_scale_and_tonic(subject_notes)
    subject_end_beat = max(float(n["offset"]) + float(n["duration"]) for n in subject_notes)
    subject_measures = max(1, int((subject_end_beat + beats_per_bar - 0.05) // beats_per_bar))
    entry_span_beats = subject_measures * beats_per_bar

    total_measures = max(target_measures, 4 * subject_measures)
    total_beats = total_measures * beats_per_bar

    voice_ranges = {
        1: (60, 81),  # Soprano
        2: (53, 74),  # Alto
        3: (46, 67),  # Tenor
        4: (38, 60),  # Bass
    }
    transpositions = {
        1: 0,    # Subject (Soprano)
        2: -7,   # Answer 5th down (Alto)
        3: -12,  # Subject Octave down (Tenor)
        4: -19,  # Answer Octave + 5th down (Bass)
    }

    # Build a complementary countersubject template from the subject
    sorted_subj = sorted(subject_notes, key=lambda x: float(x["offset"]))
    first_pitch = int(sorted_subj[0]["pitch"])
    cs_template: list[dict[str, Any]] = []
    for idx, sn in enumerate(sorted_subj):
        rel_off = float(sn["offset"])
        dur = float(sn["duration"])
        # Invert interval around first_pitch and shift by a 3rd/6th for consonant counterpoint
        inv_pitch = first_pitch - (int(sn["pitch"]) - first_pitch) + 4
        if dur >= 1.0 and idx % 2 == 0:
            # Subdivide long subject notes into flowing 8th/quarter motion in the countersubject
            half_d = round(dur / 2.0, 4)
            cs_template.append({"rel_offset": rel_off, "duration": half_d, "pitch": inv_pitch})
            cs_template.append({"rel_offset": round(rel_off + half_d, 4), "duration": half_d, "pitch": inv_pitch + 2})
        else:
            cs_template.append({"rel_offset": rel_off, "duration": dur, "pitch": inv_pitch})

    all_notes: list[dict[str, Any]] = []

    for v in range(1, 5):
        v_min, v_max = voice_ranges[v]
        entry_start = (v - 1) * entry_span_beats
        entry_end = v * entry_span_beats
        trans = transpositions[v]

        # 1. Subject / Answer Exposition entry
        if entry_start < total_beats:
            for sn in sorted_subj:
                off = round(entry_start + float(sn["offset"]), 4)
                if off >= total_beats:
                    break
                dur = min(float(sn["duration"]), total_beats - off)
                raw_p = int(sn["pitch"]) + trans
                p = _snap_to_scale(raw_p, scale_pcs, v_min, v_max) if v in (2, 4) else max(v_min, min(v_max, raw_p))
                all_notes.append({
                    "pitch": int(p),
                    "duration": round(dur, 4),
                    "offset": off,
                    "voice": v,
                })

        # 2. Countersubject & Episode blocks after the voice's exposition entry
        block_start = entry_end
        block_idx = 0
        while block_start < total_beats - 1e-3:
            rem_to_end = round(total_beats - block_start, 4)
            if rem_to_end <= beats_per_bar + 1e-3:
                # Final cadence measure for each voice
                cad_pitches = {
                    1: _snap_to_scale(60 + tonic_pc + 12, scale_pcs, v_min, v_max),
                    2: _snap_to_scale(60 + tonic_pc + 4, scale_pcs, v_min, v_max),
                    3: _snap_to_scale(48 + tonic_pc + 7, scale_pcs, v_min, v_max),
                    4: _snap_to_scale(36 + tonic_pc, scale_pcs, v_min, v_max),
                }
                dom_pitches = {
                    1: _snap_to_scale(cad_pitches[1] + 2, scale_pcs, v_min, v_max),
                    2: _snap_to_scale(cad_pitches[2] - 1, scale_pcs, v_min, v_max),
                    3: _snap_to_scale(cad_pitches[3] + 2, scale_pcs, v_min, v_max),
                    4: _snap_to_scale(cad_pitches[4] + 7, scale_pcs, v_min, v_max),
                }
                if rem_to_end >= 4.0 - 1e-3:
                    all_notes.append({
                        "pitch": int(dom_pitches[v]),
                        "duration": 2.0,
                        "offset": round(block_start, 4),
                        "voice": v,
                    })
                    all_notes.append({
                        "pitch": int(cad_pitches[v]),
                        "duration": round(rem_to_end - 2.0, 4),
                        "offset": round(block_start + 2.0, 4),
                        "voice": v,
                    })
                else:
                    all_notes.append({
                        "pitch": int(cad_pitches[v]),
                        "duration": rem_to_end,
                        "offset": round(block_start, 4),
                        "voice": v,
                    })
                break

            # Generate one span of countersubject / episode sequence
            seq_shift = ((block_idx + v) % 4) * (-2)
            for csn in cs_template:
                off = round(block_start + float(csn["rel_offset"]), 4)
                if off >= total_beats - beats_per_bar:
                    break
                dur = min(float(csn["duration"]), total_beats - beats_per_bar - off)
                if dur < 0.125:
                    continue
                raw_p = int(csn["pitch"]) + trans + seq_shift
                p = _snap_to_scale(raw_p, scale_pcs, v_min, v_max)
                all_notes.append({
                    "pitch": int(p),
                    "duration": round(dur, 4),
                    "offset": off,
                    "voice": v,
                })

            block_start = round(block_start + entry_span_beats, 4)
            block_idx += 1

    return all_notes


def run_generation(
    subject_notes: list[dict[str, Any]],
    mode: str = "chorale",
    target_measures: int = 8,
    temperature: float = 0.7,
    beats_per_bar: float = 4.0,
    force_neural: bool = False,
) -> list[dict[str, Any]]:
    """Run AI generation via FastAPI, local PyTorch neural engine, or Baroque counterpoint synthesizer."""
    if not subject_notes:
        return []

    # 1. Try running FastAPI server first for zero-latency warm inference
    api_notes = _try_api_generate(subject_notes, mode, target_measures, temperature)
    if api_notes:
        return api_notes

    # 2. If trained checkpoint weights exist (or --force-neural is specified), run PyTorch engine
    if force_neural or _has_trained_weights(mode):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                from src.v5.neural_engine import HybridFugueEngine, NeuralBachEngine

                if mode == "fugue":
                    v5_vocab = ensure_default_v5_vocab("data/processed/v5/fugue_vocab_v5.pkl")
                    v5_model = os.path.join(PROJECT_ROOT, "data", "processed", "v5", "fugue_model_v5.pt")
                    engine = HybridFugueEngine(model_path=v5_model, tokenizer_path=v5_vocab)
                    notes = engine.generate_fugue(
                        subject_notes=subject_notes,
                        target_measures=target_measures,
                        temperature=temperature,
                    )
                    if notes:
                        return notes
                else:
                    v4_vocab = os.path.join(PROJECT_ROOT, "models", "v4", "fugue_vocab.pkl")
                    v4_tok = os.path.join(PROJECT_ROOT, "data", "processed", "v4", "tokenizer.pkl")
                    tok_path = v4_tok if os.path.exists(v4_tok) else v4_vocab
                    model_path = os.path.join(PROJECT_ROOT, "data", "processed", "v4", "bach_model.pt")
                    engine = NeuralBachEngine(model_path=model_path, tokenizer_path=tok_path)
                    notes = engine.generate_response(
                        subject_notes=subject_notes,
                        target_measures=target_measures,
                        temperature=temperature,
                    )
                    if notes:
                        return notes
        except Exception:
            pass

    # 3. Rule-guided Baroque counterpoint / chorale synthesizer fallback
    if mode == "fugue":
        return generate_baroque_fugue_polyphony(
            subject_notes=subject_notes,
            target_measures=target_measures,
            beats_per_bar=beats_per_bar,
        )
    return generate_baroque_chorale_harmony(
        subject_notes=subject_notes,
        target_measures=target_measures,
        beats_per_bar=beats_per_bar,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point invoked by NoteWorthy Composer 2 User Tool."""
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Like_Bach NWC2 User Tool")
    parser.add_argument(
        "--mode",
        type=str,
        default="chorale",
        help="Generation mode: chorale or fugue (case-insensitive)",
    )
    parser.add_argument(
        "--measures",
        type=int,
        default=8,
        help="Target total measures to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature (e.g. 0.55 to 0.9)",
    )
    parser.add_argument(
        "--clip-style",
        type=str,
        default="chord",
        choices=["chord", "melody"],
        help="Output style when invoked in ClipText mode",
    )
    parser.add_argument(
        "--force-neural",
        action="store_true",
        help="Force running the PyTorch Transformer model even without trained weights",
    )

    args = parser.parse_args(argv)
    mode = args.mode.strip().lower()
    if "fugue" in mode:
        mode = "fugue"
    else:
        mode = "chorale"

    target_measures = max(2, min(128, int(args.measures)))
    temperature = max(0.1, min(2.0, float(args.temp)))

    # Read NWCTXT from stdin
    raw_input = sys.stdin.read()
    if not raw_input or not raw_input.strip():
        sys.stderr.write("Like_Bach Error: No NWC notation text received on STDIN.\n")
        return 1

    parsed_doc = parse_nwctxt(raw_input)
    subject_notes = parsed_doc.get("subject_notes", [])
    if not subject_notes:
        sys.stderr.write(
            "Like_Bach Notice: Please enter at least one note on the staff before running Like_Bach.\n"
        )
        return 1

    beats_per_bar = parse_time_signature(parsed_doc.get("time_sig_str", "4/4"))
    subj_end = max(float(n["offset"]) + float(n["duration"]) for n in subject_notes)
    subj_measures = max(1, int((subj_end + beats_per_bar - 0.05) // beats_per_bar))

    if mode == "fugue":
        effective_measures = max(target_measures, 4 * subj_measures)
    else:
        effective_measures = max(target_measures, subj_measures)

    generated_notes = run_generation(
        subject_notes=subject_notes,
        mode=mode,
        target_measures=effective_measures,
        temperature=temperature,
        beats_per_bar=beats_per_bar,
        force_neural=args.force_neural,
    )

    if not generated_notes:
        sys.stderr.write("Like_Bach Error: Failed to generate notes.\n")
        return 1

    if parsed_doc.get("mode") == "ClipText":
        output_nwctxt = build_clip_nwctxt(
            parsed_doc=parsed_doc,
            generated_notes=generated_notes,
            mode=mode,
            target_measures=effective_measures,
            clip_style=args.clip_style,
        )
    else:
        output_nwctxt = build_full_score_nwctxt(
            parsed_doc=parsed_doc,
            generated_notes=generated_notes,
            mode=mode,
            target_measures=effective_measures,
        )

    sys.stdout.write(output_nwctxt)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
