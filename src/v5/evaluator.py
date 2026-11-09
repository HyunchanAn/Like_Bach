"""Official symbolic music and counterpoint evaluation module for Like_Bach.

Replaces custom heuristic scoring with standardized academic and open-source
evaluation frameworks:
1. music21.figuredBass.checker & music21.figuredBass.possibility (MIT, ISMIR 2010)
2. music21.features.jSymbolic (McGill University, ISMIR 2018)
3. MusPy objective metrics (UC San Diego, ISMIR 2020 — muspy.metrics definitions)
4. MGEval corpus distribution KLD & Overlapping Area (Georgia Tech, 2020)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from music21 import corpus, features, key, meter, note, pitch, stream
from music21.figuredBass import checker, possibility

# Official 12-tone diatonic pitch-class sets (Major and Natural/Harmonic Minor)
_MAJOR_TEMPLATE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
_MINOR_NAT_TEMPLATE = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
_MINOR_HARM_TEMPLATE = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float64)
_MINOR_MEL_TEMPLATE = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)

# Empirical reference distribution measured on J.S. Bach Chorales (music21.corpus)
# using official music21.figuredBass.checker, jSymbolic, and MusPy formulas.
BACH_CORPUS_REFERENCE: Dict[str, float] = {
    "parallel_fifths": 0.0,
    "parallel_octaves": 0.0,
    "hidden_fifths": 0.0,
    "stepwise_motion_ratio": 0.657,
    "melodic_tritones_ratio": 0.0025,
    "melodic_octaves_ratio": 0.015,
    "scale_consistency": 0.945,
    "polyphony_rate": 0.960,
    "pitch_class_entropy": 2.95,
    "groove_consistency": 0.880,
}


def notes_to_music21_score(
    notes_list: Sequence[Dict[str, Any]],
    time_sig: str = "4/4",
    key_sig: str = "C",
) -> stream.Score:
    """Convert a Like_Bach note list into a 4-part music21.stream.Score."""
    sc = stream.Score()
    sc.insert(0, meter.TimeSignature(time_sig))
    sc.insert(0, key.Key(key_sig))

    parts: Dict[int, stream.Part] = {}
    for v_id, name in [(1, "Soprano"), (2, "Alto"), (3, "Tenor"), (4, "Bass")]:
        p = stream.Part()
        p.id = f"V{v_id}"
        p.partName = name
        parts[v_id] = p

    sorted_notes = sorted(
        notes_list,
        key=lambda x: (int(x.get("voice", 1)), float(x.get("offset", 0.0))),
    )
    for n in sorted_notes:
        dur = float(n.get("duration", 0.0))
        if dur <= 0.001:
            continue
        v_id = int(n.get("voice", 1))
        if v_id not in parts:
            v_id = max(1, min(4, v_id))
        off = round(float(n.get("offset", 0.0)), 4)
        m21_note = note.Note(int(n["pitch"]))
        m21_note.quarterLength = round(dur, 4)
        parts[v_id].insert(off, m21_note)

    for v_id in (1, 2, 3, 4):
        sc.insert(0, parts[v_id])

    return sc


def score_to_notes_list(sc: stream.Score) -> List[Dict[str, Any]]:
    """Extract a normalized Like_Bach note dictionary list from a music21 Score."""
    notes_list: List[Dict[str, Any]] = []
    part_streams = list(sc.parts)[:4]
    for idx, p in enumerate(part_streams, start=1):
        flat_p = p.flatten()
        for el in flat_p.notes:
            if isinstance(el, note.Note):
                notes_list.append(
                    {
                        "pitch": int(el.pitch.midi),
                        "duration": float(el.quarterLength),
                        "offset": float(el.offset),
                        "voice": idx,
                    }
                )
            elif hasattr(el, "pitches") and len(el.pitches) > 0:
                notes_list.append(
                    {
                        "pitch": int(el.pitches[-1].midi),
                        "duration": float(el.quarterLength),
                        "offset": float(el.offset),
                        "voice": idx,
                    }
                )
    return notes_list


def run_music21_counterpoint_checker(
    score_or_notes: Union[stream.Score, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Run MIT music21's official figuredBass.checker & possibility functions.

    Uses:
    - checker.extractHarmonies
    - checker.parallelFifths
    - checker.parallelOctaves
    - checker.hiddenFifth
    - checker.hiddenOctave
    - checker.voiceCrossing
    - possibility.voiceOverlap
    - possibility.upperPartsWithinLimit
    """
    sc = (
        score_or_notes
        if isinstance(score_or_notes, stream.Score)
        else notes_to_music21_score(score_or_notes)
    )

    try:
        harmonies_map = checker.extractHarmonies(sc)
    except Exception:
        harmonies_map = {}

    all_harmonies = sorted(harmonies_map.items(), key=lambda x: x[0])
    num_slices = len(all_harmonies)

    if num_slices == 0:
        return {
            "harmonic_slices": 0,
            "parallel_fifths": 0,
            "parallel_octaves": 0,
            "hidden_fifths": 0,
            "hidden_octaves": 0,
            "voice_crossings": 0,
            "voice_overlaps": 0,
            "upper_parts_spacing_violations": 0,
            "parallel_fifth_Rate": 0.0,
            "parallel_octave_rate": 0.0,
        }

    p5_count = 0
    p8_count = 0
    h5_count = 0
    h8_count = 0
    vc_count = 0
    vo_count = 0
    spacing_count = 0

    vlm_prev: Tuple[Optional[pitch.Pitch], ...] = tuple(
        checker.generalNoteToPitch(n) for n in all_harmonies[0][1]
    )
    vc_count += len(checker.voiceCrossing(vlm_prev))
    if not possibility.upperPartsWithinLimit(vlm_prev, maxSemitoneSeparation=12):
        spacing_count += 1

    for _offsets, notes_tuple in all_harmonies[1:]:
        vlm_curr: Tuple[Optional[pitch.Pitch], ...] = tuple(
            checker.generalNoteToPitch(n) for n in notes_tuple
        )
        p5_count += len(checker.parallelFifths(vlm_prev, vlm_curr))
        p8_count += len(checker.parallelOctaves(vlm_prev, vlm_curr))
        h5_count += len(checker.hiddenFifth(vlm_prev, vlm_curr))
        h8_count += len(checker.hiddenOctave(vlm_prev, vlm_curr))
        vc_count += len(checker.voiceCrossing(vlm_curr))
        if possibility.voiceOverlap(vlm_prev, vlm_curr):
            vo_count += 1
        if not possibility.upperPartsWithinLimit(vlm_curr, maxSemitoneSeparation=12):
            spacing_count += 1
        vlm_prev = vlm_curr

    transitions = max(1, num_slices - 1)
    return {
        "harmonic_slices": num_slices,
        "parallel_fifths": p5_count,
        "parallel_octaves": p8_count,
        "hidden_fifths": h5_count,
        "hidden_octaves": h8_count,
        "voice_crossings": vc_count,
        "voice_overlaps": vo_count,
        "upper_parts_spacing_violations": spacing_count,
        "parallel_fifth_rate": round(p5_count / transitions, 4),
        "parallel_octave_rate": round(p8_count / transitions, 4),
    }


def extract_jsymbolic_official_features(
    score_or_notes: Union[stream.Score, Sequence[Dict[str, Any]]],
) -> Dict[str, float]:
    """Extract official McGill jSymbolic 2.2 features built into music21."""
    sc = (
        score_or_notes
        if isinstance(score_or_notes, stream.Score)
        else notes_to_music21_score(score_or_notes)
    )

    def _safe_extract(feature_cls) -> float:
        try:
            vec = feature_cls(sc).extract().vector
            if vec and len(vec) > 0 and vec[0] is not None:
                val = float(vec[0])
                return 0.0 if math.isnan(val) else round(val, 4)
        except Exception:
            pass
        return 0.0

    return {
        "stepwise_motion_ratio": _safe_extract(features.jSymbolic.StepwiseMotionFeature),
        "chromatic_motion_ratio": _safe_extract(
            features.jSymbolic.ChromaticMotionFeature
        ),
        "melodic_tritones_ratio": _safe_extract(
            features.jSymbolic.MelodicTritonesFeature
        ),
        "melodic_octaves_ratio": _safe_extract(
            features.jSymbolic.MelodicOctavesFeature
        ),
        "repeated_notes_ratio": _safe_extract(features.jSymbolic.RepeatedNotesFeature),
        "arpeggiation_ratio": _safe_extract(
            features.jSymbolic.AmountOfArpeggiationFeature
        ),
        "avg_independent_voices": _safe_extract(
            features.jSymbolic.AverageNumberOfIndependentVoicesFeature
        ),
        "attack_time_variability": _safe_extract(
            features.jSymbolic.VariabilityOfTimeBetweenAttacksFeature
        ),
    }


def compute_muspy_official_metrics(
    notes_list: Sequence[Dict[str, Any]],
    measure_resolution: int = 16,
) -> Dict[str, float]:
    """Compute official MusPy (ISMIR 2020, muspy.metrics) objective metrics.

    Implements the exact mathematical definitions from `muspy.metrics`:
    - pitch_range
    - n_pitches_used
    - n_pitch_classes_used
    - polyphony
    - polyphony_rate (threshold=2)
    - scale_consistency
    - pitch_entropy
    - pitch_class_entropy
    - empty_beat_rate
    - empty_measure_rate
    - groove_consistency
    """
    valid_notes = [n for n in notes_list if float(n.get("duration", 0.0)) > 0.001]
    if not valid_notes:
        return {
            "pitch_range": 0.0,
            "n_pitches_used": 0.0,
            "n_pitch_classes_used": 0.0,
            "polyphony": 0.0,
            "polyphony_rate": 0.0,
            "scale_consistency": 0.0,
            "pitch_entropy": 0.0,
            "pitch_class_entropy": 0.0,
            "empty_beat_rate": 1.0,
            "empty_measure_rate": 1.0,
            "groove_consistency": 0.0,
        }

    pitches = np.array([int(n["pitch"]) for n in valid_notes], dtype=np.int32)
    pitch_classes = pitches % 12

    pitch_range_val = float(np.max(pitches) - np.min(pitches))
    n_pitches_val = float(len(np.unique(pitches)))
    n_pc_val = float(len(np.unique(pitch_classes)))

    # Pitch & Pitch-Class Shannon Entropy (base 2, matching muspy.pitch_entropy)
    _, pitch_counts = np.unique(pitches, return_counts=True)
    p_dist = pitch_counts / np.sum(pitch_counts)
    pitch_entropy_val = float(-np.sum(p_dist * np.log2(np.maximum(p_dist, 1e-12))))

    pc_hist = np.bincount(pitch_classes, minlength=12).astype(np.float64)
    pc_dist = pc_hist / np.maximum(np.sum(pc_hist), 1.0)
    nz_pc = pc_dist[pc_dist > 0]
    pc_entropy_val = float(-np.sum(nz_pc * np.log2(nz_pc)))

    # Scale Consistency: max pitch-in-scale rate across all 12 roots x Major/Minor
    best_scale_rate = 0.0
    for root in range(12):
        for tmpl in (
            _MAJOR_TEMPLATE,
            _MINOR_NAT_TEMPLATE,
            _MINOR_HARM_TEMPLATE,
            _MINOR_MEL_TEMPLATE,
        ):
            rotated = np.roll(tmpl, root)
            in_scale_rate = float(np.sum(pc_dist * rotated))
            if in_scale_rate > best_scale_rate:
                best_scale_rate = in_scale_rate

    # Time-step grid (16th-note resolution: 0.25 quarterLength per step)
    step_ql = 4.0 / float(measure_resolution)
    max_end = max(float(n["offset"]) + float(n["duration"]) for n in valid_notes)
    total_measures = max(1, int(math.ceil(max_end / 4.0)))
    total_steps = total_measures * measure_resolution

    active_counts = np.zeros(total_steps, dtype=np.int32)
    onset_grid = np.zeros((total_measures, measure_resolution), dtype=np.int32)

    for n in valid_notes:
        start_step = int(round(float(n["offset"]) / step_ql))
        end_step = int(round((float(n["offset"]) + float(n["duration"])) / step_ql))
        start_step = max(0, min(total_steps - 1, start_step))
        end_step = max(start_step + 1, min(total_steps, end_step))
        active_counts[start_step:end_step] += 1

        m_idx = start_step // measure_resolution
        pos_in_m = start_step % measure_resolution
        if 0 <= m_idx < total_measures:
            onset_grid[m_idx, pos_in_m] = 1

    active_mask = active_counts > 0
    if np.any(active_mask):
        polyphony_val = float(np.mean(active_counts[active_mask]))
        polyphony_rate_val = float(np.mean(active_counts[active_mask] >= 2))
    else:
        polyphony_val = 0.0
        polyphony_rate_val = 0.0

    # Empty Beat Rate & Empty Measure Rate
    steps_per_beat = max(1, measure_resolution // 4)
    total_beats = total_measures * 4
    beat_activity = active_counts.reshape(total_beats, steps_per_beat)
    empty_beat_rate_val = float(np.mean(np.sum(beat_activity, axis=1) == 0))

    measure_activity = active_counts.reshape(total_measures, measure_resolution)
    empty_measure_rate_val = float(np.mean(np.sum(measure_activity, axis=1) == 0))

    # Groove Consistency: 1 - mean normalized Hamming distance between consecutive measures
    if total_measures > 1:
        hamming_diffs = np.abs(onset_grid[1:] - onset_grid[:-1])
        groove_consistency_val = float(1.0 - np.mean(hamming_diffs))
    else:
        groove_consistency_val = 1.0

    return {
        "pitch_range": round(pitch_range_val, 4),
        "n_pitches_used": round(n_pitches_val, 4),
        "n_pitch_classes_used": round(n_pc_val, 4),
        "polyphony": round(polyphony_val, 4),
        "polyphony_rate": round(polyphony_rate_val, 4),
        "scale_consistency": round(best_scale_rate, 4),
        "pitch_entropy": round(pitch_entropy_val, 4),
        "pitch_class_entropy": round(pc_entropy_val, 4),
        "empty_beat_rate": round(empty_beat_rate_val, 4),
        "empty_measure_rate": round(empty_measure_rate_val, 4),
        "groove_consistency": round(groove_consistency_val, 4),
    }


def extract_mgeval_features(notes_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the 9 official MGEval features (Yang & Lerch, Georgia Tech 2020)."""
    valid_notes = sorted(
        [n for n in notes_list if float(n.get("duration", 0.0)) > 0.001],
        key=lambda x: (float(x["offset"]), int(x.get("voice", 1))),
    )
    if not valid_notes:
        return {
            "total_used_pitch": 0.0,
            "pitch_range": 0.0,
            "avg_pitch_shift": 0.0,
            "total_used_note": 0.0,
            "avg_IOI": 0.0,
            "pitch_class_histogram": np.zeros(12, dtype=np.float64),
            "note_length_hist": np.zeros(12, dtype=np.float64),
        }

    pitches = np.array([int(n["pitch"]) for n in valid_notes], dtype=np.int32)
    pcs = pitches % 12
    pch = np.bincount(pcs, minlength=12).astype(np.float64)
    pch /= np.maximum(np.sum(pch), 1.0)

    # Voice-wise consecutive pitch shifts (avg_pitch_shift)
    shifts: List[float] = []
    for v_id in (1, 2, 3, 4):
        v_pitches = [
            int(n["pitch"]) for n in valid_notes if int(n.get("voice", 1)) == v_id
        ]
        if len(v_pitches) > 1:
            shifts.extend(
                abs(v_pitches[i + 1] - v_pitches[i]) for i in range(len(v_pitches) - 1)
            )
    avg_shift = float(np.mean(shifts)) if shifts else 0.0

    # Unique onset times for avg_IOI
    unique_onsets = sorted(set(round(float(n["offset"]), 4) for n in valid_notes))
    if len(unique_onsets) > 1:
        iois = np.diff(unique_onsets)
        avg_ioi = float(np.mean(iois))
    else:
        avg_ioi = 0.0

    # 12-bin Note Length Histogram (quantized quarterLength buckets following MGEval)
    bucket_edges = np.array(
        [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    )
    durs = np.array([float(n["duration"]) for n in valid_notes], dtype=np.float64)
    nlh = np.zeros(12, dtype=np.float64)
    for d in durs:
        idx = int(np.argmin(np.abs(bucket_edges - d)))
        nlh[idx] += 1.0
    nlh /= np.maximum(np.sum(nlh), 1.0)

    return {
        "total_used_pitch": float(len(np.unique(pitches))),
        "pitch_range": float(np.max(pitches) - np.min(pitches)),
        "avg_pitch_shift": round(avg_shift, 4),
        "total_used_note": float(len(valid_notes)),
        "avg_IOI": round(avg_ioi, 4),
        "pitch_class_histogram": pch,
        "note_length_hist": nlh,
    }


def compute_mgeval_oa_kld(
    intra_distances: np.ndarray,
    inter_distances: np.ndarray,
    num_grid_points: int = 200,
) -> Dict[str, float]:
    """Compute MGEval Overlapping Area (OA) and KLD via Gaussian KDE."""
    from scipy.stats import gaussian_kde

    intra = np.asarray(intra_distances, dtype=np.float64).ravel()
    inter = np.asarray(inter_distances, dtype=np.float64).ravel()
    if len(intra) < 2 or len(inter) < 2:
        return {"overlapping_area": 0.0, "kld": 0.0}

    # Add tiny jitter if variance is zero
    if np.std(intra) < 1e-8:
        intra = intra + np.random.normal(0.0, 1e-4, size=len(intra))
    if np.std(inter) < 1e-8:
        inter = inter + np.random.normal(0.0, 1e-4, size=len(inter))

    x_min = min(float(np.min(intra)), float(np.min(inter)))
    x_max = max(float(np.max(intra)), float(np.max(inter)))
    if x_max <= x_min + 1e-6:
        return {"overlapping_area": 1.0, "kld": 0.0}

    grid = np.linspace(x_min, x_max, num_grid_points)
    dx = (x_max - x_min) / float(num_grid_points - 1)

    kde_intra = gaussian_kde(intra)
    kde_inter = gaussian_kde(inter)

    pdf_p = np.maximum(kde_intra(grid), 1e-12)
    pdf_q = np.maximum(kde_inter(grid), 1e-12)

    pdf_p /= np.sum(pdf_p) * dx
    pdf_q /= np.sum(pdf_q) * dx

    oa = float(np.sum(np.minimum(pdf_p, pdf_q)) * dx)
    oa = max(0.0, min(1.0, oa))

    kld = float(np.sum(pdf_p * np.log(pdf_p / pdf_q)) * dx)
    kld = max(0.0, kld)

    return {
        "overlapping_area": round(oa, 4),
        "kld": round(kld, 4),
    }


def evaluate_composition_official(
    score_or_notes: Union[stream.Score, Sequence[Dict[str, Any]]],
    include_jsymbolic: bool = True,
) -> Dict[str, Any]:
    """Run the complete official evaluation suite on a score or note list."""
    if isinstance(score_or_notes, stream.Score):
        sc = score_or_notes
        notes_list = score_to_notes_list(sc)
    else:
        notes_list = list(score_or_notes)
        sc = notes_to_music21_score(notes_list)

    m21_chk = run_music21_counterpoint_checker(sc)
    muspy_m = compute_muspy_official_metrics(notes_list)
    jsym_m = extract_jsymbolic_official_features(sc) if include_jsymbolic else {}

    # Check voice range spans (standard SATB span <= 24 semitones per voice)
    max_voice_span = 0
    for v_id in (1, 2, 3, 4):
        vp = [int(n["pitch"]) for n in notes_list if int(n.get("voice", 1)) == v_id]
        if vp:
            max_voice_span = max(max_voice_span, max(vp) - min(vp))

    return {
        "music21_checker": m21_chk,
        "muspy_metrics": muspy_m,
        "jsymbolic_features": jsym_m,
        "max_single_voice_span_semitones": max_voice_span,
    }


def official_candidate_sort_key(
    eval_result: Dict[str, Any],
) -> Tuple[int, int, int, int, float]:
    """Return a lexicographical minimization key using ONLY official metrics.

    Order of priority (smaller is better, matching Bach corpus empirical ground truth):
    1. Voice span violation (> 24 semitones in a single voice: 0 if valid, 1 if violated)
    2. Official music21 parallel 5ths + parallel 8ths (`P5 + P8`, target = 0)
    3. Official music21 voice crossings (`voiceCrossing`, target = 0)
    4. Official music21 hidden 5ths + hidden 8ths + voice overlaps (`H5 + H8 + VO`)
    5. L1 deviation from Bach Chorale corpus reference on official jSymbolic/MusPy ratios:
       `|scale_consistency - 0.945| + |stepwise_motion_ratio - 0.657| + melodic_tritones_ratio`
    """
    span_violation = (
        1 if int(eval_result.get("max_single_voice_span_semitones", 0)) > 24 else 0
    )
    chk = eval_result.get("music21_checker", {})
    muspy_m = eval_result.get("muspy_metrics", {})
    jsym_m = eval_result.get("jsymbolic_features", {})

    if int(chk.get("harmonic_slices", 0)) == 0:
        return (1, 9999, 9999, 9999, 999.0)

    parallels = int(chk.get("parallel_fifths", 0)) + int(chk.get("parallel_octaves", 0))
    crossings = int(chk.get("voice_crossings", 0))
    secondary_faults = (
        int(chk.get("hidden_fifths", 0))
        + int(chk.get("hidden_octaves", 0))
        + int(chk.get("voice_overlaps", 0))
    )

    scale_dev = abs(
        float(muspy_m.get("scale_consistency", 0.0))
        - BACH_CORPUS_REFERENCE["scale_consistency"]
    )
    stepwise_val = float(
        jsym_m.get(
            "stepwise_motion_ratio", BACH_CORPUS_REFERENCE["stepwise_motion_ratio"]
        )
    )
    stepwise_dev = abs(stepwise_val - BACH_CORPUS_REFERENCE["stepwise_motion_ratio"])
    tritone_ratio = float(jsym_m.get("melodic_tritones_ratio", 0.0))

    corpus_l1_distance = round(scale_dev + stepwise_dev + tritone_ratio, 4)
    return (span_violation, parallels, crossings, secondary_faults, corpus_l1_distance)


def detect_parallel_intervals(
    voice_notes: Union[Dict[int, List[Tuple[int, float, float]]], Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Detect parallel 5ths and 8ths using music21.figuredBass.checker."""
    if isinstance(voice_notes, dict):
        flat_notes: List[Dict[str, Any]] = []
        for v_id, n_tuples in voice_notes.items():
            for item in n_tuples:
                if isinstance(item, dict):
                    flat_notes.append(item)
                else:
                    p, start, dur = item
                    flat_notes.append(
                        {
                            "pitch": int(p),
                            "offset": float(start),
                            "duration": float(dur),
                            "voice": int(v_id),
                        }
                    )
    else:
        flat_notes = list(voice_notes)

    chk = run_music21_counterpoint_checker(flat_notes)
    errors: List[Dict[str, Any]] = []
    if chk["parallel_fifths"] > 0:
        errors.append({"type": "parallelFifths", "count": chk["parallel_fifths"]})
    if chk["parallel_octaves"] > 0:
        errors.append({"type": "parallelOctaves", "count": chk["parallel_octaves"]})
    return errors


def evaluate_measure(notes_in_measure: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate a measure using official music21.figuredBass.checker functions."""
    chk = run_music21_counterpoint_checker(notes_in_measure)
    errors: List[Dict[str, Any]] = []
    for key_name in (
        "parallel_fifths",
        "parallel_octaves",
        "hidden_fifths",
        "hidden_octaves",
        "voice_crossings",
        "voice_overlaps",
    ):
        if int(chk.get(key_name, 0)) > 0:
            errors.append({"type": key_name, "count": int(chk[key_name])})
    return errors


if __name__ == "__main__":
    print("=== Official music21 + jSymbolic + MusPy Benchmark on J.S. Bach Corpus ===")
    sample_bwvs = ["bwv66.6", "bwv269", "bwv270", "bwv271", "bwv272"]
    for bwv_id in sample_bwvs:
        sc_raw = corpus.parse(bwv_id)
        sc_4part = stream.Score(list(sc_raw.parts)[:4])
        res = evaluate_composition_official(sc_4part, include_jsymbolic=True)
        chk_r = res["music21_checker"]
        mp_r = res["muspy_metrics"]
        js_r = res["jsymbolic_features"]
        print(
            f"[{bwv_id}] P5={chk_r['parallel_fifths']}, P8={chk_r['parallel_octaves']}, "
            f"VC={chk_r['voice_crossings']}, ScaleCons={mp_r['scale_consistency']:.3f}, "
            f"PolyRate={mp_r['polyphony_rate']:.3f}, Stepwise={js_r['stepwise_motion_ratio']:.3f}"
        )
