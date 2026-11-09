import sys
from pathlib import Path

import numpy as np
from music21 import corpus, stream

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v5.evaluator import (  # noqa: E402
    compute_mgeval_oa_kld,
    compute_muspy_official_metrics,
    detect_parallel_intervals,
    evaluate_composition_official,
    evaluate_measure,
    extract_jsymbolic_official_features,
    extract_mgeval_features,
    official_candidate_sort_key,
    run_music21_counterpoint_checker,
)


def test_bach_corpus_bwv66_6_official_evaluation():
    sc_raw = corpus.parse("bwv66.6")
    sc_4part = stream.Score(list(sc_raw.parts)[:4])
    res = evaluate_composition_official(sc_4part, include_jsymbolic=True)

    chk = res["music21_checker"]
    muspy_m = res["muspy_metrics"]
    jsym_m = res["jsymbolic_features"]

    assert chk["harmonic_slices"] > 20
    assert chk["parallel_fifths"] == 0
    assert chk["parallel_octaves"] == 0
    assert 0.50 <= jsym_m["stepwise_motion_ratio"] <= 0.80
    assert 0.85 <= muspy_m["scale_consistency"] <= 1.0
    assert muspy_m["polyphony_rate"] > 0.90


def test_music21_checker_detects_parallel_fifths_and_octaves():
    # Parallel 5ths between Bass (V4: C3=48 -> D3=50) and Tenor (V3: G3=55 -> A3=57)
    # Parallel 8ves between Bass (V4: C3=48 -> D3=50) and Soprano (V1: C5=72 -> D5=74)
    bad_notes = [
        {"pitch": 72, "duration": 1.0, "offset": 0.0, "voice": 1},
        {"pitch": 64, "duration": 1.0, "offset": 0.0, "voice": 2},
        {"pitch": 55, "duration": 1.0, "offset": 0.0, "voice": 3},
        {"pitch": 48, "duration": 1.0, "offset": 0.0, "voice": 4},
        {"pitch": 74, "duration": 1.0, "offset": 1.0, "voice": 1},
        {"pitch": 65, "duration": 1.0, "offset": 1.0, "voice": 2},
        {"pitch": 57, "duration": 1.0, "offset": 1.0, "voice": 3},
        {"pitch": 50, "duration": 1.0, "offset": 1.0, "voice": 4},
    ]

    chk = run_music21_counterpoint_checker(bad_notes)
    assert chk["parallel_fifths"] >= 1
    assert chk["parallel_octaves"] >= 1

    p_errors = detect_parallel_intervals(bad_notes)
    assert len(p_errors) >= 2
    m_errors = evaluate_measure(bad_notes)
    assert any(e["type"] == "parallel_fifths" for e in m_errors)


def test_official_candidate_sort_key_prefers_clean_counterpoint():
    # Clean contrary motion C major -> G major (S: E5->D5, A: G4->G4, T: C4->B3, B: C3->G2)
    clean_notes = [
        {"pitch": 76, "duration": 2.0, "offset": 0.0, "voice": 1},
        {"pitch": 67, "duration": 2.0, "offset": 0.0, "voice": 2},
        {"pitch": 60, "duration": 2.0, "offset": 0.0, "voice": 3},
        {"pitch": 48, "duration": 2.0, "offset": 0.0, "voice": 4},
        {"pitch": 74, "duration": 2.0, "offset": 2.0, "voice": 1},
        {"pitch": 67, "duration": 2.0, "offset": 2.0, "voice": 2},
        {"pitch": 59, "duration": 2.0, "offset": 2.0, "voice": 3},
        {"pitch": 43, "duration": 2.0, "offset": 2.0, "voice": 4},
    ]
    bad_notes = [
        {"pitch": 72, "duration": 2.0, "offset": 0.0, "voice": 1},
        {"pitch": 64, "duration": 2.0, "offset": 0.0, "voice": 2},
        {"pitch": 55, "duration": 2.0, "offset": 0.0, "voice": 3},
        {"pitch": 48, "duration": 2.0, "offset": 0.0, "voice": 4},
        {"pitch": 74, "duration": 2.0, "offset": 2.0, "voice": 1},
        {"pitch": 65, "duration": 2.0, "offset": 2.0, "voice": 2},
        {"pitch": 57, "duration": 2.0, "offset": 2.0, "voice": 3},
        {"pitch": 50, "duration": 2.0, "offset": 2.0, "voice": 4},
    ]

    clean_eval = evaluate_composition_official(clean_notes)
    bad_eval = evaluate_composition_official(bad_notes)

    assert official_candidate_sort_key(clean_eval) < official_candidate_sort_key(
        bad_eval
    )


def test_mgeval_and_muspy_metrics():
    notes = [
        {"pitch": 72, "duration": 1.0, "offset": 0.0, "voice": 1},
        {"pitch": 64, "duration": 1.0, "offset": 0.0, "voice": 2},
        {"pitch": 60, "duration": 1.0, "offset": 0.0, "voice": 3},
        {"pitch": 48, "duration": 1.0, "offset": 0.0, "voice": 4},
        {"pitch": 71, "duration": 1.0, "offset": 1.0, "voice": 1},
        {"pitch": 67, "duration": 1.0, "offset": 1.0, "voice": 2},
        {"pitch": 62, "duration": 1.0, "offset": 1.0, "voice": 3},
        {"pitch": 55, "duration": 1.0, "offset": 1.0, "voice": 4},
    ]
    mp = compute_muspy_official_metrics(notes)
    mg = extract_mgeval_features(notes)
    js = extract_jsymbolic_official_features(notes)

    assert mp["scale_consistency"] == 1.0
    assert mp["polyphony"] == 4.0
    assert mg["total_used_note"] == 8.0
    assert js["stepwise_motion_ratio"] >= 0.0

    rng = np.random.default_rng(42)
    intra = rng.normal(1.0, 0.2, size=50)
    inter = rng.normal(1.05, 0.22, size=50)
    oa_kld = compute_mgeval_oa_kld(intra, inter)
    assert 0.5 <= oa_kld["overlapping_area"] <= 1.0
    assert oa_kld["kld"] >= 0.0
