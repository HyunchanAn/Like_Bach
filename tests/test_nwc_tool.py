"""Unit and integration tests for NoteWorthy Composer 2 (NWC2) User Tool bridge."""

from __future__ import annotations

import io
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nwc.nwc_user_tool import main as nwc_main
from src.nwc.nwctxt_bridge import (
    build_clip_nwctxt,
    build_full_score_nwctxt,
    decompose_duration_to_nwc,
    detect_nwctxt_mode,
    midi_to_nwc_pos,
    nwc_pos_to_midi,
    parse_key_signature,
    parse_nwctxt,
)


def test_detect_nwctxt_mode() -> None:
    """Verify automatic detection of FileText vs ClipText input streams."""
    file_sample = "#/ReturnFormat:\tFileText\n!NoteWorthyComposer(2.75)\n|AddStaff|\n!NoteWorthyComposer-End"
    clip_sample = "!NoteWorthyComposerClip(2.75,Single)\n|Note|Dur:4th|Pos:0\n!NoteWorthyComposerClip-End"

    assert detect_nwctxt_mode(file_sample) == "FileText"
    assert detect_nwctxt_mode(clip_sample) == "ClipText"


def test_pitch_and_pos_roundtrip() -> None:
    """Verify NWC staff position <-> MIDI pitch conversion across Treble and Bass clefs."""
    key_c = parse_key_signature("C")
    acc_treble: dict[tuple[str, int], int] = {}

    # Treble clef: Pos -6 is C4 (MIDI 60), Pos 0 is B4 (MIDI 71), Pos 1 is C5 (MIDI 72)
    p_c4, tied = nwc_pos_to_midi("-6", "TREBLE", key_c, acc_treble)
    assert p_c4 == 60
    assert not tied

    p_b4, _ = nwc_pos_to_midi("0", "TREBLE", key_c, acc_treble)
    assert p_b4 == 71

    p_fsharp4, tied_fs = nwc_pos_to_midi("#-3^", "TREBLE", key_c, acc_treble)
    assert p_fsharp4 == 66
    assert tied_fs

    # Subsequent F4 in same measure inherits # unless natural 'n' is given
    p_fsharp4_inh, _ = nwc_pos_to_midi("-3", "TREBLE", key_c, acc_treble)
    assert p_fsharp4_inh == 66

    p_fnat4, _ = nwc_pos_to_midi("n-3", "TREBLE", key_c, acc_treble)
    assert p_fnat4 == 65

    # Bass clef: Pos -1 is C3 (MIDI 48), Pos 0 is D3 (MIDI 50), Pos 3 is G3 (MIDI 55)
    acc_bass: dict[tuple[str, int], int] = {}
    p_c3, _ = nwc_pos_to_midi("-1", "BASS", key_c, acc_bass)
    assert p_c3 == 48
    p_g3, _ = nwc_pos_to_midi("3", "BASS", key_c, acc_bass)
    assert p_g3 == 55

    # Reverse conversion midi_to_nwc_pos
    acc_out: dict[tuple[str, int], int] = {}
    assert midi_to_nwc_pos(60, "TREBLE", key_c, acc_out) == "-6"
    assert midi_to_nwc_pos(66, "TREBLE", key_c, acc_out) == "#-3"
    # Repeating F#4 in same measure should omit redundant '#'
    assert midi_to_nwc_pos(66, "TREBLE", key_c, acc_out) == "-3"
    # Returning to F natural in same measure requires 'n'
    assert midi_to_nwc_pos(65, "TREBLE", key_c, acc_out) == "n-3"


def test_parse_nwctxt_and_ties() -> None:
    """Verify parsing of notes, rests, ties, and key signature from NWCTXT."""
    sample = "\n".join([
        "!NoteWorthyComposer(2.75)",
        '|AddStaff|Name:"Soprano"',
        "|Clef|Type:Treble",
        "|Key|Signature:F#,C#|Tonic:D",
        "|TimeSig|Signature:4/4",
        "|Note|Dur:4th|Pos:-5",       # D4 (62)
        "|Note|Dur:4th|Pos:-3^",      # F#4 (66) tied
        "|Note|Dur:8th|Pos:-3",       # F#4 (66) +0.5 -> total 1.5
        "|Rest|Dur:8th",              # rest 0.5
        "|Note|Dur:4th|Pos:-1",       # A4 (69)
        "!NoteWorthyComposer-End",
    ])
    parsed = parse_nwctxt(sample)
    notes = parsed["subject_notes"]
    assert len(notes) == 3
    assert notes[0] == {"pitch": 62, "duration": 1.0, "offset": 0.0, "voice": 1}
    assert notes[1] == {"pitch": 66, "duration": 1.5, "offset": 1.0, "voice": 1}
    assert notes[2] == {"pitch": 69, "duration": 1.0, "offset": 3.0, "voice": 1}


def test_duration_decomposition() -> None:
    """Verify compound durations decompose into valid NWC tokens summing to exact duration."""
    parts = decompose_duration_to_nwc(2.5)
    assert sum(p[0] for p in parts) == pytest.approx(2.5)
    assert [p[1] for p in parts] == ["Half", "8th"]


def test_user_tool_cli_full_score_and_clip(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI execution for both FileText (Fugue & Chorale) and ClipText modes."""
    full_score_input = "\n".join([
        "#/File:\tC:\\test.nwc",
        "#/SavePending:\tY",
        "#/ReturnFormat:\tFileText",
        "!NoteWorthyComposer(2.75)",
        '|AddStaff|Name:"Soprano"|Group:"Standard"',
        "|Clef|Type:Treble",
        "|Key|Signature:C|Tonic:C",
        "|TimeSig|Signature:4/4",
        "|Note|Dur:4th|Pos:-6",
        "|Note|Dur:4th|Pos:-5",
        "|Note|Dur:4th|Pos:-4",
        "|Note|Dur:4th|Pos:-2",
        "!NoteWorthyComposer-End",
    ])

    monkeypatch.setattr("sys.stdin", io.StringIO(full_score_input))
    rc = nwc_main(["--mode", "fugue", "--measures", "4"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.startswith("!NoteWorthyComposer(2.75)")
    assert '|AddStaff|Name:"Soprano"' in captured.out
    assert '|AddStaff|Name:"Alto"' in captured.out
    assert '|AddStaff|Name:"Tenor"' in captured.out
    assert '|AddStaff|Name:"Bass"' in captured.out
    assert "!NoteWorthyComposer-End" in captured.out

    # Test ClipText mode
    clip_input = "\n".join([
        "!NoteWorthyComposerClip(2.75,Single)",
        "|Clef|Type:Treble",
        "|Key|Signature:C|Tonic:C",
        "|TimeSig|Signature:4/4",
        "|Note|Dur:Half|Pos:-6",
        "|Note|Dur:Half|Pos:-4",
        "!NoteWorthyComposerClip-End",
    ])
    monkeypatch.setattr("sys.stdin", io.StringIO(clip_input))
    rc_clip = nwc_main(["--mode", "chorale", "--measures", "2", "--clip-style", "chord"])
    captured_clip = capsys.readouterr()
    assert rc_clip == 0
    assert captured_clip.out.startswith("!NoteWorthyComposerClip(2.75,Single)")
    assert "|Chord|" in captured_clip.out
    assert "!NoteWorthyComposerClip-End" in captured_clip.out
