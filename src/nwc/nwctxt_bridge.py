"""Bidirectional bridge between NoteWorthy Composer 2 (NWCTXT) and Like_Bach notes."""

from __future__ import annotations

import re
from typing import Any

STEP_TO_INDEX: dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}
INDEX_TO_STEP: dict[int, str] = {v: k for k, v in STEP_TO_INDEX.items()}

STEP_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

# Center line (Pos:0) absolute diatonic step (octave * 7 + step_index)
CLEF_CENTER_ABS_STEP: dict[str, int] = {
    "TREBLE": 34,      # B4
    "TREBLE8DW": 27,   # B3
    "TREBLE8UP": 41,   # B5
    "BASS": 22,        # D3
    "BASS8DW": 15,     # D2
    "BASS8UP": 29,     # D4
    "ALTO": 28,        # C4
    "TENOR": 26,       # A3
    "PERCUSSION": 22,  # D3
}

NWC_BASE_DURATIONS: dict[str, float] = {
    "Whole": 4.0,
    "Half": 2.0,
    "4th": 1.0,
    "8th": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
}

# Ordered list of (quarter_length, nwc_dur_token, is_beamable) for decomposition
DURATION_TABLE: list[tuple[float, str, bool]] = [
    (4.0, "Whole", False),
    (3.0, "Half,Dotted", False),
    (2.0, "Half", False),
    (1.5, "4th,Dotted", False),
    (1.0, "4th", False),
    (0.75, "8th,Dotted", True),
    (0.5, "8th", True),
    (0.375, "16th,Dotted", True),
    (0.25, "16th", True),
    (0.125, "32nd", True),
]

ALTER_TO_PREFIX: dict[int, str] = {
    -2: "v",
    -1: "b",
    0: "n",
    1: "#",
    2: "x",
}

PREFIX_TO_ALTER: dict[str, int] = {
    "v": -2,
    "b": -1,
    "n": 0,
    "#": 1,
    "x": 2,
}

SHARP_PITCH_MAP: dict[int, tuple[str, int]] = {
    0: ("C", 0),
    1: ("C", 1),
    2: ("D", 0),
    3: ("D", 1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("F", 1),
    7: ("G", 0),
    8: ("G", 1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}

FLAT_PITCH_MAP: dict[int, tuple[str, int]] = {
    0: ("C", 0),
    1: ("D", -1),
    2: ("D", 0),
    3: ("E", -1),
    4: ("E", 0),
    5: ("F", 0),
    6: ("G", -1),
    7: ("G", 0),
    8: ("A", -1),
    9: ("A", 0),
    10: ("B", -1),
    11: ("B", 0),
}


def detect_nwctxt_mode(raw_text: str) -> str:
    """Detect whether the input from NWC2 is FileText or ClipText."""
    for line in raw_text.splitlines():
        s = line.strip()
        if s.startswith("#/ReturnFormat:"):
            fmt = s.split(":", 1)[1].strip()
            if "ClipText" in fmt:
                return "ClipText"
            if "FileText" in fmt:
                return "FileText"
        if s.startswith("!NoteWorthyComposerClip"):
            return "ClipText"
        if s.startswith("!NoteWorthyComposer("):
            return "FileText"
    return "FileText"


def parse_key_signature(sig_str: str) -> dict[str, int]:
    """Parse NWC Key Signature attribute (e.g. 'F#,C#' or 'Bb,Eb') into step->alter dict."""
    key_map: dict[str, int] = {}
    if not sig_str or sig_str.strip().upper() == "C":
        return key_map
    for item in sig_str.split(","):
        tok = item.strip()
        if len(tok) >= 2:
            step = tok[0].upper()
            acc = tok[1:]
            if step in STEP_TO_INDEX and acc in PREFIX_TO_ALTER:
                key_map[step] = PREFIX_TO_ALTER[acc]
    return key_map


def parse_time_signature(sig_str: str) -> float:
    """Parse NWC TimeSig attribute into beats (quarter lengths) per measure."""
    s = sig_str.strip()
    if s.lower() == "common":
        return 4.0
    if s.lower() == "allabreve":
        return 4.0
    if "/" in s:
        try:
            num_s, den_s = s.split("/", 1)
            num = int(num_s.strip())
            den = int(den_s.strip())
            if den > 0:
                return float(num) * (4.0 / float(den))
        except ValueError:
            pass
    return 4.0


def parse_nwc_duration(dur_spec: str) -> float:
    """Convert NWC Dur field (e.g. '4th,Dotted' or '8th,Triplet=First') to quarterLength."""
    parts = [p.strip() for p in dur_spec.split(",") if p.strip()]
    if not parts:
        return 1.0
    base_ql = NWC_BASE_DURATIONS.get(parts[0], 1.0)
    dots = 0
    is_triplet = False
    is_grace = False
    for p in parts[1:]:
        if p == "Dotted":
            dots = 1
        elif p == "DblDotted":
            dots = 2
        elif p.startswith("Triplet"):
            is_triplet = True
        elif p == "Grace":
            is_grace = True
    if is_grace:
        return 0.0
    ql = base_ql
    if dots == 1:
        ql *= 1.5
    elif dots == 2:
        ql *= 1.75
    if is_triplet:
        ql = ql * 2.0 / 3.0
    return round(ql, 4)


def normalize_clef_key(clef_type: str, octave_shift: str | None = None) -> str:
    """Normalize NWC Clef Type and OctaveShift into CLEF_CENTER_ABS_STEP key."""
    base = (clef_type or "Treble").strip().upper()
    if base not in ("TREBLE", "BASS", "ALTO", "TENOR", "PERCUSSION"):
        base = "TREBLE"
    shift = (octave_shift or "").strip().upper()
    if base in ("TREBLE", "BASS"):
        if "DOWN" in shift:
            return f"{base}8DW"
        if "UP" in shift:
            return f"{base}8UP"
    return base


def nwc_pos_to_midi(
    pos_token: str,
    clef_key: str,
    key_sig_map: dict[str, int],
    active_accidentals: dict[tuple[str, int], int],
) -> tuple[int, bool]:
    """
    Convert a single NWC Pos token (e.g. '#-3^', 'b2', '0') into (midi_pitch, is_tied).
    Updates active_accidentals in place for the current measure.
    """
    s = pos_token.strip()
    is_tied = "^" in s
    # Strip trailing notehead/tie modifiers
    s = re.sub(r"[\^xXzZ]+$", "", s)

    acc_char = ""
    if s and s[0] in PREFIX_TO_ALTER:
        acc_char = s[0]
        s = s[1:]

    m = re.search(r"-?\d+", s)
    pos_int = int(m.group(0)) if m else 0

    center_abs = CLEF_CENTER_ABS_STEP.get(clef_key, 34)
    abs_step = center_abs + pos_int
    octave = abs_step // 7
    step_idx = abs_step % 7
    step_name = INDEX_TO_STEP[step_idx]

    note_id = (step_name, octave)
    if acc_char:
        alter = PREFIX_TO_ALTER[acc_char]
        active_accidentals[note_id] = alter
    elif note_id in active_accidentals:
        alter = active_accidentals[note_id]
    else:
        alter = key_sig_map.get(step_name, 0)

    midi_pitch = (octave + 1) * 12 + STEP_TO_SEMITONE[step_name] + alter
    midi_pitch = max(0, min(127, midi_pitch))
    return midi_pitch, is_tied


def _parse_pipe_line(line: str) -> tuple[str, dict[str, str]]:
    """Parse an NWC pipe command line '|Cmd|Key:Val|...' into (cmd, attrs)."""
    parts = line.strip().split("|")
    if len(parts) < 2:
        return "", {}
    cmd = parts[1].strip()
    attrs: dict[str, str] = {}
    for seg in parts[2:]:
        if ":" in seg:
            k, v = seg.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k not in attrs:
                attrs[k] = v
    return cmd, attrs


def parse_nwctxt(raw_text: str) -> dict[str, Any]:
    """
    Parse an NWC2 FileText or ClipText string.
    Extracts header lines, staff metadata, musical context, and subject_notes from the primary staff.
    """
    mode = detect_nwctxt_mode(raw_text)
    lines = raw_text.splitlines()

    header_lines: list[str] = []
    version_header = "!NoteWorthyComposer(2.75)"
    clip_header = "!NoteWorthyComposerClip(2.75,Single)"

    staves: list[dict[str, Any]] = []
    current_staff: dict[str, Any] | None = None

    in_body = False
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#/"):
            continue
        if s.startswith("!NoteWorthyComposerClip("):
            clip_header = s
            in_body = True
            current_staff = {
                "name": "Clip",
                "lines": [],
                "header_lines": [],
            }
            staves.append(current_staff)
            continue
        if s.startswith("!NoteWorthyComposer("):
            version_header = s
            in_body = True
            continue
        if s.startswith("!NoteWorthyComposer"):
            # End marker
            break
        if not in_body:
            continue

        if s.startswith("|AddStaff"):
            _, attrs = _parse_pipe_line(s)
            staff_name = attrs.get("Name", f'"Staff {len(staves) + 1}"').strip('"')
            current_staff = {
                "name": staff_name,
                "add_staff_line": s,
                "lines": [],
                "header_lines": [s],
            }
            staves.append(current_staff)
        elif current_staff is None:
            header_lines.append(s)
        else:
            if s.startswith("|StaffProperties") or s.startswith("|StaffInstrument"):
                current_staff["header_lines"].append(s)
            else:
                current_staff["lines"].append(s)

    # Extract notes from each staff to find the primary melody/subject staff
    parsed_staves: list[dict[str, Any]] = []
    for idx, st in enumerate(staves):
        staff_info = _extract_notes_from_staff_lines(st["lines"], voice_num=idx + 1)
        staff_info["name"] = st["name"]
        staff_info["header_lines"] = st.get("header_lines", [])
        staff_info["raw_lines"] = st["lines"]
        parsed_staves.append(staff_info)

    # Choose the first staff that contains at least one note
    primary_staff = None
    for pst in parsed_staves:
        if pst["notes"]:
            primary_staff = pst
            break
    if primary_staff is None and parsed_staves:
        primary_staff = parsed_staves[0]

    subject_notes: list[dict[str, Any]] = []
    clef_type = "Treble"
    key_sig_str = ""
    key_tonic = "C"
    time_sig_str = "4/4"
    tempo_line = None

    if primary_staff is not None:
        for n in primary_staff["notes"]:
            subject_notes.append({
                "pitch": int(n["pitch"]),
                "duration": float(n["duration"]),
                "offset": float(n["offset"]),
                "voice": 1,
            })
        clef_type = primary_staff["clef_type"]
        key_sig_str = primary_staff["key_sig_str"]
        key_tonic = primary_staff["key_tonic"]
        time_sig_str = primary_staff["time_sig_str"]
        tempo_line = primary_staff["tempo_line"]

    return {
        "mode": mode,
        "version_header": version_header,
        "clip_header": clip_header,
        "header_lines": header_lines,
        "staves": parsed_staves,
        "subject_notes": subject_notes,
        "clef_type": clef_type,
        "key_sig_str": key_sig_str,
        "key_tonic": key_tonic,
        "time_sig_str": time_sig_str,
        "tempo_line": tempo_line,
    }


def _extract_notes_from_staff_lines(staff_lines: list[str], voice_num: int = 1) -> dict[str, Any]:
    """Parse staff lines into sequential notes with offsets, durations, and MIDI pitches."""
    clef_type = "Treble"
    clef_octave = None
    clef_key = "TREBLE"
    key_sig_str = ""
    key_tonic = "C"
    key_sig_map: dict[str, int] = {}
    time_sig_str = "4/4"
    beats_per_bar = 4.0
    tempo_line: str | None = None

    active_accidentals: dict[tuple[str, int], int] = {}
    notes: list[dict[str, Any]] = []
    tied_pending: dict[int, dict[str, Any]] = {}
    current_offset = 0.0

    for line in staff_lines:
        if not line.startswith("|"):
            continue
        cmd, attrs = _parse_pipe_line(line)
        if cmd == "Clef":
            clef_type = attrs.get("Type", "Treble")
            clef_octave = attrs.get("OctaveShift")
            clef_key = normalize_clef_key(clef_type, clef_octave)
        elif cmd == "Key":
            key_sig_str = attrs.get("Signature", "")
            key_tonic = attrs.get("Tonic", "C")
            key_sig_map = parse_key_signature(key_sig_str)
        elif cmd == "TimeSig":
            time_sig_str = attrs.get("Signature", "4/4")
            beats_per_bar = parse_time_signature(time_sig_str)
        elif cmd == "Tempo":
            if tempo_line is None:
                tempo_line = line
        elif cmd == "Bar":
            active_accidentals.clear()
        elif cmd == "Rest":
            dur = parse_nwc_duration(attrs.get("Dur", "4th"))
            current_offset = round(current_offset + dur, 4)
        elif cmd == "RestMultiBar":
            try:
                num_bars = int(attrs.get("NumBars", "1"))
            except ValueError:
                num_bars = 1
            current_offset = round(current_offset + num_bars * beats_per_bar, 4)
        elif cmd in ("Note", "Chord", "RestChord"):
            dur = parse_nwc_duration(attrs.get("Dur", "4th"))
            if dur <= 0.0:
                continue
            pos_field = attrs.get("Pos", attrs.get("Pos2", "0"))
            pos_tokens = [p.strip() for p in pos_field.split(",") if p.strip()]
            if not pos_tokens:
                current_offset = round(current_offset + dur, 4)
                continue

            pitches_and_ties = [
                nwc_pos_to_midi(pt, clef_key, key_sig_map, active_accidentals)
                for pt in pos_tokens
            ]
            # Select top note for melody extraction
            top_pitch, is_tied = max(pitches_and_ties, key=lambda x: x[0])

            if top_pitch in tied_pending:
                prev_note = tied_pending.pop(top_pitch)
                prev_note["duration"] = round(prev_note["duration"] + dur, 4)
                if is_tied:
                    tied_pending[top_pitch] = prev_note
            else:
                new_note = {
                    "pitch": top_pitch,
                    "duration": round(dur, 4),
                    "offset": round(current_offset, 4),
                    "voice": voice_num,
                }
                notes.append(new_note)
                if is_tied:
                    tied_pending[top_pitch] = new_note

            current_offset = round(current_offset + dur, 4)

    return {
        "notes": notes,
        "clef_type": clef_type,
        "key_sig_str": key_sig_str,
        "key_tonic": key_tonic,
        "time_sig_str": time_sig_str,
        "tempo_line": tempo_line,
    }


def midi_to_nwc_pos(
    midi_pitch: int,
    clef_key: str,
    key_sig_map: dict[str, int],
    active_accidentals: dict[tuple[str, int], int],
) -> str:
    """Convert a MIDI pitch to an NWC Pos string (e.g. '-6', '#-3', 'n0') under the current Clef and Key."""
    midi_pitch = max(12, min(108, int(midi_pitch)))
    octave = (midi_pitch // 12) - 1
    pc = midi_pitch % 12

    # First check if any diatonic step in the current key signature naturally produces this pitch class
    chosen_step: str | None = None
    chosen_alter: int = 0
    chosen_octave = octave

    for step_name, base_semi in STEP_TO_SEMITONE.items():
        key_alt = key_sig_map.get(step_name, 0)
        natural_pc = (base_semi + key_alt) % 12
        if natural_pc == pc:
            chosen_step = step_name
            chosen_alter = key_alt
            # Handle wrap-around for B# or Cb
            raw_semi = base_semi + key_alt
            if raw_semi < 0:
                chosen_octave = octave + 1
            elif raw_semi >= 12:
                chosen_octave = octave - 1
            break

    if chosen_step is None:
        is_flat_key = any(v < 0 for v in key_sig_map.values())
        table = FLAT_PITCH_MAP if is_flat_key else SHARP_PITCH_MAP
        chosen_step, chosen_alter = table[pc]

    abs_step = chosen_octave * 7 + STEP_TO_INDEX[chosen_step]
    center_abs = CLEF_CENTER_ABS_STEP.get(clef_key, 34)
    pos_int = abs_step - center_abs

    note_id = (chosen_step, chosen_octave)
    current_effective_alter = active_accidentals.get(note_id, key_sig_map.get(chosen_step, 0))

    prefix = ""
    if chosen_alter != current_effective_alter:
        prefix = ALTER_TO_PREFIX.get(chosen_alter, "")
        active_accidentals[note_id] = chosen_alter

    return f"{prefix}{pos_int}"


def decompose_duration_to_nwc(duration_ql: float) -> list[tuple[float, str, bool]]:
    """
    Decompose a quarterLength duration into a sequence of (ql, nwc_dur_str, is_beamable) tuples.
    Guarantees exact measure alignment.
    """
    rem = round(float(duration_ql), 4)
    result: list[tuple[float, str, bool]] = []
    if rem <= 0.05:
        return result

    # Quantize to nearest 0.125 (32nd note)
    rem = round(rem * 8.0) / 8.0
    if rem <= 0.0:
        rem = 0.125

    while rem >= 0.125 - 1e-4:
        matched = False
        for ql_val, dur_token, beamable in DURATION_TABLE:
            if rem >= ql_val - 1e-4:
                result.append((ql_val, dur_token, beamable))
                rem = round(rem - ql_val, 4)
                matched = True
                break
        if not matched:
            break

    return result


def render_voice_to_nwctxt_lines(
    voice_notes: list[dict[str, Any]],
    total_measures: int,
    clef_type: str = "Treble",
    octave_shift: str | None = None,
    key_sig_str: str = "",
    key_tonic: str = "C",
    time_sig_str: str = "4/4",
    tempo_line: str | None = None,
) -> list[str]:
    """Render a single voice's note list into a complete sequence of NWC2 notation lines."""
    lines: list[str] = []
    if octave_shift:
        lines.append(f"|Clef|Type:{clef_type}|OctaveShift:{octave_shift}")
    else:
        lines.append(f"|Clef|Type:{clef_type}")

    if key_sig_str:
        lines.append(f"|Key|Signature:{key_sig_str}|Tonic:{key_tonic}")
    else:
        lines.append(f"|Key|Signature:C|Tonic:{key_tonic}")

    if tempo_line:
        lines.append(tempo_line)

    lines.append(f"|TimeSig|Signature:{time_sig_str}")

    clef_key = normalize_clef_key(clef_type, octave_shift)
    key_sig_map = parse_key_signature(key_sig_str)
    beats_per_bar = parse_time_signature(time_sig_str)

    # Sort and deduplicate notes by offset
    sorted_notes = sorted(voice_notes, key=lambda n: float(n["offset"]))
    cleaned_notes: list[dict[str, Any]] = []
    for n in sorted_notes:
        off = round(float(n["offset"]), 4)
        dur = round(float(n["duration"]), 4)
        if dur <= 0.05:
            continue
        if cleaned_notes:
            prev = cleaned_notes[-1]
            prev_end = round(prev["offset"] + prev["duration"], 4)
            if off < prev_end - 1e-3:
                # Trim previous note to prevent overlap within a single monophonic staff
                new_prev_dur = round(off - prev["offset"], 4)
                if new_prev_dur >= 0.125:
                    prev["duration"] = new_prev_dur
                else:
                    cleaned_notes.pop()
        cleaned_notes.append({
            "pitch": int(n["pitch"]),
            "offset": off,
            "duration": dur,
        })

    # Split notes across barlines and fill rests per measure
    max_beat = total_measures * beats_per_bar
    events_by_bar: dict[int, list[dict[str, Any]]] = {m: [] for m in range(total_measures)}

    cursor = 0.0
    note_idx = 0
    num_notes = len(cleaned_notes)

    while cursor < max_beat - 1e-3:
        bar_idx = int(cursor // beats_per_bar)
        if bar_idx >= total_measures:
            break
        bar_end = (bar_idx + 1) * beats_per_bar

        # Skip any notes that end before cursor
        while note_idx < num_notes and round(cleaned_notes[note_idx]["offset"] + cleaned_notes[note_idx]["duration"], 4) <= cursor + 1e-3:
            note_idx += 1

        if note_idx < num_notes:
            curr_n = cleaned_notes[note_idx]
            n_start = max(cursor, curr_n["offset"])
            if n_start > cursor + 1e-3:
                # Fill gap before next note within current measure
                gap_end = min(n_start, bar_end)
                gap_dur = round(gap_end - cursor, 4)
                if gap_dur >= 0.125:
                    events_by_bar[bar_idx].append({
                        "type": "rest",
                        "duration": gap_dur,
                        "offset_in_bar": round(cursor - bar_idx * beats_per_bar, 4),
                    })
                cursor = gap_end
                continue

            # We are at the start (or continuation) of curr_n
            n_end = round(curr_n["offset"] + curr_n["duration"], 4)
            seg_end = min(n_end, bar_end)
            seg_dur = round(seg_end - cursor, 4)
            tied_to_next_bar = n_end > bar_end + 1e-3

            if seg_dur >= 0.125:
                events_by_bar[bar_idx].append({
                    "type": "note",
                    "pitch": curr_n["pitch"],
                    "duration": seg_dur,
                    "tied_out": tied_to_next_bar,
                    "offset_in_bar": round(cursor - bar_idx * beats_per_bar, 4),
                })
            cursor = seg_end
            if cursor >= n_end - 1e-3:
                note_idx += 1
        else:
            # Fill remaining measure with rest
            rem_dur = round(bar_end - cursor, 4)
            if rem_dur >= 0.125:
                events_by_bar[bar_idx].append({
                    "type": "rest",
                    "duration": rem_dur,
                    "offset_in_bar": round(cursor - bar_idx * beats_per_bar, 4),
                })
            cursor = bar_end

    # Convert bar events to NWC lines with beaming and accidentals
    for bar_idx in range(total_measures):
        active_accidentals: dict[tuple[str, int], int] = {}
        bar_tokens: list[dict[str, Any]] = []
        running_beat = 0.0

        for ev in events_by_bar.get(bar_idx, []):
            parts = decompose_duration_to_nwc(ev["duration"])
            for p_idx, (ql_val, dur_str, beamable) in enumerate(parts):
                is_last_part = p_idx == len(parts) - 1
                if ev["type"] == "rest":
                    bar_tokens.append({
                        "kind": "Rest",
                        "dur_str": dur_str,
                        "pos_str": "",
                        "beamable": False,
                        "beat_group": int(running_beat),
                    })
                else:
                    pos_str = midi_to_nwc_pos(
                        ev["pitch"],
                        clef_key,
                        key_sig_map,
                        active_accidentals,
                    )
                    need_tie = (not is_last_part) or bool(ev.get("tied_out", False))
                    if need_tie:
                        pos_str = f"{pos_str}^"
                    bar_tokens.append({
                        "kind": "Note",
                        "dur_str": dur_str,
                        "pos_str": pos_str,
                        "beamable": beamable,
                        "beat_group": int(running_beat),
                    })
                running_beat = round(running_beat + ql_val, 4)

        # Assign beam options for consecutive 8th/16th notes within the same beat group
        i = 0
        while i < len(bar_tokens):
            if bar_tokens[i]["beamable"]:
                grp = bar_tokens[i]["beat_group"]
                j = i
                while j + 1 < len(bar_tokens) and bar_tokens[j + 1]["beamable"] and bar_tokens[j + 1]["beat_group"] == grp:
                    j += 1
                if j > i:
                    bar_tokens[i]["beam_opt"] = "Beam=First"
                    for k in range(i + 1, j):
                        bar_tokens[k]["beam_opt"] = "Beam"
                    bar_tokens[j]["beam_opt"] = "Beam=End"
                i = j + 1
            else:
                i += 1

        for tok in bar_tokens:
            if tok["kind"] == "Rest":
                lines.append(f"|Rest|Dur:{tok['dur_str']}")
            else:
                beam_opt = tok.get("beam_opt")
                if beam_opt:
                    lines.append(f"|Note|Dur:{tok['dur_str']}|Pos:{tok['pos_str']}|Opts:{beam_opt}")
                else:
                    lines.append(f"|Note|Dur:{tok['dur_str']}|Pos:{tok['pos_str']}")

        if bar_idx < total_measures - 1:
            lines.append("|Bar")

    return lines


def build_full_score_nwctxt(
    parsed_doc: dict[str, Any],
    generated_notes: list[dict[str, Any]],
    mode: str = "chorale",
    target_measures: int = 8,
) -> str:
    """
    Build a complete 4-voice NWC2 FileText score (!NoteWorthyComposer(2.75))
    from the parsed document metadata and AI-generated 4-voice notes.
    """
    out_lines: list[str] = [parsed_doc.get("version_header", "!NoteWorthyComposer(2.75)")]

    header_lines = parsed_doc.get("header_lines", [])
    if header_lines:
        out_lines.extend(header_lines)
    else:
        out_lines.extend([
            "|Editor|ActiveStaff:1|CaretIndex:1|CaretPos:0",
            f'|SongInfo|Title:"Like Bach ({mode.capitalize()})"|Author:"J.S. Bach / Like_Bach AI"',
            "|PgSetup|StaffSize:16|Zoom:4|TitlePage:Y|JustifyVertically:Y|ExtendLastSystem:N|DurationPadding:Y|PageNumbers:1|StaffLabels:First System|BarNumbers:Boxed|StartingBar:1|AllowLayering:Y",
            "|PgMargins|Left:1.78|Top:1.78|Right:1.78|Bottom:1.78|Mirror:N",
        ])

    key_sig_str = parsed_doc.get("key_sig_str", "")
    key_tonic = parsed_doc.get("key_tonic", "C")
    time_sig_str = parsed_doc.get("time_sig_str", "4/4")
    tempo_line = parsed_doc.get("tempo_line") or '|Tempo|Tempo:100|Pos:10|Placement:BestFitForward'

    beats_per_bar = parse_time_signature(time_sig_str)
    if generated_notes:
        max_end = max(float(n["offset"]) + float(n["duration"]) for n in generated_notes)
        inferred_measures = max(1, int((max_end + beats_per_bar - 0.05) // beats_per_bar))
        total_measures = max(target_measures, inferred_measures)
    else:
        total_measures = max(1, target_measures)

    voice_configs = [
        {
            "voice": 1,
            "name": "Soprano",
            "label": "Soprano",
            "abbr": "S.",
            "clef": "Treble",
            "octave_shift": None,
            "channel": 1,
            "patch": 0,
            "with_next": "|WithNextStaff:Bracket,ConnectBars",
        },
        {
            "voice": 2,
            "name": "Alto",
            "label": "Alto",
            "abbr": "A.",
            "clef": "Treble",
            "octave_shift": None,
            "channel": 2,
            "patch": 0,
            "with_next": "|WithNextStaff:Bracket,ConnectBars",
        },
        {
            "voice": 3,
            "name": "Tenor",
            "label": "Tenor",
            "abbr": "T.",
            "clef": "Bass",
            "octave_shift": None,
            "channel": 3,
            "patch": 0,
            "with_next": "|WithNextStaff:Bracket,ConnectBars",
        },
        {
            "voice": 4,
            "name": "Bass",
            "label": "Bass",
            "abbr": "B.",
            "clef": "Bass",
            "octave_shift": None,
            "channel": 4,
            "patch": 0,
            "with_next": "",
        },
    ]

    # Inherit instrument patch from original primary staff if present
    orig_staves = parsed_doc.get("staves", [])
    inherited_patch = 19 if mode.lower() == "fugue" else 0  # Church Organ (19) for Fugue, Piano (0) for Chorale
    if orig_staves:
        for hl in orig_staves[0].get("header_lines", []):
            if hl.startswith("|StaffInstrument"):
                _, attrs = _parse_pipe_line(hl)
                if "Patch" in attrs:
                    try:
                        inherited_patch = int(attrs["Patch"])
                    except ValueError:
                        pass

    for cfg in voice_configs:
        v_num = cfg["voice"]
        v_notes = [n for n in generated_notes if int(n.get("voice", 1)) == v_num]

        out_lines.append(
            f'|AddStaff|Name:"{cfg["name"]}"|Label:"{cfg["label"]}"|LabelAbbr:"{cfg["abbr"]}"|Group:"Standard"'
        )
        out_lines.append(
            f'|StaffProperties|EndingBar:Section Close|Visible:Y|BoundaryTop:14|BoundaryBottom:13|Lines:5{cfg["with_next"]}|Color:Default'
        )
        out_lines.append(
            f'|StaffProperties|Muted:N|Volume:127|StereoPan:64|Device:0|Channel:{cfg["channel"]}'
        )
        out_lines.append(
            f'|StaffInstrument|Patch:{inherited_patch}|Trans:0|DynVel:10,30,45,60,75,92,108,127'
        )

        staff_notation = render_voice_to_nwctxt_lines(
            voice_notes=v_notes,
            total_measures=total_measures,
            clef_type=cfg["clef"],
            octave_shift=cfg["octave_shift"],
            key_sig_str=key_sig_str,
            key_tonic=key_tonic,
            time_sig_str=time_sig_str,
            tempo_line=tempo_line if v_num == 1 else None,
        )
        out_lines.extend(staff_notation)

    out_lines.append("!NoteWorthyComposer-End")
    return "\n".join(out_lines) + "\n"


def build_clip_nwctxt(
    parsed_doc: dict[str, Any],
    generated_notes: list[dict[str, Any]],
    mode: str = "chorale",
    target_measures: int = 8,
    clip_style: str = "chord",
) -> str:
    """
    Build an NWC2 ClipText string (!NoteWorthyComposerClip(2.75,Single))
    for replacing a selected clip in NWC2.
    If clip_style == 'chord', merges simultaneous notes across voices into NWC |Chord| items.
    If clip_style == 'melody', outputs Voice 1 (Soprano) extended melody.
    """
    clip_header = parsed_doc.get("clip_header", "!NoteWorthyComposerClip(2.75,Single)")
    out_lines: list[str] = [clip_header]

    clef_type = parsed_doc.get("clef_type", "Treble")
    key_sig_str = parsed_doc.get("key_sig_str", "")
    key_tonic = parsed_doc.get("key_tonic", "C")
    time_sig_str = parsed_doc.get("time_sig_str", "4/4")
    beats_per_bar = parse_time_signature(time_sig_str)

    if generated_notes:
        max_end = max(float(n["offset"]) + float(n["duration"]) for n in generated_notes)
        inferred_measures = max(1, int((max_end + beats_per_bar - 0.05) // beats_per_bar))
        total_measures = max(target_measures, inferred_measures)
    else:
        total_measures = max(1, target_measures)

    if clip_style == "melody":
        v1_notes = [n for n in generated_notes if int(n.get("voice", 1)) == 1]
        body_lines = render_voice_to_nwctxt_lines(
            voice_notes=v1_notes,
            total_measures=total_measures,
            clef_type=clef_type,
            key_sig_str=key_sig_str,
            key_tonic=key_tonic,
            time_sig_str=time_sig_str,
            tempo_line=None,
        )
        # Skip redundant header lines inside clip if already placed on staff
        for ln in body_lines:
            if ln.startswith("|Clef") or ln.startswith("|Key") or ln.startswith("|TimeSig"):
                continue
            out_lines.append(ln)
    else:
        # Merge simultaneous notes into chords on a single staff (Treble clef)
        clef_key = normalize_clef_key(clef_type)
        key_sig_map = parse_key_signature(key_sig_str)

        by_offset: dict[float, list[dict[str, Any]]] = {}
        for n in generated_notes:
            off = round(float(n["offset"]), 4)
            by_offset.setdefault(off, []).append(n)

        sorted_offsets = sorted(by_offset.keys())
        max_beat = total_measures * beats_per_bar

        for bar_idx in range(total_measures):
            active_accidentals: dict[tuple[str, int], int] = {}
            bar_start = bar_idx * beats_per_bar
            bar_end = (bar_idx + 1) * beats_per_bar
            cursor = bar_start

            bar_offs = [o for o in sorted_offsets if bar_start - 1e-3 <= o < bar_end - 1e-3]
            for idx_o, off in enumerate(bar_offs):
                if off > cursor + 1e-3:
                    gap = round(off - cursor, 4)
                    for _, dur_str, _ in decompose_duration_to_nwc(gap):
                        out_lines.append(f"|Rest|Dur:{dur_str}")
                    cursor = off

                next_off = bar_offs[idx_o + 1] if idx_o + 1 < len(bar_offs) else bar_end
                chord_notes = by_offset[off]
                min_dur = min(float(cn["duration"]) for cn in chord_notes)
                seg_dur = round(min(min_dur, next_off - off, bar_end - off), 4)
                if seg_dur < 0.125:
                    continue

                pitches = sorted({int(cn["pitch"]) for cn in chord_notes if int(cn["pitch"]) >= 48})
                if not pitches:
                    pitches = sorted({int(cn["pitch"]) for cn in chord_notes})

                parts = decompose_duration_to_nwc(seg_dur)
                for p_idx, (ql_val, dur_str, _) in enumerate(parts):
                    is_last = p_idx == len(parts) - 1
                    pos_list: list[str] = []
                    for mp in pitches:
                        pos_s = midi_to_nwc_pos(mp, clef_key, key_sig_map, active_accidentals)
                        if not is_last:
                            pos_s = f"{pos_s}^"
                        pos_list.append(pos_s)
                    if len(pos_list) == 1:
                        out_lines.append(f"|Note|Dur:{dur_str}|Pos:{pos_list[0]}")
                    else:
                        out_lines.append(f"|Chord|Dur:{dur_str}|Pos:{','.join(pos_list)}")
                    cursor = round(cursor + ql_val, 4)

            if cursor < bar_end - 1e-3 and cursor < max_beat:
                rem = round(bar_end - cursor, 4)
                for _, dur_str, _ in decompose_duration_to_nwc(rem):
                    out_lines.append(f"|Rest|Dur:{dur_str}")

            if bar_idx < total_measures - 1:
                out_lines.append("|Bar")

    out_lines.append("!NoteWorthyComposerClip-End")
    return "\n".join(out_lines) + "\n"
