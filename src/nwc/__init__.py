"""NWC2 (NoteWorthy Composer 2) integration package for Like_Bach."""

from src.nwc.nwctxt_bridge import (
    build_clip_nwctxt,
    build_full_score_nwctxt,
    detect_nwctxt_mode,
    parse_nwctxt,
)

__all__ = [
    "build_clip_nwctxt",
    "build_full_score_nwctxt",
    "detect_nwctxt_mode",
    "parse_nwctxt",
]
