import music21
from typing import List, Dict

def extract_subject_heuristic(score: music21.stream.Score) -> music21.stream.Part:
    """
    Extracts the subject of a fugue/invention by finding the first entering voice
    and returning its notes up to the entry of the second voice.
    """
    parts = list(score.parts)
    if len(parts) < 2:
        return parts[0] if parts else None
        
    # Find the start offset of the first note for each part
    part_starts = []
    for p in parts:
        notes = p.recurse().notes
        if notes:
            part_starts.append((p, notes[0].offset))
            
    # Sort parts by entry time
    part_starts.sort(key=lambda x: x[1])
    
    first_part = part_starts[0][0]
    second_part_entry = part_starts[1][1] if len(part_starts) > 1 else first_part.highestTime
    
    # Extract notes from the first part up to the second part's entry
    subject_stream = music21.stream.Part()
    for el in first_part.recurse().getElementsByClass(['Note', 'Rest', 'Chord']):
        if el.offset < second_part_entry:
            subject_stream.insert(el.offset, el)
            
    return subject_stream

def generate_hybrid_tokens(score: music21.stream.Score) -> List[str]:
    """
    Experimental function to generate token sequence with [SUBJECT], [ANSWER] markers.
    To be expanded in Phase 2 for fine-tuning.
    """
    # For now, just return a placeholder representing the structure
    tokens = ["[SUBJECT]"]
    # ... extraction logic ...
    tokens.append("[ANSWER]")
    # ... extraction logic ...
    tokens.append("[EPISODE]")
    return tokens

if __name__ == "__main__":
    # Test with a known file if available
    pass
