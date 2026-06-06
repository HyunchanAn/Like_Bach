import os
import sys
import pytest
import queue
import base64
from unittest.mock import MagicMock

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_fugue_transposition():
    """
    Test that the Fugue engine correctly schedules the exposition entries.
    V1 (Soprano): Subject
    V2 (Alto): Answer (transposed by -7 semitones, Perfect 5th down)
    V3 (Tenor): Subject (transposed by -12 semitones, Octave down)
    V4 (Bass): Answer (transposed by -19 semitones, Octave + Perfect 5th down)
    """
    from unittest.mock import patch
    
    with patch("builtins.open", MagicMock()), \
         patch("pickle.load", return_value={'stoi': {'[PAD]': 0, '[REST]': 1, 'D4.0': 2}, 'itos': {0: '[PAD]', 1: '[REST]', 2: 'D4.0'}}), \
         patch("torch.load", return_value={}), \
         patch("src.v5.neural_engine.UnifiedTransformerV5") as MockModel, \
         patch("os.path.exists", return_value=True):
        
        # Configure the mock model to return a dummy logit tensor
        mock_instance = MagicMock()
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Return logits of shape (1, 1, vocab_size) where the highest prob is token index 1 ([REST])
        mock_instance.return_value = (torch.tensor([[[0.0, 10.0, 0.0]]], device=device), None)
        mock_instance.to.return_value = mock_instance
        MockModel.return_value = mock_instance
        
        from src.v5.neural_engine import HybridFugueEngine
        engine = HybridFugueEngine()
    
    # 1-measure subject
    subject_notes = [
        {"pitch": 60, "duration": 1.0, "offset": 0.0},
        {"pitch": 62, "duration": 1.0, "offset": 1.0},
        {"pitch": 64, "duration": 2.0, "offset": 2.0}
    ]
    
    # generate_fugue returns notes directly
    notes = engine.generate_fugue(subject_notes, target_measures=4)
    
    assert notes is not None and len(notes) > 0, "Engine did not generate any notes."
        
    v1_notes = [n for n in notes if n["voice"] == 1]
    v2_notes = [n for n in notes if n["voice"] == 2]
    v3_notes = [n for n in notes if n["voice"] == 3]
    v4_notes = [n for n in notes if n["voice"] == 4]
    
    # Voice 1 should have the subject at measure 1 (offset 0)
    assert len(v1_notes) >= 3
    assert v1_notes[0]["pitch"] == 60
    assert v1_notes[1]["pitch"] == 62
    
    # Voice 2 should have the answer at measure 2 (offset 4.0)
    assert len(v2_notes) >= 3
    # First note should be 60 - 7 = 53
    first_v2 = sorted(v2_notes, key=lambda x: x["offset"])[0]
    assert first_v2["pitch"] == 53
    assert first_v2["offset"] == 4.0
    
    # Voice 3 should have the subject at measure 3 (offset 8.0)
    assert len(v3_notes) >= 3
    first_v3 = sorted(v3_notes, key=lambda x: x["offset"])[0]
    assert first_v3["pitch"] == 48 # 60 - 12
    assert first_v3["offset"] == 8.0
    
    # Voice 4 should have the answer at measure 4 (offset 12.0)
    assert len(v4_notes) >= 3
    first_v4 = sorted(v4_notes, key=lambda x: x["offset"])[0]
    assert first_v4["pitch"] == 41 # 60 - 19
    assert first_v4["offset"] == 12.0
