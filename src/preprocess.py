import os
from music21 import converter, note, chord, stream
import pickle

def tokenize_piece(file_path):
    """
    MusicXML 파일을 성부별 토큰으로 변환합니다.
    형식: [VOICE_N] Pitch Duration Offset
    """
    try:
        score = converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    tokens = []
    # 각 성부(Part)를 순회
    parts = score.parts
    if not parts:
        # 성부가 없는 경우 전체 스트림에서 노트 추출 (단성부일 수 있음)
        parts = [score]
        
    print(f"Piece: {os.path.basename(file_path)}, Found {len(parts)} parts.")
    
    for i, part in enumerate(parts):
        voice_tag = f"[VOICE_{i+1}]"
        note_count = 0
        for element in part.recurse():
            if isinstance(element, note.Note):
                # Pitch, QuarterLength Duration, Offset
                tokens.append(f"{voice_tag} P{element.pitch.midi} D{element.duration.quarterLength} O{element.offset}")
                note_count += 1
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    tokens.append(f"{voice_tag} P{n.pitch.midi} D{n.duration.quarterLength} O{element.offset}")
                    note_count += 1
        # print(f"Part {i+1} note count: {note_count}")
    
    return tokens

def preprocess_all(input_dirs, output_file):
    """
    여러 디렉터리 내의 모든 MusicXML 파일을 전처리하여 하나의 토큰 파일로 저장합니다.
    """
    all_tokens_dataset = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            continue
            
        for filename in os.listdir(input_dir):
            if filename.endswith('.xml') or filename.endswith('.mxl'):
                file_path = os.path.join(input_dir, filename)
                print(f"Preprocessing {filename}...")
                tokens = tokenize_piece(file_path)
                if tokens and len(tokens) > 0:
                    all_tokens_dataset.append(tokens)
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_tokens_dataset, f)
    
    print(f"Preprocessed {len(all_tokens_dataset)} pieces total and saved to {output_file}")

if __name__ == "__main__":
    raw_dirs = ['data/raw/inventions', 'data/raw/wtc']
    processed_file = 'data/processed/bach_tokens.pkl'
    
    preprocess_all(raw_dirs, processed_file)
