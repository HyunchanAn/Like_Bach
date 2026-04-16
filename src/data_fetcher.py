import os
from music21 import corpus, converter
from tqdm import tqdm

def fetch_all_bach_works(output_dir='data/raw/bach'):
    """
    music21.corpus에서 바흐의 모든 작품을 검색하고 유효한 데이터(코랄, 인벤션, WTC 등)를 수집합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Searching for Bach works in music21.corpus...")
    all_bach = corpus.search('bach', 'composer')
    print(f"Found total {len(all_bach)} potential works.")
    
    count = 0
    for work in tqdm(all_bach):
        try:
            # Metadata based filtering
            path = str(work.sourcePath)
            path_lower = path.lower()
            
            # Labeling for organization
            label = "other"
            if 'chorale' in path_lower or 'bwv' in path_lower: label = "chorale"
            elif 'inven' in path_lower: label = "inven"
            elif 'wtc' in path_lower or 'fugue' in path_lower: label = "poly"
            
            # Get the actual score
            s = work.parse()
            
            fname = os.path.basename(path).replace('.mxl', '.xml').replace('.krn', '.xml').replace('.mid', '.xml')
            out_path = os.path.join(output_dir, f"{label}_{fname}")
            
            if not os.path.exists(out_path):
                s.write('musicxml', fp=out_path)
                count += 1
        except Exception as e:
            continue

    print(f"\nData collection complete. Total {count} works saved in {output_dir}")
    return output_dir

if __name__ == "__main__":
    fetch_all_bach_works()
