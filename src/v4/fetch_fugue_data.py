import os
from music21 import corpus
from tqdm import tqdm

def fetch_fugue_data(output_dir='data/raw/# Fugue_Corpus'):
    """
    Fetch Bach, Handel, and Buxtehude works that are likely fugues, inventions, or polyphonic 
    from music21.corpus. This serves as the Phase 1 dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Searching for Baroque polyphonic works in music21.corpus...")
    
    # We will search for Bach and Handel as primary Baroque representatives in music21 corpus
    # Added Palestrina, Monteverdi, and Josquin to inject strict polyphonic textures
    composers = ['bach', 'handel', 'palestrina', 'monteverdi', 'josquin', 'buxtehude']
    all_works = []
    
    for comp in composers:
        works = corpus.search(comp, 'composer')
        all_works.extend(works)
        print(f"Found {len(works)} potential works for {comp}.")
        
    print(f"Total {len(all_works)} works to filter.")
    
    count = 0
    for work in tqdm(all_works):
        try:
            path = str(work.sourcePath)
            path_lower = path.lower()
            
            # We want inventions, fugues, wtc, sinfonias, and general polyphony (motets, madrigals)
            if any(k in path_lower for k in ['inven', 'wtc', 'fugue', 'sinfonia', 'motet', 'madrigal', 'mass', 'missa', 'fantasia', 'prelude']):
                s = work.parse()
                fname = os.path.basename(path).replace('.mxl', '.xml').replace('.krn', '.xml').replace('.mid', '.xml')
                out_path = os.path.join(output_dir, fname)
                
                if not os.path.exists(out_path):
                    s.write('musicxml', fp=out_path)
                    count += 1
        except Exception as e:
            continue

    print(f"\nFugue data collection complete. Total {count} works saved in {output_dir}")
    return output_dir

if __name__ == "__main__":
    fetch_fugue_data()
