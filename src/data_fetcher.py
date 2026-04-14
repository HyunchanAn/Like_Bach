import os
import requests
from music21 import converter
from tqdm import tqdm

def fetch_bach_inventions_from_kern(output_dir='data/raw/inventions'):
    """
    KernScores에서 바흐의 2성부 인벤션(.krn)을 다운로드하여 MusicXML로 변환 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Inventions: inven01.krn ~ inven15.krn
    # Note: Correct location for inventions used earlier worked (bach/inventions)
    base_url = "https://kern.humdrum.org/cgi-bin/ksdata?l=bach/inventions&format=kern&file=inven"
    
    print("Fetching Bach Inventions from KernScores...")
    found_works = []
    for i in tqdm(range(1, 16)):
        idx = f"{i:02d}"
        file_name = f"inven{idx}.krn"
        url = f"{base_url}{idx}.krn"
        try:
            response = requests.get(url)
            if response.status_code == 200 and len(response.text) > 100: # Some threshold to avoid empty html
                temp_filename = f"temp_{file_name}"
                with open(temp_filename, 'w') as f:
                    f.write(response.text)
                
                work = converter.parse(temp_filename)
                output_path = os.path.join(output_dir, f'inven{idx}.xml')
                work.write('musicxml', fp=output_path)
                found_works.append(output_path)
                os.remove(temp_filename)
            else:
                pass
        except Exception as e:
            pass
            
    print(f"Successfully fetched {len(found_works)} inventions.")
    return found_works

def fetch_wtc_from_kern(book=1, output_dir='data/raw/wtc'):
    """
    KernScores에서 WTC 데이터를 다운로드합니다.
    Location: osu/classical/bach/wtc-1 or wtc-2
    """
    os.makedirs(output_dir, exist_ok=True)
    location = f"osu/classical/bach/wtc-{book}"
    base_url = f"https://kern.humdrum.org/cgi-bin/ksdata?l={location}&format=kern&file=wtc{book}f"
    
    print(f"Fetching Bach WTC Book {book} Fugues from KernScores...")
    found_works = []
    for i in tqdm(range(1, 25)):
        idx = f"{i:02d}"
        url = f"{base_url}{idx}.krn"
        try:
            response = requests.get(url)
            if response.status_code == 200 and len(response.text) > 100:
                temp_filename = f"temp_wtc{book}f{idx}.krn"
                with open(temp_filename, 'w') as f:
                    f.write(response.text)
                
                work = converter.parse(temp_filename)
                output_path = os.path.join(output_dir, f'wtc{book}f{idx}.xml')
                work.write('musicxml', fp=output_path)
                found_works.append(output_path)
                os.remove(temp_filename)
        except Exception as e:
            pass
            
    print(f"Successfully fetched {len(found_works)} WTC Book {book} fugues.")
    return found_works

if __name__ == "__main__":
    fetch_bach_inventions_from_kern()
    fetch_wtc_from_kern(book=1)
    fetch_wtc_from_kern(book=2)
