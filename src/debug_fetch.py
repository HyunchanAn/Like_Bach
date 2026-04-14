import requests
from music21 import converter

def debug_fetch():
    url = "https://kern.humdrum.org/cgi-bin/ksdata?location=bach/inventions&format=kern&file=inv1.krn"
    print(f"Fetching {url}...")
    resp = requests.get(url)
    print(f"Status: {resp.status_code}")
    print(f"Content length: {len(resp.text)}")
    print("Preview of content:")
    print(resp.text[:200])
    
    with open('debug_inv1.krn', 'w') as f:
        f.write(resp.text)
        
    print("Parsing with music21...")
    try:
        score = converter.parse('debug_inv1.krn')
        print(f"Score parts: {len(score.parts)}")
        print(f"Score length: {len(score)}")
        for i, part in enumerate(score.parts):
            notes = list(part.recurse().notes)
            print(f"Part {i+1} notes count: {len(notes)}")
    except Exception as e:
        print(f"Error parsing: {e}")

if __name__ == "__main__":
    debug_fetch()
