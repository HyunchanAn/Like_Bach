import os
import urllib.request
import time

def fetch_kernscores():
    base_dir = "data/raw/kernscores"
    os.makedirs(base_dir, exist_ok=True)
    
    # WTC Book 1 & 2 (BWV 846-893)
    # Book 1: wtc1f01.krn to wtc1f24.krn
    # Book 2: wtc2f01.krn to wtc2f24.krn
    
    base_url = "https://kern.humdrum.org/cgi-bin/ksdata?l=users/craig/classical/bach/wtc&format=kern&file="
    
    print("Downloading KernScores WTC Data...")
    
    total = 48
    count = 0
    
    for book in [1, 2]:
        for i in range(1, 25):
            filename = f"wtc{book}f{i:02d}.krn"
            url = base_url + filename
            filepath = os.path.join(base_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Skipping {filename} (already exists)")
                count += 1
                continue
                
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
                count += 1
                time.sleep(0.5) # rate limit
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                
    print(f"Successfully downloaded {count}/{total} files from Stanford KernScores.")

if __name__ == "__main__":
    fetch_kernscores()
