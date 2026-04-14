from music21 import corpus

def search_inventions():
    print("Searching for 'invention' in all corpora...")
    results = corpus.search('invention')
    print(f"Total results: {len(results)}")
    for res in results:
        print(res.sourcePath)

if __name__ == "__main__":
    search_inventions()
