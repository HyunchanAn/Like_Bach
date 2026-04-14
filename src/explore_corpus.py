from music21 import corpus

def explore_bach():
    print("Searching for Bach in music21 corpus...")
    bach_works = corpus.getComposer('bach')
    print(f"Total works found for Bach: {len(bach_works)}")
    
    # Print first 20 to see naming convention
    for work in bach_works[:20]:
        print(work)

if __name__ == "__main__":
    explore_bach()
