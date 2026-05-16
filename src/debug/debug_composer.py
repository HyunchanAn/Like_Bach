from src.engine import FugueEngine

def debug():
    engine = FugueEngine()
    # Simple subject: C, D, E, F, G (MIDI 60, 62, 64, 65, 67)
    test_data = [
        {"pitch": 60, "duration": 1.0, "offset": 0.0},
        {"pitch": 62, "duration": 1.0, "offset": 1.0},
        {"pitch": 64, "duration": 1.0, "offset": 2.0},
        {"pitch": 65, "duration": 1.0, "offset": 3.0},
        {"pitch": 67, "duration": 1.0, "offset": 4.0},
    ]
    
    print("Testing compose_full_piece...")
    try:
        result = engine.compose_full_piece(test_data)
        print("Success!")
        print(f"Key: {result['key']}")
        print(f"Part 1 notes: {len(result['part1'])}")
        print(f"Part 2 notes: {len(result['part2'])}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
