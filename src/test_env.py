import sys
import os

def check_env():
    print("=== BPGE Environment Check ===")
    print(f"Python Version: {sys.version}")
    
    modules = ['music21', 'tqdm', 'mido', 'pretty_midi', 'numpy']
    missing = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {module} is installed.")
        except ImportError:
            print(f"[MISSING] {module} is NOT installed.")
            missing.append(module)
            
    try:
        import mlx
        print("[OK] mlx (Apple Silicon) is installed.")
    except ImportError:
        if sys.version_info < (3, 10):
            print("[INFO] MLX requires Python 3.10+. Current version is below 3.10.")
        else:
            print("[MISSING] mlx is NOT installed.")
            missing.append('mlx')

    if not missing:
        print("\nAll required core libraries for data processing are ready.")
    else:
        print(f"\nPlease install missing libraries: pip install {' '.join(missing)}")
        
    print("\nNote: MLX is required for model training/inference on Apple Silicon.")
    print("If you haven't, please update Python to 3.10+ to install MLX.")

if __name__ == "__main__":
    check_env()
