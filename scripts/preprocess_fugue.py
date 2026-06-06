import runpy
import sys
import os

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "v5", "preprocess_fugue.py")
    runpy.run_path(target, run_name="__main__")
