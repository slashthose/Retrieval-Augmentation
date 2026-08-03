"""Challenge entry point: `python code/main.py` from the repository root."""
from pathlib import Path
from router import predict

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    predict(root / "dataset", root / "output.csv")
