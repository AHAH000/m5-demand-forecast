"""Download and unzip the M5 data from Kaggle.

Before running:
  1. pip install kaggle
  2. Put your API token at ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token)
  3. Accept the competition rules at kaggle.com/c/m5-forecasting-accuracy
"""
import subprocess
import zipfile
from pathlib import Path

RAW = Path("data/raw")
COMPETITION = "m5-forecasting-accuracy"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW)],
        check=True,
    )
    zip_path = RAW / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    print("Done. Files in", RAW.resolve())


if __name__ == "__main__":
    main()
