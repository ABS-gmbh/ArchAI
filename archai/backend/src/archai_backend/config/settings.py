from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
DB_PATH = DATA_DIR / "indexes" / "archai.sqlite"
