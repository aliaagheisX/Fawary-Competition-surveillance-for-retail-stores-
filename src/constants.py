from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent 

DATA_DIR = BASE_DIR / "datasets"
TRAIN_PATH = DATA_DIR 
TEST_PATH = DATA_DIR / "test"

assert BASE_DIR.exists(), "Not BASE_DIR exists"
assert TRAIN_PATH.exists(), "Not TRAIN_PATHexists"
assert TEST_PATH.exists(), "Not TEST_PATH exists"
