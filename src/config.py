import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
EMB_DIR = os.path.join(OUT_DIR, "embeddings")

os.makedirs(EMB_DIR, exist_ok=True)