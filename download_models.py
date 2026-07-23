"""Pre-download all enabled HuggingFace embedding models from embedder_config.toml."""

import os
import sys
from pathlib import Path

import toml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, SparseEncoder

load_dotenv()

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True)

config = toml.load("embedder_config.toml")

dense_models = [m for m in config.get("dense", []) if m.get("source") == "huggingface" and m.get("enabled", False)]
sparse_models = [
    m for m in config.get("sparse", [])
    if m.get("source") == "huggingface" and m.get("enabled", False) and m.get("model_id") != "bm25"
]

failed = []
for m in dense_models:
    model_id = m["model_id"]
    print(f"Downloading {m['name']} ({model_id}) ...", flush=True)
    try:
        SentenceTransformer(model_id, trust_remote_code=True, cache_folder=str(MODELS_DIR))
        print(f"  OK\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        failed.append(model_id)

for m in sparse_models:
    model_id = m["model_id"]
    print(f"Downloading {m['name']} ({model_id}) ...", flush=True)
    try:
        SparseEncoder(model_id, trust_remote_code=True, cache_folder=str(MODELS_DIR))
        print(f"  OK\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        failed.append(model_id)

if failed:
    print(f"Failed models: {failed}")
    sys.exit(1)

print("All models downloaded.")
