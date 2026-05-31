"""Pre-download all enabled HuggingFace embedding models from embedder_config.toml."""

import sys
import toml
from sentence_transformers import SentenceTransformer

config = toml.load("embedder_config.toml")

hf_models = [
    m for section in ("dense", "sparse")
    for m in config.get(section, [])
    if m.get("source") == "huggingface" and m.get("enabled", False)
]

failed = []
for m in hf_models:
    model_id = m["model_id"]
    print(f"Downloading {m['name']} ({model_id}) ...", flush=True)
    try:
        SentenceTransformer(model_id, trust_remote_code=True)
        print(f"  OK\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        failed.append(model_id)

if failed:
    print(f"Failed models: {failed}")
    sys.exit(1)

print("All models downloaded.")
