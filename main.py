import argparse
import csv
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import toml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR.resolve())

import torch
from haystack.components.converters import HTMLToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice
from langdetect import DetectorFactory, LangDetectException, detect
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DetectorFactory.seed = 0

QDRANT_URL = "http://localhost:6333"

EMBEDDING_DIMS = {
    "gerulata/slovakbert": 768,
    "UWB-AIR/Czert-B-base-cased": 768,
    "FacebookAI/xlm-roberta-base": 768,
    "google-bert/bert-base-multilingual-cased": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "naver/splade-v3": 768,
}

file_types = {
    ".md", ".pdf", ".yaml", ".html", ".txt", ".rb", ".java", ".json",
    ".go", ".php", ".js", ".tx", ".cs", ".cshtml", ".csv", ".groovy", ".bat", ".sh",
}
dataset_dir = Path("submodules")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
def load_model_configs(path: str = "embedder_config.toml") -> list[dict]:
    config = toml.load(path)
    models = []
    for section in ("dense", "sparse"):
        for m in config.get(section, []):
            if (
                m.get("enabled", False)
                and m.get("source") != "openai"
                and "jade" not in m.get("name", "").lower()
            ):
                m["type"] = section
                models.append(m)
    return models


def slugify(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def get_file_metadata(filepath: Path) -> dict:
    """Return git authors, ISO-8601 created_at and modified_at; falls back to filesystem times."""
    abs_path = filepath.resolve()
    authors: list[str] = []
    created_at: str | None = None
    modified_at: str | None = None

    root_r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=str(abs_path.parent),
    )
    if root_r.returncode == 0:
        git_root = Path(root_r.stdout.strip())
        rel = abs_path.relative_to(git_root)

        r = subprocess.run(
            ["git", "log", "--follow", "--format=%aN", "--", str(rel)],
            capture_output=True, text=True, cwd=str(git_root),
        )
        authors = list(dict.fromkeys(l.strip() for l in r.stdout.splitlines() if l.strip()))

        r = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(rel)],
            capture_output=True, text=True, cwd=str(git_root),
        )
        modified_at = r.stdout.strip() or None

        r = subprocess.run(
            ["git", "log", "--follow", "--format=%aI", "--", str(rel)],
            capture_output=True, text=True, cwd=str(git_root),
        )
        lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        created_at = lines[-1] if lines else None

    stat = abs_path.stat()
    if not modified_at:
        modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
    if not created_at:
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()

    return {"authors": authors, "created_at": created_at, "modified_at": modified_at}


def _collection_name(submodule: str, coll_alias: str) -> str:
    return f"{slugify(submodule)}_{coll_alias}"



class RAGPipeline:
    def __init__(self, model_configs: list[dict], submodules: list[str]):
        self.model_configs = {cfg["name"]: cfg for cfg in model_configs}
        self.embedders: dict[str, SentenceTransformersDocumentEmbedder] = {}
        self.bm25_store: InMemoryDocumentStore | None = None
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.splitter = DocumentSplitter(
            split_by="word", split_length=CHUNK_SIZE, split_overlap=CHUNK_OVERLAP
        )
        self._initialised_submodules: set[str] = set()

        for cfg in model_configs:
            self._setup_model(cfg)

    def _setup_model(self, cfg: dict) -> None:
        if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
            self.bm25_store = InMemoryDocumentStore()

    def load_embedder(self, name: str) -> None:
        cfg = self.model_configs[name]
        if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
            return
        device = ComponentDevice.from_str("cuda:0" if torch.cuda.is_available() else "cpu")
        embedder = SentenceTransformersDocumentEmbedder(
            model=cfg["model_id"], trust_remote_code=True, device=device
        )
        embedder.warm_up()
        self.embedders[name] = embedder

    def unload_embedder(self, name: str) -> None:
        embedder = self.embedders.pop(name, None)
        if embedder is None:
            return
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_submodule_collections(self, submodule: str) -> None:
        if submodule in self._initialised_submodules:
            return
        for cfg in self.model_configs.values():
            if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
                continue
            coll = _collection_name(submodule, cfg["coll_alias"])
            if not self.qdrant.collection_exists(coll):
                dim = EMBEDDING_DIMS.get(cfg.get("model_id"), 768)
                log.info("Creating collection %r", coll)
                self.qdrant.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        self._initialised_submodules.add(submodule)

    def index_document(self, filepath: Path, model_name: str, file_meta: dict) -> dict:
        """Index one document with one model. Returns a metrics dict."""
        filepath = Path(filepath)
        cfg = self.model_configs[model_name]
        file_ext = filepath.suffix.lower()
        abs_path = str(filepath.resolve())
        file_size = filepath.stat().st_size
        submodule = filepath.relative_to(dataset_dir).parts[0]
        self._ensure_submodule_collections(submodule)

        t0 = time.perf_counter()

        if file_ext == ".html":
            converter = HTMLToDocument()
        elif file_ext == ".pdf":
            converter = PyPDFToDocument()
        else:
            converter = TextFileToDocument()

        docs = converter.run(sources=[filepath])["documents"]
        docs = [d for d in docs if d.content is not None]
        if not docs:
            elapsed = time.perf_counter() - t0
            log.warning("[%s] Skipping %s — no text content extracted", model_name, filepath)
            return {"status": "skipped", "reason": "no_content", "chunks": 0, "elapsed_s": elapsed, "file_size_bytes": file_size}

        if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
            DocumentWriter(document_store=self.bm25_store).run(documents=docs)
            elapsed = time.perf_counter() - t0
            return {"status": "bm25", "chunks": len(docs), "elapsed_s": elapsed, "file_size_bytes": file_size}

        try:
            chunks = self.splitter.run(documents=docs)["documents"]
        except ValueError as e:
            elapsed = time.perf_counter() - t0
            log.warning("[%s] Skipping %s — %s", model_name, filepath, e)
            return {"status": "skipped", "reason": str(e), "chunks": 0, "elapsed_s": elapsed, "file_size_bytes": file_size}

        total = len(chunks)
        embedded = self.embedders[model_name].run(documents=chunks)["documents"]
        collection = _collection_name(submodule, cfg["coll_alias"])

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "file_path": abs_path,
                    "file_name": filepath.name,
                    "file_type": file_ext,
                    "chunk_index": i,
                    "chunk_count": total,
                    "language": detect_language(chunk.content or ""),
                    "authors": file_meta["authors"],
                    "created_at": file_meta["created_at"],
                    "modified_at": file_meta["modified_at"],
                    "model_name": model_name,
                    "model_id": cfg.get("model_id"),
                },
            )
            for i, chunk in enumerate(embedded)
            if chunk.embedding is not None
        ]

        if points:
            self.qdrant.upsert(collection_name=collection, points=points)

        elapsed = time.perf_counter() - t0
        return {"status": "indexed", "chunks": total, "elapsed_s": elapsed, "file_size_bytes": file_size}


parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Delete all Qdrant collections before indexing")
args = parser.parse_args()

submodules = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
model_configs = load_model_configs()

if args.reset:
    client = QdrantClient(url=QDRANT_URL)
    for collection in client.get_collections().collections:
        log.info("Deleting collection: %s", collection.name)
        client.delete_collection(collection.name)

rag = RAGPipeline(model_configs, submodules)

files = [
    (filepath, get_file_metadata(filepath))
    for filepath in dataset_dir.rglob("*")
    if filepath.is_file() and (filepath.suffix in file_types or filepath.stem == "License")
]

log.info("Loaded %d files across %d submodules", len(files), len(submodules))

run_start = datetime.now()
log.info("Run started at %s", run_start.isoformat())

metrics_rows: list[dict] = []

for model_name in rag.model_configs:
    log.info("=== Model: %s ===", model_name)
    rag.load_embedder(model_name)
    for filepath, file_meta in files:
        log.info("[%s] Indexing: %s  (%.1f KB)", model_name, filepath, filepath.stat().st_size / 1024)
        result = rag.index_document(filepath, model_name, file_meta)
        log.info(
            "[%s] %s → status=%s  chunks=%d  elapsed=%.3fs  size=%d B",
            model_name, filepath.name, result["status"], result["chunks"],
            result["elapsed_s"], result["file_size_bytes"],
        )
        metrics_rows.append({
            "model_name": model_name,
            "filepath": str(filepath),
            "file_name": filepath.name,
            "file_size_bytes": result["file_size_bytes"],
            "status": result["status"],
            "chunks": result["chunks"],
            "elapsed_s": round(result["elapsed_s"], 4),
        })
    rag.unload_embedder(model_name)

run_end = datetime.now()
log.info("Run finished at %s  (total %.1fs)", run_end.isoformat(), (run_end - run_start).total_seconds())

metrics_dir = Path("metrics")
metrics_dir.mkdir(exist_ok=True)
csv_path = metrics_dir / f"run_{run_start.strftime('%Y%m%d_%H%M%S')}.csv"

fieldnames = ["run_start", "run_end", "model_name", "filepath", "file_name", "file_size_bytes", "status", "chunks", "elapsed_s"]
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for row in metrics_rows:
        writer.writerow({"run_start": run_start.isoformat(), "run_end": run_end.isoformat(), **row})

log.info("Metrics saved to %s", csv_path)
