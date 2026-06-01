import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import toml
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR.resolve())

from haystack.components.converters import HTMLToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from langdetect import DetectorFactory, LangDetectException, detect
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DetectorFactory.seed = 0  # deterministic language detection

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

# langdetect returns ISO 639-1 codes; map to the collection suffixes the user expects.
# Chunks in any other language are skipped.
LANG_SUFFIX: dict[str, str] = {"sk": "sk", "cs": "cz"}


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


class RAGPipeline:
    def __init__(self, model_configs: list[dict]):
        self.model_configs = {cfg["name"]: cfg for cfg in model_configs}
        self.embedders: dict[str, SentenceTransformersDocumentEmbedder] = {}
        self.qdrant_collections: dict[str, dict[str, str]] = {}  # model → {lang_suffix → collection}
        self.bm25_store: InMemoryDocumentStore | None = None
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.splitter = DocumentSplitter(
            split_by="word", split_length=CHUNK_SIZE, split_overlap=CHUNK_OVERLAP
        )

        for cfg in model_configs:
            self._setup_model(cfg)

    def _setup_model(self, cfg: dict) -> None:
        name = cfg["name"]
        source = cfg.get("source")
        model_id = cfg.get("model_id")

        if source == "haystack" and model_id == "bm25":
            self.bm25_store = InMemoryDocumentStore()
            return

        base = name.lower().replace(" ", "_").replace("-", "_")
        dim = EMBEDDING_DIMS.get(model_id, 768)

        existing = {c.name for c in self.qdrant.get_collections().collections}
        lang_collections: dict[str, str] = {}
        for suffix in LANG_SUFFIX.values():
            coll = f"{base}_{suffix}"
            if coll not in existing:
                self.qdrant.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            lang_collections[suffix] = coll
        self.qdrant_collections[name] = lang_collections

        embedder = SentenceTransformersDocumentEmbedder(model=model_id, trust_remote_code=True)
        embedder.warm_up()
        self.embedders[name] = embedder

    def index_document(self, filepath: Path, model_name: str, file_meta: dict) -> None:
        filepath = Path(filepath)
        cfg = self.model_configs[model_name]
        file_ext = filepath.suffix.lower()
        abs_path = str(filepath.resolve())

        if file_ext == ".html":
            converter = HTMLToDocument()
        elif file_ext == ".pdf":
            converter = PyPDFToDocument()
        else:
            converter = TextFileToDocument()

        docs = converter.run(sources=[filepath])["documents"]
        if not docs:
            return

        if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
            DocumentWriter(document_store=self.bm25_store).run(documents=docs)
            return

        chunks = self.splitter.run(documents=docs)["documents"]
        total = len(chunks)

        embedded = self.embedders[model_name].run(documents=chunks)["documents"]
        lang_collections = self.qdrant_collections[model_name]

        buckets: dict[str, list[PointStruct]] = {s: [] for s in lang_collections}
        for i, chunk in enumerate(embedded):
            if chunk.embedding is None:
                continue
            lang = detect_language(chunk.content or "")
            lang_suffix = LANG_SUFFIX.get(lang)
            if lang_suffix is None:
                continue
            buckets[lang_suffix].append(PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "file_path": abs_path,
                    "file_name": filepath.name,
                    "file_type": file_ext,
                    "chunk_index": i,
                    "chunk_count": total,
                    "language": lang,
                    "authors": file_meta["authors"],
                    "created_at": file_meta["created_at"],
                    "modified_at": file_meta["modified_at"],
                    "model_name": model_name,
                    "model_id": cfg.get("model_id"),
                },
            ))

        for lang_suffix, points in buckets.items():
            if points:
                self.qdrant.upsert(collection_name=lang_collections[lang_suffix], points=points)


model_configs = load_model_configs()
rag = RAGPipeline(model_configs)

for filepath in dataset_dir.rglob("*"):
    if filepath.is_file() and (filepath.suffix in file_types or filepath.stem == "License"):
        print(f"Indexing: {filepath}")
        file_meta = get_file_metadata(filepath)
        for model_name in rag.model_configs:
            print(f"  Model: {model_name}")
            rag.index_document(filepath, model_name, file_meta)
