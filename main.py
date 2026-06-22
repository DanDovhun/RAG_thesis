import argparse
import csv
import gc
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from qdrant_client.http.exceptions import UnexpectedResponse

sys.stdout.reconfigure(line_buffering=True)

import toml
from dotenv import load_dotenv

load_dotenv()

date = datetime.now
#log_folder = Path(f"%v-%v-%v_%v:%v:%v", date.year, date.month, date.day, date.hour, date.minute, date.second)
#log_folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR.resolve())

# Models are pre-downloaded (see download_models.py) — skip the per-file HTTP
# cache-validation calls to huggingface.co that otherwise slow down every load.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import deepl
import openai
import torch
from haystack import Document
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
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

file_types = {
    ".md", ".pdf", ".yaml", ".html", ".txt", ".rb", ".java", ".json",
    ".go", ".php", ".js", ".tx", ".cs", ".cshtml", ".csv", ".groovy", ".bat", ".sh",
}
dataset_dir = Path("submodules")

# Natural-language prose formats — translation candidates. Source code and
# structured/tabular formats (.json, .yaml, .csv, .java, ...) are excluded
# since machine-translating them would corrupt the content.
PROSE_FILE_TYPES = {".md", ".pdf", ".html", ".txt"}

DEEPL_TARGET_LANG_CODES = {"en": "en-US"}

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))

def load_model_configs(path: str = "embedder_config.toml") -> list[dict]:
    config = toml.load(path)
    models = []
    for section in ("dense", "sparse"):
        for m in config.get(section, []):
            if (
                m.get("enabled", False)
                and "jade" not in m.get("name", "").lower()
            ):
                m["type"] = section
                models.append(m)
    return models


def load_translation_config(path: str = "embedder_config.toml") -> dict:
    return toml.load(path).get("translation", {})


def is_prose_document(filepath: Path) -> bool:
    """Whether a file holds natural-language prose worth machine-translating."""
    return filepath.suffix.lower() in PROSE_FILE_TYPES or filepath.stem == "License"


def slugify(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


_git_root_cache: dict[Path, Path | None] = {}


def _git_root(directory: Path) -> Path | None:
    """Resolve and cache the repo root for `directory` (one rev-parse per directory, not per file)."""
    if directory not in _git_root_cache:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=str(directory),
        )
        _git_root_cache[directory] = Path(r.stdout.strip()) if r.returncode == 0 else None
    return _git_root_cache[directory]


def get_file_metadata(filepath: Path) -> dict:
    """Return git authors, ISO-8601 created_at and modified_at; falls back to filesystem times."""
    abs_path = filepath.resolve()
    authors: list[str] = []
    created_at: str | None = None
    modified_at: str | None = None

    git_root = _git_root(abs_path.parent)
    if git_root is not None:
        rel = abs_path.relative_to(git_root)

        r = subprocess.run(
            ["git", "log", "--follow", "--format=%aN%x1f%aI", "--", str(rel)],
            capture_output=True, text=True, cwd=str(git_root),
        )
        entries = [l.split("\x1f") for l in r.stdout.splitlines() if l.strip()]
        authors = list(dict.fromkeys(name for name, _ in entries))
        if entries:
            modified_at = entries[0][1]
            created_at = entries[-1][1]

    stat = abs_path.stat()
    if not modified_at:
        modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
    if not created_at:
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()

    return {"authors": authors, "created_at": created_at, "modified_at": modified_at}


def _collection_name(submodule: str, coll_alias: str) -> str:
    return f"{slugify(submodule)}_{coll_alias}"



class OpenAIDocumentEmbedder:
    """Thin wrapper around the OpenAI Embeddings API with the same run() interface as Haystack embedders."""

    # OpenAI limit is 300k tokens/request; 300-word chunks are ~400 tokens each,
    # so cap at 500 documents (~200k tokens) to stay safely under the limit.
    _BATCH_SIZE = 500

    def __init__(self, model_id: str, api_key: str) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id

    def run(self, documents: list[Document]) -> dict:
        texts = [d.content or "" for d in documents]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            response = self.client.embeddings.create(input=batch, model=self.model_id)
            embeddings.extend(e.embedding for e in sorted(response.data, key=lambda x: x.index))
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return {"documents": documents}


class RAGPipeline:
    def __init__(self, model_configs: list[dict], submodules: list[str], translation_config: dict | None = None):
        self.model_configs = {cfg["name"]: cfg for cfg in model_configs}
        self.embedders: dict[str, SentenceTransformersDocumentEmbedder] = {}
        self.bm25_store: InMemoryDocumentStore | None = None
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.splitter = DocumentSplitter(
            split_by="word", split_length=CHUNK_SIZE, split_overlap=CHUNK_OVERLAP
        )
        self._initialised_submodules: set[str] = set()
        self._translation_cache: dict[Path, list[Document] | None] = {}

        translation_config = translation_config or {}
        self.translation_languages = set(translation_config.get("languages", []))
        target = translation_config.get("target", "en")
        self.translation_target_lang = DEEPL_TARGET_LANG_CODES.get(target, target.upper())
        api_key = os.environ.get("DEEPL_API_KEY")
        if translation_config.get("enabled") and api_key:
            self.translator = deepl.Translator(api_key)
        else:
            self.translator = None
            if translation_config.get("enabled"):
                log.warning("Translation enabled in config but DEEPL_API_KEY is not set — skipping translation")

        for cfg in model_configs:
            self._setup_model(cfg)

    def _setup_model(self, cfg: dict) -> None:
        if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
            self.bm25_store = InMemoryDocumentStore()

    def load_embedder(self, name: str) -> None:
        cfg = self.model_configs[name]
        if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
            return
        if cfg.get("source") == "openai":
            api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                log.warning("OPENAI_KEY not set — skipping OpenAI embedder %s", name)
                return
            self.embedders[name] = OpenAIDocumentEmbedder(model_id=cfg["model_id"], api_key=api_key)
            return
        device = ComponentDevice.from_str("cuda:0" if torch.cuda.is_available() else "cpu")
        embedder = SentenceTransformersDocumentEmbedder(
            model=cfg["model_id"], trust_remote_code=True, device=device,
            model_kwargs={"torch_dtype": torch.float16},
            batch_size=cfg.get("batch_size", 32),
        )
        embedder.warm_up()
        self.embedders[name] = embedder

    def unload_embedder(self, name: str) -> None:
        cfg = self.model_configs.get(name, {})
        embedder = self.embedders.pop(name, None)
        if embedder is None:
            return
        del embedder
        if cfg.get("source") != "openai" and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def _embed_chunks(self, model_name: str, chunks: list[Document]) -> list[Document]:
        """Embed chunks, halving the sub-batch on each OOM until it succeeds or reaches size 1."""
        if not chunks:
            return []
        sub_batch_size = len(chunks)
        while True:
            try:
                results = []
                for i in range(0, len(chunks), sub_batch_size):
                    results.extend(
                        self.embedders[model_name].run(documents=chunks[i : i + sub_batch_size])["documents"]
                    )
                return results
            except torch.OutOfMemoryError:
                if sub_batch_size == 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                sub_batch_size = max(1, sub_batch_size // 2)
                log.warning("[%s] OOM — retrying embedding with sub-batch size %d", model_name, sub_batch_size)

    def _ensure_collection(self, name: str, model_id: str | None) -> None:
        if not self.qdrant.collection_exists(name):
            dim = EMBEDDING_DIMS.get(model_id, 768)
            log.info("Creating collection %r", name)
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _ensure_submodule_collections(self, submodule: str) -> None:
        if submodule in self._initialised_submodules:
            return
        for cfg in self.model_configs.values():
            if cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25":
                continue
            self._ensure_collection(_collection_name(submodule, cfg["coll_alias"]), cfg.get("model_id"))
        self._initialised_submodules.add(submodule)

    def translate_document(self, docs: list[Document], source_lang: str) -> list[Document] | None:
        """Translate prose document(s) from `source_lang` into English via DeepL.

        Returns new Documents holding the translated content (meta preserved,
        plus `source_language`), or None when translation isn't applicable —
        translator unavailable, `source_lang` out of scope, or the API call fails.
        """
        if self.translator is None or source_lang not in self.translation_languages:
            return None

        translated = []
        for doc in docs:
            if not doc.content:
                continue
            try:
                result = self.translator.translate_text(
                    doc.content, source_lang=source_lang, target_lang=self.translation_target_lang
                )
            except deepl.DeepLException as e:
                log.warning("Translation failed for %s: %s", doc.meta.get("file_path", "?"), e)
                return None
            translated.append(Document(content=result.text, meta={**doc.meta, "source_language": source_lang}))
        return translated or None

    def _index_translation(
        self, filepath: Path, docs: list[Document], model_name: str, cfg: dict,
        submodule: str, file_meta: dict, abs_path: str, file_ext: str,
    ) -> None:
        """Translate cs/sk prose documents to English and index them into a separate '<collection>_en'."""
        if cfg.get("language") in self.translation_languages:
            return  # don't run Czech/Slovak-specific embedders on machine-translated English text
        if not is_prose_document(filepath):
            return  # only translate natural-language prose, not source code / structured data

        if filepath not in self._translation_cache:
            source_lang = detect_language(" ".join(d.content for d in docs if d.content))
            self._translation_cache[filepath] = self.translate_document(docs, source_lang)
        translated = self._translation_cache[filepath]
        if not translated:
            return

        try:
            chunks = self.splitter.run(documents=translated)["documents"]
        except ValueError as e:
            log.warning("[%s] Skipping translated chunks for %s — %s", model_name, filepath, e)
            return

        embedded = self._embed_chunks(model_name, chunks)
        collection = f"{_collection_name(submodule, cfg['coll_alias'])}_en"
        self._ensure_collection(collection, cfg.get("model_id"))

        total = len(chunks)
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
                    "language": "en",
                    "source_language": chunk.meta.get("source_language"),
                    "authors": file_meta["authors"],
                    "created_at": file_meta["created_at"],
                    "modified_at": file_meta["modified_at"],
                    "model_name": model_name,
                    "model_id": cfg.get("model_id"),
                    "translated": True,
                },
            )
            for i, chunk in enumerate(embedded)
            if chunk.embedding is not None
        ]
        if points:
            self.qdrant.upsert(collection_name=collection, points=points)

    def index_document(self, filepath: Path, model_name: str, file_meta: dict) -> dict:
        log.info(f"Processing: {filepath}")
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

        if model_name not in self.embedders:
            elapsed = time.perf_counter() - t0
            log.warning("[%s] Skipping %s — embedder not loaded (check API key?)", model_name, filepath)
            return {"status": "skipped", "reason": "embedder_not_loaded", "chunks": 0, "elapsed_s": elapsed, "file_size_bytes": file_size}

        try:
            chunks = self.splitter.run(documents=docs)["documents"]
        except ValueError as e:
            elapsed = time.perf_counter() - t0
            log.warning("[%s] Skipping %s — %s", model_name, filepath, e)
            return {"status": "skipped", "reason": str(e), "chunks": 0, "elapsed_s": elapsed, "file_size_bytes": file_size}

        total = len(chunks)
        embedded = self._embed_chunks(model_name, chunks)
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
            try:
                logging.info("Uploading to database")
                self.qdrant.upsert(collection_name=collection, points=points)

            except UnexpectedResponse as e:
                logging.warning(e)
                points = split(points, 10)
                logging.info(f"Split points to {len(points)} of {len(points[0])}")
                
                for point in points:
                    self.qdrant.upsert(collection_name=collection, points=point)

            logging.info(f"Processed {filepath}")

        self._index_translation(filepath, docs, model_name, cfg, submodule, file_meta, abs_path, file_ext)

        elapsed = time.perf_counter() - t0
        return {"status": "indexed", "chunks": total, "elapsed_s": elapsed, "file_size_bytes": file_size}

def process_docs(args):
    failed_files = []
    count = 0
    """Iterates through documents and sends them off to the function to be processed into a vector database"""
    submodules = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    model_configs = load_model_configs()
    translation_config = load_translation_config()

    if args.reset:
        client = QdrantClient(url=QDRANT_URL)
        for collection in client.get_collections().collections:
            log.info("Deleting collection: %s", collection.name)
            client.delete_collection(collection.name)

    rag = RAGPipeline(model_configs, submodules, translation_config)

    files = [
        (filepath, get_file_metadata(filepath))
        for filepath in dataset_dir.rglob("*")
        if filepath.is_file() and (filepath.suffix in file_types or filepath.stem == "License")
    ]

    log.info("Loaded %d files across %d submodules", len(files), len(submodules))

    run_start = datetime.now()
    log.info("Run started at %s", run_start.isoformat())

    metrics_rows: list[dict] = []

    model_names = list(rag.model_configs.keys())

    rag.load_embedder(model_names[0])

    for i, model_name in enumerate(model_names):
        log.info("=== Model: %s ===", model_name)
        for filepath, file_meta in files:
            try:
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
                count += 1

            except torch.OutOfMemoryError as e:
                print(f"Can't process {filepath}: {e}")
                failed_files.append(filepath)
                continue

        rag.unload_embedder(model_name)

        # Load next model only after current is freed from GPU — two models
        # simultaneously in VRAM exceeds the available 5.67 GiB.
        if i + 1 < len(model_names):
            rag.load_embedder(model_names[i + 1])

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
    log.info(f"Fail rate: {len(failed_files)} / {count} ({len(failed_files)/count})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete all Qdrant collections before indexing")
    parser.add_argument("--translate", action="store_true", help="Translates documents to English")
    args = parser.parse_args()


    process_docs(args)

if __name__ == "__main__":
    main()