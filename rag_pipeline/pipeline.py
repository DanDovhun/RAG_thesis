import gc
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

import deepl
import openai
import toml
import torch
from haystack import Document
from haystack.components.converters import HTMLToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice, Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from langdetect import DetectorFactory, LangDetectException, detect
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

DetectorFactory.seed = 0

log = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"

# Matches the `pgvector` service in compose.yml (host port 5433 → container 5432,
# user/password postgres/postgres, db vectordb). PgvectorDocumentStore reads its
# connection string from the PG_CONN_STR env var, so seed it with this default
# unless the environment (e.g. .env) already overrides it.
PGVECTOR_CONN_STR = os.environ.get(
    "PG_CONN_STR", "postgresql://postgres:postgres@localhost:5433/vectordb"
)
os.environ.setdefault("PG_CONN_STR", PGVECTOR_CONN_STR)

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

# Natural-language prose formats — translation candidates. Source code and
# structured/tabular formats (.json, .yaml, .csv, .java, ...) are excluded
# since machine-translating them would corrupt the content.
PROSE_FILE_TYPES = {".md", ".pdf", ".html", ".txt"}

DEEPL_TARGET_LANG_CODES = {"en": "en-US"}

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

RRF_K = 60


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


def _point_to_hit(point) -> dict:
    payload = point.payload or {}
    return {
        "score": point.score,
        "file_path": payload.get("file_path"),
        "file_name": payload.get("file_name"),
        "chunk_index": payload.get("chunk_index"),
        "content": payload.get("content"),
    }


def _doc_to_hit(doc: Document) -> dict:
    return {
        "score": doc.score,
        "file_path": doc.meta.get("file_path"),
        "file_name": doc.meta.get("file_name"),
        "chunk_index": doc.meta.get("chunk_index"),
        "content": doc.content,
    }


def reciprocal_rank_fusion(dense_hits: list[dict], bm25_hits: list[dict], top_k: int) -> list[dict]:
    """Merge dense and BM25 rankings via Reciprocal Rank Fusion, keyed by (file_path, chunk_index)."""
    scores: dict[tuple, float] = {}
    hits_by_key: dict[tuple, dict] = {}
    for hits in (dense_hits, bm25_hits):
        for rank, hit in enumerate(hits, start=1):
            key = (hit["file_path"], hit["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            hits_by_key.setdefault(key, hit)
    ranked_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**hits_by_key[key], "score": scores[key]} for key in ranked_keys]


class OpenAIDocumentEmbedder:
    """Thin wrapper around the OpenAI Embeddings API with the same run() interface as Haystack embedders."""

    _BATCH_SIZE = 500
    # 8192 token limit × ~3 chars/token, with headroom for multi-byte chars
    _MAX_CHARS_PER_DOC = 24_000

    def __init__(self, model_id: str, api_key: str) -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        log.warning("Truncating document from %d to %d chars for OpenAI embedding", len(text), max_chars)
        return text[:max_chars]

    def run(self, documents: list[Document]) -> dict:
        texts = [self._truncate(d.content or "", self._MAX_CHARS_PER_DOC) for d in documents]
        embeddings: list[list[float]] = []
        batch_size = self._BATCH_SIZE
        i = 0
        while i < len(texts):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(input=batch, model=self.model_id)
                embeddings.extend(e.embedding for e in sorted(response.data, key=lambda x: x.index))
                i += batch_size
            except openai.RateLimitError:
                log.warning("OpenAI rate limit (429) — waiting 10 s before retry")
                time.sleep(10)
            except openai.BadRequestError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                log.warning("OpenAI token limit hit — retrying with batch size %d", batch_size)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return {"documents": documents}


class RAGPipeline:
    def __init__(
        self, model_configs: list[dict], translation_config: dict | None = None,
        dataset_dir: Path | None = None,
    ):
        self.model_configs = {cfg["name"]: cfg for cfg in model_configs}
        self.dataset_dir = dataset_dir
        self.embedders: dict[str, SentenceTransformersDocumentEmbedder] = {}
        self.bm25_store: InMemoryDocumentStore | None = None
        self.qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
        self._pg_stores: dict[str, PgvectorDocumentStore] = {}
        self.splitter = DocumentSplitter(
            split_by="word", split_length=CHUNK_SIZE, split_overlap=CHUNK_OVERLAP
        )
        self._initialised_submodules: set[str] = set()
        self._translation_cache: dict[Path, list[Document] | None] = {}
        self._bm25_stores: dict[str, InMemoryDocumentStore] = {}
        self._alias_to_cfg = {
            cfg["coll_alias"]: cfg
            for cfg in self.model_configs.values()
            if not (cfg.get("source") == "haystack" and cfg.get("model_id") == "bm25")
        }

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

    def resolve_collection_model(self, project: str, collection: str) -> dict | None:
        """Map a Qdrant collection name back to the dense model config that produced it."""
        prefix = f"{slugify(project)}_"
        if not collection.startswith(prefix):
            return None
        alias = collection[len(prefix):]
        if alias.endswith("_en"):
            alias = alias[: -len("_en")]
        return self._alias_to_cfg.get(alias)

    def search_dense(self, collection: str, model_name: str, query: str, top_k: int) -> list[dict]:
        if model_name not in self.embedders:
            self.load_embedder(model_name)
        if model_name not in self.embedders:
            log.warning("Embedder %s unavailable — skipping dense search on %s", model_name, collection)
            return []
        embedded = self.embedders[model_name].run(documents=[Document(content=query)])["documents"][0]
        if embedded.embedding is None:
            return []
        response = self.qdrant.query_points(
            collection_name=collection, query=embedded.embedding, limit=top_k, with_payload=True
        )
        return [_point_to_hit(p) for p in response.points]

    def _build_bm25_store(self, collection: str) -> InMemoryDocumentStore:
        store = InMemoryDocumentStore()
        docs = []
        next_offset = None
        while True:
            records, next_offset = self.qdrant.scroll(
                collection_name=collection,
                limit=500,
                offset=next_offset,
                with_payload=["content", "file_path", "file_name", "chunk_index"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                content = payload.get("content")
                if not content:
                    continue
                docs.append(Document(content=content, meta={
                    "file_path": payload.get("file_path"),
                    "file_name": payload.get("file_name"),
                    "chunk_index": payload.get("chunk_index"),
                }))
            if next_offset is None:
                break
        if docs:
            DocumentWriter(document_store=store).run(documents=docs)
        return store

    def search_bm25(self, collection: str, query: str, top_k: int) -> list[dict]:
        if collection not in self._bm25_stores:
            if not self.qdrant.collection_exists(collection):
                log.warning("Collection %r does not exist — skipping BM25 search", collection)
                return []
            log.info("Building BM25 index for %r ...", collection)
            self._bm25_stores[collection] = self._build_bm25_store(collection)
        hits = self._bm25_stores[collection].bm25_retrieval(query, top_k=top_k)
        return [_doc_to_hit(d) for d in hits]

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

    def _get_pg_store(self, table_name: str, model_id: str | None) -> PgvectorDocumentStore:
        """Lazily create (and cache) the Pgvector table backing one Qdrant-equivalent collection."""
        if table_name not in self._pg_stores:
            self._pg_stores[table_name] = PgvectorDocumentStore(
                connection_string=Secret.from_env_var("PG_CONN_STR"),
                table_name=table_name,
                embedding_dimension=EMBEDDING_DIMS.get(model_id, 768),
            )
        return self._pg_stores[table_name]

    def _qdrant_has_document(self, collection: str, file_path: str) -> bool:
        """Whether any chunk of `file_path` is already stored in this Qdrant collection."""
        if not self.qdrant.collection_exists(collection):
            return False
        try:
            records, _ = self.qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(records) > 0
        except UnexpectedResponse as e:
            log.warning(
                "[qdrant] Existence check failed for %r in %r: %s — assuming not present", file_path, collection, e
            )
            return False

    def _pgvector_has_document(self, table_name: str, file_path: str, model_id: str | None) -> bool:
        """Whether any chunk of `file_path` is already stored in this Pgvector table."""
        try:
            store = self._get_pg_store(table_name, model_id)
            docs = store.filter_documents(filters={"field": "meta.file_path", "operator": "==", "value": file_path})
            return len(docs) > 0
        except DocumentStoreError as e:
            log.warning(
                "[pgvector] Existence check failed for %r in table %r: %s — assuming not present",
                file_path, table_name, e,
            )
            self._pg_stores.pop(table_name, None)
            return False

    def _write_pgvector(self, table_name: str, model_id: str | None, documents: list[Document]) -> bool:
        """Mirror embedded chunks into Pgvector. Returns True on success; logs and returns False on any failure."""
        if not documents:
            return True
        try:
            store = self._get_pg_store(table_name, model_id)
            store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
            return True
        except DocumentStoreError as e:
            log.warning("[pgvector] Failed to write %d document(s) to table %r: %s", len(documents), table_name, e)
            # Drop the cached store so a stale/broken connection is retried fresh next time.
            self._pg_stores.pop(table_name, None)
            return False

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

        collection = f"{_collection_name(submodule, cfg['coll_alias'])}_en"
        self._ensure_collection(collection, cfg.get("model_id"))

        in_qdrant = self._qdrant_has_document(collection, abs_path)
        in_pgvector = self._pgvector_has_document(collection, abs_path, cfg.get("model_id"))
        if in_qdrant and in_pgvector:
            log.info(
                "[%s] Translated %s already present in Qdrant and Pgvector — skipping", model_name, filepath.name
            )
            return

        embedded = self._embed_chunks(model_name, chunks)

        total = len(chunks)
        points: list[PointStruct] = []
        pg_documents: list[Document] = []
        for i, chunk in enumerate(embedded):
            if chunk.embedding is None:
                continue
            payload = {
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
            }
            if not in_qdrant:
                points.append(PointStruct(id=str(uuid.uuid4()), vector=chunk.embedding, payload=payload))
            if not in_pgvector:
                pg_documents.append(Document(content=chunk.content, embedding=chunk.embedding, meta=payload))

        if points:
            self.qdrant.upsert(collection_name=collection, points=points)
        if pg_documents:
            self._write_pgvector(collection, cfg.get("model_id"), pg_documents)

    def index_document(self, filepath: Path, model_name: str, file_meta: dict) -> dict:
        log.info(f"Processing: {filepath}")
        """Index one document with one model. Returns a metrics dict."""
        filepath = Path(filepath)
        cfg = self.model_configs[model_name]
        file_ext = filepath.suffix.lower()
        abs_path = str(filepath.resolve())
        file_size = filepath.stat().st_size
        submodule = filepath.relative_to(self.dataset_dir).parts[0]
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
            elapsed = time.perf_counter() - t0
            try:
                DocumentWriter(document_store=self.bm25_store).run(documents=docs)
            except DuplicateDocumentError as e:
                log.warning("[%s] Skipping %s — already present in BM25 store: %s", model_name, filepath, e)
                return {"status": "skipped", "reason": "duplicate_bm25", "chunks": 0, "elapsed_s": elapsed, "file_size_bytes": file_size}
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
        collection = _collection_name(submodule, cfg["coll_alias"])

        in_qdrant = self._qdrant_has_document(collection, abs_path)
        in_pgvector = self._pgvector_has_document(collection, abs_path, cfg.get("model_id"))

        if in_qdrant and in_pgvector:
            log.info("[%s] Skipping %s — already indexed in Qdrant and Pgvector", model_name, filepath)
            self._index_translation(filepath, docs, model_name, cfg, submodule, file_meta, abs_path, file_ext)
            elapsed = time.perf_counter() - t0
            return {
                "status": "already_indexed", "chunks": total, "elapsed_s": elapsed, "file_size_bytes": file_size,
                "qdrant_status": "skipped_exists", "pgvector_status": "skipped_exists",
            }

        embedded = self._embed_chunks(model_name, chunks)

        points: list[PointStruct] = []
        pg_documents: list[Document] = []
        for i, chunk in enumerate(embedded):
            if chunk.embedding is None:
                continue
            payload = {
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
            }
            if not in_qdrant:
                points.append(PointStruct(id=str(uuid.uuid4()), vector=chunk.embedding, payload=payload))
            if not in_pgvector:
                pg_documents.append(Document(content=chunk.content, embedding=chunk.embedding, meta=payload))

        if in_qdrant:
            log.info("[%s] %s already present in Qdrant collection %r — skipping upload", model_name, filepath.name, collection)
            qdrant_status = "skipped_exists"
        elif points:
            try:
                log.info("Uploading to database")
                self.qdrant.upsert(collection_name=collection, points=points)

            except UnexpectedResponse as e:
                log.warning(e)
                point_batches = split(points, 10)
                log.info(f"Split points to {len(point_batches)} of {len(point_batches[0])}")

                for point_batch in point_batches:
                    self.qdrant.upsert(collection_name=collection, points=point_batch)

            log.info(f"Processed {filepath}")
            qdrant_status = "written"
        else:
            qdrant_status = "skipped"

        if in_pgvector:
            log.info("[%s] %s already present in Pgvector table %r — skipping upload", model_name, filepath.name, collection)
            pgvector_status = "skipped_exists"
        elif pg_documents:
            pgvector_status = "written" if self._write_pgvector(collection, cfg.get("model_id"), pg_documents) else "failed"
        else:
            pgvector_status = "skipped"

        self._index_translation(filepath, docs, model_name, cfg, submodule, file_meta, abs_path, file_ext)

        elapsed = time.perf_counter() - t0
        return {
            "status": "indexed", "chunks": total, "elapsed_s": elapsed, "file_size_bytes": file_size,
            "qdrant_status": qdrant_status, "pgvector_status": pgvector_status,
        }
