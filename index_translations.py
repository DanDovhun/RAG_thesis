"""Index the translations/ output tree produced by translate_to_english.py into the
"<...>_en" Qdrant/Pgvector collections — the English embedding space main.py's
DeepL-based rag_pipeline.RAGPipeline._index_translation would otherwise populate.

For each project folder under --translations-dir, in order:
  1. every "<stem>.md" / "<stem>_metadata.json" pair is embedded as-is (no further
     text extraction — the .md content already IS the final text), tagging chunks
     with the metadata.json's authors/created_at/modified_at/source_path instead of
     recomputing them.
  2. every file listed in non-prose.json (any reason — not just "not_prose") is then
     extracted and embedded exactly like main.py's normal indexing would extract any
     supported file (extract_file_text + get_file_metadata), so the "_en" collection
     ends up covering the full corpus, not just the subset that happened to be
     translatable prose.

Only three embedders run here — the English-capable dense ones the config marks as
"applied to EN machine-translated versions of cs/sk documents" plus the multilingual
BERT baseline: Qwen3-Embedding, text-embedding-3-small, bert-base-multilingual-cased.
Language-specific (e5-sk-small, simcse-small-e-czech) and sparse (BM25, SPLADE)
embedders are never applicable to this English-space corpus. Like main.py, embedders
are loaded one at a time (not all simultaneously) to fit in GPU memory.

Usage:
    python index_translations.py [--dataset-dir submodules] [--translations-dir translations]
"""

import argparse
import json
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import openai
import torch
from haystack import Document
from qdrant_client.models import PointStruct

from qdrant_client.http.exceptions import UnexpectedResponse

from rag_pipeline import RAGPipeline, extract_file_text, get_file_metadata, load_model_configs, load_translation_config
from rag_pipeline.pipeline import _collection_name, detect_language

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# The only embedders that ever write into "_en" collections.
TARGET_MODEL_NAMES = ["Qwen3-Embedding", "text-embedding-3-small", "bert-base-multilingual-cased"]

# Fallback batch size when a single upsert exceeds Qdrant's request-size cap — a
# fixed small batch size (rather than a fixed batch *count*) keeps each retry batch
# small regardless of how many chunks the offending file produced.
QDRANT_RETRY_BATCH_SIZE = 3


def select_target_models(model_configs: list[dict]) -> list[dict]:
    by_name = {cfg["name"]: cfg for cfg in model_configs}
    selected = []
    for name in TARGET_MODEL_NAMES:
        cfg = by_name.get(name)
        if cfg is None:
            log.warning("%r is not enabled in embedder_config.toml — skipping", name)
            continue
        selected.append(cfg)
    return selected


def load_non_prose_entries(project_dir: Path, dataset_dir: Path) -> list[tuple[Path, str]]:
    """Returns [(original_file_path, skip_reason), ...] for every non-prose.json entry
    that still has a file on disk."""
    non_prose_path = project_dir / "non-prose.json"
    if not non_prose_path.exists():
        return []
    entries = json.loads(non_prose_path.read_text(encoding="utf-8"))
    files = []
    for entry in entries:
        original_path = dataset_dir / entry["source_relative_path"]
        if not original_path.is_file():
            log.warning("non-prose.json entry %r has no file on disk — skipping", entry["source_relative_path"])
            continue
        files.append((original_path, entry.get("reason", "")))
    return files


def load_translated_pairs(project_dir: Path) -> list[tuple[Path, dict]]:
    pairs = []
    for md_path in sorted(project_dir.glob("*.md")):
        json_path = project_dir / f"{md_path.stem}_metadata.json"
        if not json_path.exists():
            log.warning("%s has no matching *_metadata.json — skipping", md_path)
            continue
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        pairs.append((md_path, metadata))
    return pairs


def _upsert_en_chunks(
    rag: RAGPipeline, collection: str, model_name: str, cfg: dict, chunks: list[Document],
    abs_path: str, file_name: str, file_ext: str, extra_payload: dict,
) -> str:
    """Shared tail for both _en indexing paths: dedup check, embed, build payload, upsert."""
    rag._ensure_collection(collection, cfg.get("model_id"))

    in_qdrant = rag._qdrant_has_document(collection, abs_path)
    in_pgvector = rag._pgvector_has_document(collection, abs_path, cfg.get("model_id"))
    if in_qdrant and in_pgvector:
        return "already_indexed"

    embedded = rag._embed_chunks(model_name, chunks)

    total = len(chunks)
    points: list[PointStruct] = []
    pg_documents: list[Document] = []
    for i, chunk in enumerate(embedded):
        if chunk.embedding is None:
            continue
        payload = {
            "content": chunk.content,
            "file_path": abs_path,
            "file_name": file_name,
            "file_type": file_ext,
            "chunk_index": i,
            "chunk_count": total,
            "model_name": model_name,
            "model_id": cfg.get("model_id"),
            **extra_payload,
        }
        if not in_qdrant:
            points.append(PointStruct(id=str(uuid.uuid4()), vector=chunk.embedding, payload=payload))
        if not in_pgvector:
            pg_documents.append(Document(content=chunk.content, embedding=chunk.embedding, meta=payload))

    if points:
        try:
            rag.qdrant.upsert(collection_name=collection, points=points)
        except UnexpectedResponse as e:
            # A single upsert of every chunk in one file can exceed Qdrant's
            # request-size cap — retry in small fixed-size batches so it scales to
            # however many chunks the offending file produced.
            log.warning(e)
            point_batches = [
                points[i : i + QDRANT_RETRY_BATCH_SIZE]
                for i in range(0, len(points), QDRANT_RETRY_BATCH_SIZE)
            ]
            log.info("Split points into %d batch(es) of up to %d", len(point_batches), QDRANT_RETRY_BATCH_SIZE)
            for point_batch in point_batches:
                rag.qdrant.upsert(collection_name=collection, points=point_batch)
    if pg_documents:
        rag._write_pgvector(collection, cfg.get("model_id"), pg_documents)

    return "indexed"


def index_translated_pair(rag: RAGPipeline, md_path: Path, metadata: dict, model_name: str) -> str:
    """Embed an already-translated .md file as-is and upsert it into the "<...>_en"
    collection, tagging every chunk with the metadata translate_to_english.py already
    computed instead of recomputing it."""
    cfg = rag.model_configs[model_name]
    if model_name not in rag.embedders:
        return "embedder_not_loaded"

    text = md_path.read_text(encoding="utf-8")
    if not text.strip():
        return "empty_content"

    try:
        chunks = rag.splitter.run(documents=[Document(content=text)])["documents"]
    except ValueError as e:
        log.warning("[%s] Skipping %s — %s", model_name, md_path, e)
        return "skipped_split_error"

    submodule = metadata["project"]
    abs_path = metadata["source_path"]
    collection = f"{_collection_name(submodule, cfg['coll_alias'])}_en"

    return _upsert_en_chunks(
        rag, collection, model_name, cfg, chunks,
        abs_path=abs_path,
        file_name=Path(abs_path).name,
        file_ext=metadata.get("file_type", Path(abs_path).suffix.lower()),
        extra_payload={
            "language": "en",
            "source_language": metadata.get("source_language"),
            "authors": metadata.get("authors", []),
            "created_at": metadata.get("created_at"),
            "modified_at": metadata.get("modified_at"),
            "translated": True,
            "translation_engine": metadata.get("translation_engine"),
            "translation_model_id": metadata.get("translation_model_id"),
        },
    )


def index_original_into_en(rag: RAGPipeline, filepath: Path, project: str, model_name: str) -> str:
    """Extract a non-prose.json original exactly like main.py's normal indexing would
    (extract_file_text + get_file_metadata), then embed it as-is into the "<...>_en"
    collection — there's nothing to translate in a config/code/data file, or the
    prose file was already skipped for another reason, so its original content
    stands in directly."""
    cfg = rag.model_configs[model_name]
    if model_name not in rag.embedders:
        return "embedder_not_loaded"

    try:
        text = extract_file_text(filepath)
    except Exception as e:
        log.warning("Could not extract text from %s: %s", filepath, e)
        return "failed_extract"
    if not text.strip():
        return "empty_content"

    try:
        chunks = rag.splitter.run(documents=[Document(content=text)])["documents"]
    except ValueError as e:
        log.warning("[%s] Skipping %s — %s", model_name, filepath, e)
        return "skipped_split_error"

    abs_path = str(filepath.resolve())
    file_meta = get_file_metadata(filepath)
    collection = f"{_collection_name(project, cfg['coll_alias'])}_en"

    return _upsert_en_chunks(
        rag, collection, model_name, cfg, chunks,
        abs_path=abs_path,
        file_name=filepath.name,
        file_ext=filepath.suffix.lower(),
        extra_payload={
            "language": detect_language(text),
            "authors": file_meta["authors"],
            "created_at": file_meta["created_at"],
            "modified_at": file_meta["modified_at"],
            "translated": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="submodules", help="Root directory the original submodules live in")
    parser.add_argument("--translations-dir", default="translations", help="Output of translate_to_english.py")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    translations_dir = Path(args.translations_dir)

    projects = sorted(d for d in translations_dir.iterdir() if d.is_dir())
    log.info("Found %d project folder(s) under %s", len(projects), translations_dir)

    # Gather per-project work once, up front, so embedders can still be loaded one
    # at a time below — main.py's process_docs() does the same, since loading
    # multiple dense models simultaneously exceeds available GPU memory.
    non_prose_by_project: dict[str, list[tuple[Path, str]]] = {}
    translated_by_project: dict[str, list[tuple[Path, dict]]] = {}
    for project_dir in projects:
        non_prose_by_project[project_dir.name] = load_non_prose_entries(project_dir, dataset_dir)
        translated_by_project[project_dir.name] = load_translated_pairs(project_dir)

    total_non_prose = sum(len(v) for v in non_prose_by_project.values())
    total_translated = sum(len(v) for v in translated_by_project.values())
    log.info(
        "%d translated .md/.json pair(s) and %d non-prose.json file(s) to index into _en collections",
        total_translated, total_non_prose,
    )

    model_configs = select_target_models(load_model_configs())
    if not model_configs:
        log.error("None of %s are enabled in embedder_config.toml — nothing to do", TARGET_MODEL_NAMES)
        return
    translation_config = load_translation_config()
    rag = RAGPipeline(model_configs, translation_config, dataset_dir=dataset_dir)

    model_names = list(rag.model_configs.keys())
    log.info("Embedders: %s", model_names)

    counts: dict[str, int] = {}
    rag.load_embedder(model_names[0])
    for i, model_name in enumerate(model_names):
        log.info("=== Model: %s ===", model_name)

        for project_dir in projects:
            project = project_dir.name

            # 1. Translated prose -> _en.
            for md_path, metadata in translated_by_project[project]:
                try:
                    status = index_translated_pair(rag, md_path, metadata, model_name)
                except (torch.OutOfMemoryError, openai.BadRequestError) as e:
                    log.warning("Can't process %s with %s: %s", md_path, model_name, e)
                    status = "failed"
                counts[status] = counts.get(status, 0) + 1
                log.info("[%s] [%s] %s -> %s", model_name, project, md_path.name, status)

            # 2. Everything non-prose.json recorded (any reason) -> _en, extracted fresh.
            for original_path, reason in non_prose_by_project[project]:
                try:
                    status = index_original_into_en(rag, original_path, project, model_name)
                except (torch.OutOfMemoryError, openai.BadRequestError) as e:
                    log.warning("Can't process %s with %s: %s", original_path, model_name, e)
                    status = "failed"
                counts[status] = counts.get(status, 0) + 1
                log.info("[%s] [%s] %s (%s) -> %s", model_name, project, original_path.name, reason, status)

        rag.unload_embedder(model_name)
        if i + 1 < len(model_names):
            rag.load_embedder(model_names[i + 1])

    log.info("Done. Summary: %s", counts)


if __name__ == "__main__":
    main()
