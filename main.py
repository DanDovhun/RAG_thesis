import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

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

# Models are pre-downloaded (see download_models.py) — skip the per-file HTTP
# cache-validation calls to huggingface.co that otherwise slow down every load.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import openai
import torch
from qdrant_client import QdrantClient

from rag_pipeline import (
    QDRANT_URL,
    RAGPipeline,
    get_file_metadata,
    load_model_configs,
    load_translation_config,
    reciprocal_rank_fusion,
)

file_types = {
    ".md", ".pdf", ".yaml", ".html", ".txt", ".rb", ".java", ".json",
    ".go", ".php", ".js", ".tx", ".cs", ".cshtml", ".csv", ".groovy", ".bat", ".sh",
}
dataset_dir = Path("part_1/submodules")


def process_docs(args):
    failed_files = []
    count = 0
    """Iterates through documents and sends them off to the function to be processed into a vector database"""
    submodules = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    model_configs = load_model_configs()
    translation_config = load_translation_config()

    if args.reset:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        for collection in client.get_collections().collections:
            log.info("Deleting collection: %s", collection.name)
            client.delete_collection(collection.name)

    rag = RAGPipeline(model_configs, translation_config, dataset_dir=dataset_dir)

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

            except (torch.OutOfMemoryError, openai.BadRequestError) as e:
                log.warning("Can't process %s with %s: %s", filepath, model_name, e)
                failed_files.append({"model_name": model_name, "filepath": str(filepath), "error": str(e)})
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

    if failed_files:
        failed_path = metrics_dir / f"failed_{run_start.strftime('%Y%m%d_%H%M%S')}.csv"
        with failed_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model_name", "filepath", "error"])
            writer.writeheader()
            writer.writerows(failed_files)
        log.warning("%d file(s) failed — see %s", len(failed_files), failed_path)

    log.info("Fail rate: %d / %d", len(failed_files), count)


def load_queries_config(path: str) -> list[dict]:
    """Load a JSON file of [{"project", "collections", "queries"}, ...]."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_retrieval(args) -> None:
    """Run each project's queries against its listed collections and save the results."""
    model_configs = load_model_configs()
    rag = RAGPipeline(model_configs, dataset_dir=dataset_dir)
    entries = load_queries_config(args.queries_file)

    run_start = datetime.now()
    results: list[dict] = []

    for entry in entries:
        project = entry["project"]
        collections = entry["collections"]
        for query in entry["queries"]:
            for collection in collections:
                dense_hits: list[dict] = []
                bm25_hits: list[dict] = []

                if args.mode in ("dense", "hybrid"):
                    cfg = rag.resolve_collection_model(project, collection)
                    if cfg is None:
                        log.warning("Could not resolve embedder for collection %r (project %r)", collection, project)
                    elif not rag.qdrant.collection_exists(collection):
                        log.warning("Collection %r does not exist — skipping", collection)
                    else:
                        dense_hits = rag.search_dense(collection, cfg["name"], query, args.top_k)

                if args.mode in ("bm25", "hybrid"):
                    bm25_hits = rag.search_bm25(collection, query, args.top_k)

                if args.mode == "dense":
                    hits = dense_hits
                elif args.mode == "bm25":
                    hits = bm25_hits
                else:
                    hits = reciprocal_rank_fusion(dense_hits, bm25_hits, args.top_k)

                log.info("[%s/%s] %r → %d hit(s)", project, collection, query, len(hits))
                for rank, hit in enumerate(hits, start=1):
                    snippet = (hit["content"] or "").replace("\n", " ")[:120]
                    log.info(
                        "  #%d  score=%.4f  %s [chunk %s]  %s",
                        rank, hit["score"] or 0.0, hit["file_name"], hit["chunk_index"], snippet,
                    )

                results.append({
                    "project": project,
                    "collection": collection,
                    "mode": args.mode,
                    "query": query,
                    "hits": hits,
                })

    out_dir = Path("retrieval_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"retrieval_{run_start.strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    log.info("Retrieval results saved to %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete all Qdrant collections before indexing")
    parser.add_argument("--translate", action="store_true", help="Translates documents to English")
    parser.add_argument("--retrieve", action="store_true", help="Run retrieval queries instead of indexing")
    parser.add_argument(
        "--queries-file", default="queries.json",
        help="JSON file of [{project, collections, queries}, ...] to run in --retrieve mode",
    )
    parser.add_argument(
        "--mode", choices=["dense", "bm25", "hybrid"], default="dense",
        help="Retrieval mode for --retrieve",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return per query")
    args = parser.parse_args()

    if args.retrieve:
        run_retrieval(args)
    else:
        process_docs(args)

if __name__ == "__main__":
    main()
