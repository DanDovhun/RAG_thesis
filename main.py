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
    PGVECTOR_CONN_STR,
    QDRANT_URL,
    RAGPipeline,
    evaluate_context_metrics,
    extract_file_text,
    get_file_metadata,
    load_model_configs,
    load_translation_config,
    reciprocal_rank_fusion,
)

file_types = {
    ".md", ".pdf", ".yaml", ".html", ".txt", ".rb", ".java", ".json",
    ".go", ".php", ".js", ".tx", ".cs", ".cshtml", ".csv", ".groovy", ".bat", ".sh",
}
dataset_dir = Path("submodules")


def _mask_conn_str(conn_str: str) -> str:
    """Redact the password portion of a postgresql:// connection string for logging."""
    creds, _, rest = conn_str.partition("@")
    if "://" not in creds or ":" not in creds.split("://", 1)[1]:
        return conn_str
    scheme, userinfo = creds.split("://", 1)
    user, _, _password = userinfo.partition(":")
    return f"{scheme}://{user}:***@{rest}"


def process_docs(args):
    failed_files = []
    pgvector_failures = []
    count = 0
    """Iterates through documents and sends them off to the function to be processed into a vector database"""
    submodules = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    model_configs = load_model_configs()
    translation_config = load_translation_config()

    log.info("Qdrant target: %s", QDRANT_URL)
    log.info("Pgvector target: %s", _mask_conn_str(PGVECTOR_CONN_STR))

    if args.reset:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        for collection in client.get_collections().collections:
            log.info("Deleting collection: %s", collection.name)
            client.delete_coly7u8lection(collection.name)

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
                qdrant_status = result.get("qdrant_status", "n/a")
                pgvector_status = result.get("pgvector_status", "n/a")
                log.info(
                    "[%s] %s → status=%s  chunks=%d  elapsed=%.3fs  size=%d B  qdrant=%s  pgvector=%s",
                    model_name, filepath.name, result["status"], result["chunks"],
                    result["elapsed_s"], result["file_size_bytes"], qdrant_status, pgvector_status,
                )
                if pgvector_status == "failed":
                    pgvector_failures.append({
                        "model_name": model_name, "filepath": str(filepath),
                        "error": "pgvector write failed — see log above for details",
                    })
                metrics_rows.append({
                    "model_name": model_name,
                    "filepath": str(filepath),
                    "file_name": filepath.name,
                    "file_size_bytes": result["file_size_bytes"],
                    "status": result["status"],
                    "chunks": result["chunks"],
                    "elapsed_s": round(result["elapsed_s"], 4),
                    "qdrant_status": qdrant_status,
                    "pgvector_status": pgvector_status,
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

    _save_run_metrics(run_start, metrics_rows, failed_files, pgvector_failures, count)


def _save_run_metrics(
    run_start: datetime, metrics_rows: list[dict], failed_files: list[dict],
    pgvector_failures: list[dict], count: int,
) -> None:
    run_end = datetime.now()
    log.info("Run finished at %s  (total %.1fs)", run_end.isoformat(), (run_end - run_start).total_seconds())

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    csv_path = metrics_dir / f"run_{run_start.strftime('%Y%m%d_%H%M%S')}.csv"

    fieldnames = ["run_start", "run_end", "model_name", "filepath", "file_name", "file_size_bytes", "status", "chunks", "elapsed_s", "qdrant_status", "pgvector_status"]
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

    if pgvector_failures:
        pgvector_failed_path = metrics_dir / f"pgvector_failed_{run_start.strftime('%Y%m%d_%H%M%S')}.csv"
        with pgvector_failed_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model_name", "filepath", "error"])
            writer.writeheader()
            writer.writerows(pgvector_failures)
        log.warning(
            "%d file(s) failed to write to Pgvector (Qdrant indexing was unaffected) — see %s",
            len(pgvector_failures), pgvector_failed_path,
        )

    log.info("Fail rate: %d / %d", len(failed_files), count)


def process_sparse(args):
    """Index the same document set as process_docs, but only the BM25 sparse vectors.

    Skips dense embedders entirely — useful for backfilling sparse vectors into
    Qdrant without re-running the much slower dense embedding models.
    """
    failed_files = []
    count = 0
    submodules = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    model_configs = load_model_configs()
    sparse_configs = [cfg for cfg in model_configs if cfg["type"] == "sparse"]
    if not sparse_configs:
        log.warning("No sparse models enabled in embedder_config.toml — nothing to do")
        return

    log.info("Qdrant target: %s", QDRANT_URL)

    rag = RAGPipeline(model_configs, dataset_dir=dataset_dir)

    if args.reset:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        for submodule in submodules:
            collection = rag.sparse_collection_name(submodule)
            if collection and client.collection_exists(collection):
                log.info("Deleting collection: %s", collection)
                client.delete_collection(collection)

    files = [
        (filepath, get_file_metadata(filepath))
        for filepath in dataset_dir.rglob("*")
        if filepath.is_file() and (filepath.suffix in file_types or filepath.stem == "License")
    ]
    log.info("Loaded %d files across %d submodules", len(files), len(submodules))

    run_start = datetime.now()
    log.info("Sparse-only run started at %s", run_start.isoformat())

    metrics_rows: list[dict] = []

    for cfg in sparse_configs:
        model_name = cfg["name"]
        log.info("=== Sparse model: %s ===", model_name)
        rag.load_embedder(model_name)  # no-op for BM25; loads the SparseEncoder for learned sparse models
        for filepath, file_meta in files:
            try:
                log.info("[%s] Indexing: %s  (%.1f KB)", model_name, filepath, filepath.stat().st_size / 1024)
                result = rag.index_document(filepath, model_name, file_meta)
                qdrant_status = result.get("qdrant_status", "n/a")
                log.info(
                    "[%s] %s → status=%s  chunks=%d  elapsed=%.3fs  size=%d B  qdrant=%s",
                    model_name, filepath.name, result["status"], result["chunks"],
                    result["elapsed_s"], result["file_size_bytes"], qdrant_status,
                )
                metrics_rows.append({
                    "model_name": model_name,
                    "filepath": str(filepath),
                    "file_name": filepath.name,
                    "file_size_bytes": result["file_size_bytes"],
                    "status": result["status"],
                    "chunks": result["chunks"],
                    "elapsed_s": round(result["elapsed_s"], 4),
                    "qdrant_status": qdrant_status,
                    "pgvector_status": "n/a",
                })
                count += 1
            except Exception as e:
                log.warning("Can't process %s with %s: %s", filepath, model_name, e)
                failed_files.append({"model_name": model_name, "filepath": str(filepath), "error": str(e)})
                continue
        rag.unload_embedder(model_name)

    _save_run_metrics(run_start, metrics_rows, failed_files, [], count)


def load_queries_config(path: str) -> list[dict]:
    """Load a JSON file of [{"project", "collections", "queries": [{"query", "relevant_files"}, ...]}, ...]."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# Cap per-file context fed to the answer-generation LLM to bound cost/context length —
# generous enough for READMEs/configs/source files, not the whole of huge generated files.
_REFERENCE_ANSWER_MAX_CHARS_PER_FILE = 8000


def generate_reference_answers(args) -> None:
    """Auto-generate a ground-truth `reference_answer` per query in the queries file.

    Feeds an LLM the query plus the full text of its `relevant_files` and asks for a
    concise, extractive answer. Persists answers back into the same JSON file so
    `evaluate_context_metrics`'s RAGAS LLM judge has real ground truth to score
    retrieved chunks against, instead of only knowing which files should be relevant.
    Queries that already have a `reference_answer` are left untouched (safe to re-run
    after adding new queries).
    """
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_KEY/OPENAI_API_KEY not set — cannot generate reference answers")
        return
    client = openai.OpenAI(api_key=api_key)

    entries = load_queries_config(args.queries_file)
    generated = 0
    for entry in entries:
        for query_entry in entry["queries"]:
            if query_entry.get("reference_answer"):
                continue

            query = query_entry["query"]
            file_texts = []
            for relative_path in query_entry.get("relevant_files", []):
                try:
                    text = extract_file_text(relative_path)
                except Exception as e:
                    log.warning("Could not read %s for %r: %s", relative_path, query, e)
                    continue
                if text:
                    file_texts.append(f"### {relative_path}\n{text[:_REFERENCE_ANSWER_MAX_CHARS_PER_FILE]}")

            if not file_texts:
                log.warning("No readable relevant_files for %r — skipping", query)
                continue

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using only the information in the provided file "
                            "content. Write a detailed paragraph (at least 4-5 sentences): give the "
                            "direct answer first, then add the surrounding facts, values, and context "
                            "from the file that support or relate to it. Every sentence must be a "
                            "distinct, verifiable fact actually present in the file content — don't "
                            "pad with restatements or vague filler. If the content doesn't contain the "
                            "answer, say so explicitly."
                        ),
                    },
                    {"role": "user", "content": f"Question: {query}\n\n" + "\n\n".join(file_texts)},
                ],
            )
            answer = response.choices[0].message.content.strip()
            query_entry["reference_answer"] = answer
            generated += 1
            log.info("[%d] %r → %s", generated, query, answer[:150])

    with open(args.queries_file, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log.info("Generated %d reference answer(s), saved to %s", generated, args.queries_file)


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
        for query_entry in entry["queries"]:
            query = query_entry["query"]
            relevant_files = query_entry.get("relevant_files", [])
            reference_answer = query_entry.get("reference_answer")
            for collection in collections:
                dense_hits: list[dict] = []
                bm25_hits: list[dict] = []
                dense_model_name: str | None = None

                if args.mode in ("dense", "hybrid"):
                    cfg = rag.resolve_collection_model(project, collection)
                    if cfg is None:
                        log.warning("Could not resolve embedder for collection %r (project %r)", collection, project)
                    elif not rag.qdrant.collection_exists(collection):
                        log.warning("Collection %r does not exist — skipping", collection)
                    else:
                        dense_model_name = cfg["name"]
                        dense_hits = rag.search_dense(collection, dense_model_name, query, args.top_k)

                if args.mode in ("bm25", "hybrid"):
                    bm25_hits = rag.search_bm25(project, query, args.top_k)

                splade_hits: list[dict] = []
                if args.mode == "splade":
                    splade_hits = rag.search_splade(project, query, args.top_k)

                if args.mode == "dense":
                    hits = dense_hits
                    embedder = dense_model_name
                elif args.mode == "bm25":
                    hits = bm25_hits
                    embedder = rag.bm25_model_name
                elif args.mode == "splade":
                    hits = splade_hits
                    embedder = rag.splade_model_name
                else:
                    hits = reciprocal_rank_fusion(dense_hits, bm25_hits, args.top_k)
                    embedder = "+".join(n for n in (dense_model_name, rag.bm25_model_name) if n) or None

                scores = evaluate_context_metrics(query, hits, reference_answer)

                log.info(
                    "[%s/%s] embedder=%s %r → %d hit(s)  context_precision=%s  context_recall=%s",
                    project, collection, embedder, query, len(hits),
                    scores["context_precision"], scores["context_recall"],
                )
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
                    "embedder": embedder,
                    "query": query,
                    "relevant_files": relevant_files,
                    "reference_answer": reference_answer,
                    "hits": hits,
                    "context_precision": scores["context_precision"],
                    "context_recall": scores["context_recall"],
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
        "--generate-answers", action="store_true",
        help="Auto-generate reference_answer values in --queries-file via an LLM, then exit",
    )
    parser.add_argument(
        "--sparse-only", action="store_true",
        help="Index only BM25 sparse vectors (skips dense embedders)",
    )
    parser.add_argument(
        "--queries-file", default="queries.json",
        help="JSON file of [{project, collections, queries}, ...] to run in --retrieve mode",
    )
    parser.add_argument(
        "--mode", choices=["dense", "bm25", "splade", "hybrid"], default="dense",
        help="Retrieval mode for --retrieve",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return per query")
    args = parser.parse_args()

    if args.retrieve:
        run_retrieval(args)
    elif args.generate_answers:
        generate_reference_answers(args)
    elif args.sparse_only:
        process_sparse(args)
    else:
        process_docs(args)

if __name__ == "__main__":
    main()
