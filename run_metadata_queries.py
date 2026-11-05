"""Run the metadata-tiered query files against Qdrant and score them with the same
RAGAS context-precision/context-recall judge used for the earlier retrieval runs
(see main.py's --retrieve). One retrieval_results/retrieval_<tier>_<timestamp>.json
is written per input queries file, in the same schema as main.py's output, so it
drops straight into the existing retrieval_results/graphs.ipynb analysis.

Unlike main.py --retrieve (which applies one --mode to every collection in a
query entry, so bm25/splade collections listed together both get searched with
whichever single mode was passed — see the "mislabeled embedder=BM25" note in
graphs.ipynb), this script inspects each collection's *_bm25_hybrid /
*_splade_hybrid / dense suffix and routes it to the matching search, so mode and
embedder in the output always reflect what was actually queried.

Only embedders enabled in embedder_config.toml are queryable — enable whichever
ones the target queries files need before running. Each queries_meta_*.json
file lists both dense and sparse (bm25_hybrid/splade_hybrid) collections per
project, so a single run covers both.

Each row also carries "query_time_s": the wall-clock time spent retrieving that
row's hits (embedding the query and searching Qdrant), excluding the RAGAS
evaluation. The run's total retrieval time is the sum of these, so evaluation
time between queries is not counted either. Output files are shaped as:

    {"query_count": N, "total_query_time_s": ..., "mean_query_time_s": ...,
     "results": [<row>, ...]}

Older result files are a bare list of rows; load_rows() accepts both.
"""
import argparse
import fcntl
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rag_pipeline import RAGPipeline, evaluate_context_metrics, load_model_configs, load_translation_config

dataset_dir = Path("submodules")

DEFAULT_QUERIES_FILES = [
    "queries_meta_curated.json",
    "queries_meta_half.json",
    "queries_meta_full.json",
]

# Cross-process mutex around GPU-heavy sections. Loading/unloading one embedder
# at a time only bounds VRAM *within* a single process -- two concurrent
# invocations (e.g. curated + half) process the same projects in the same
# order at the same pace, so they reach the same large embedder (e.g.
# Qwen3-Embedding) at close to the same moment regardless. Without this lock
# that's two independent ~2.25GB loads racing for a ~5.67GB card. The lock
# makes them take turns instead: one blocks until the other finishes with
# that embedder, so nothing OOMs and nothing gets skipped.
GPU_LOCK_PATH = Path(__file__).resolve().parent / ".gpu.lock"


@contextmanager
def gpu_lock():
    with open(GPU_LOCK_PATH, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_queries_config(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_rows(path: Path) -> list[dict]:
    """Result rows from a results file, either the current {"results": [...]}
    shape or the older bare list."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["results"] if isinstance(data, dict) else data


def route_search(rag: RAGPipeline, project: str, collection: str, query: str, top_k: int):
    """Search `collection` with whichever backend actually produced it, inferred
    from its name. Returns (hits, mode, embedder)."""
    if collection.endswith("_bm25_hybrid"):
        return rag.search_bm25(project, query, top_k), "bm25", rag.bm25_model_name
    if collection.endswith("_splade_hybrid"):
        return rag.search_splade(project, query, top_k), "splade", rag.splade_model_name

    cfg = rag.resolve_collection_model(project, collection)
    if cfg is None:
        log.warning("Could not resolve embedder for collection %r (project %r)", collection, project)
        return [], "dense", None
    if not rag.qdrant.collection_exists(collection):
        log.warning("Collection %r does not exist — skipping", collection)
        return [], "dense", cfg["name"]
    return rag.search_dense(collection, cfg["name"], query, top_k), "dense", cfg["name"]


def build_tasks(entries: list[dict]) -> list[dict]:
    """Flatten entries into one task per (project, query, collection), each
    tagged with its original position so output order is preserved."""
    tasks = []
    for p_idx, entry in enumerate(entries):
        project = entry["project"]
        for q_idx, query_entry in enumerate(entry["queries"]):
            for c_idx, collection in enumerate(entry["collections"]):
                tasks.append({
                    "order": (p_idx, q_idx, c_idx),
                    "project": project,
                    "collection": collection,
                    "query": query_entry["query"],
                    "relevant_files": query_entry.get("relevant_files", []),
                    "reference_answer": query_entry.get("reference_answer"),
                    "metadata": query_entry.get("metadata"),
                })
    return tasks


def group_key(rag: RAGPipeline, task: dict) -> tuple[str, str | None]:
    """Grouping key so tasks are run one dense embedder at a time -- loading every
    enabled dense model up front (the previous behaviour) parks all of them in GPU
    memory for the rest of the process, which is what caused the CUDA OOM on this
    GPU once queries_meta_*.json started listing every embedder per project."""
    collection = task["collection"]
    if collection.endswith("_bm25_hybrid"):
        return ("bm25", None)
    if collection.endswith("_splade_hybrid"):
        return ("splade", rag.splade_model_name)
    cfg = rag.resolve_collection_model(task["project"], collection)
    return ("dense", cfg["name"] if cfg else None)


def process_group(rag: RAGPipeline, group_tasks: list[dict], top_k: int) -> list[dict]:
    results = []
    for task in group_tasks:
        project, collection, query = task["project"], task["collection"], task["query"]
        # Time only retrieval (query embedding, Qdrant search); the RAGAS
        # evaluation below is deliberately left outside.
        start = time.perf_counter()
        hits, mode, embedder = route_search(rag, project, collection, query, top_k)
        query_time_s = time.perf_counter() - start
        scores = evaluate_context_metrics(query, hits, task["reference_answer"])

        log.info(
            "[%s/%s] embedder=%s %r → %d hit(s)  query_time=%.4fs  context_precision=%s  context_recall=%s",
            project, collection, embedder, query, len(hits), query_time_s,
            scores["context_precision"], scores["context_recall"],
        )
        for rank, hit in enumerate(hits, start=1):
            snippet = (hit["content"] or "").replace("\n", " ")[:120]
            log.info(
                "  #%d  score=%.4f  %s [chunk %s]  %s",
                rank, hit["score"] or 0.0, hit["file_name"], hit["chunk_index"], snippet,
            )

        results.append({
            "order": task["order"],
            "project": project,
            "collection": collection,
            "mode": mode,
            "embedder": embedder,
            "query": query,
            "relevant_files": task["relevant_files"],
            "reference_answer": task["reference_answer"],
            "metadata": task["metadata"],
            "hits": hits,
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
            "query_time_s": query_time_s,
        })
    return results


def run_queries_file(rag: RAGPipeline, queries_file: str, top_k: int, out_dir: Path) -> Path:
    entries = load_queries_config(queries_file)
    run_start = datetime.now()
    tasks = build_tasks(entries)

    groups: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    for task in tasks:
        groups[group_key(rag, task)].append(task)

    results: list[dict] = []
    for (kind, model_name), group_tasks in groups.items():
        needs_gpu = kind in ("dense", "splade") and model_name is not None
        if not needs_gpu:
            results.extend(process_group(rag, group_tasks, top_k))
            continue

        with gpu_lock():
            rag.load_embedder(model_name)
            try:
                results.extend(process_group(rag, group_tasks, top_k))
            finally:
                rag.unload_embedder(model_name)

    results.sort(key=lambda r: r["order"])
    for r in results:
        del r["order"]

    total_query_time_s = sum(r["query_time_s"] for r in results)
    log.info(
        "[%s] Total retrieval time: %.4fs over %d quer(y/ies) (evaluation excluded)",
        queries_file, total_query_time_s, len(results),
    )
    output = {
        "query_count": len(results),
        "total_query_time_s": total_query_time_s,
        "mean_query_time_s": total_query_time_s / len(results) if results else None,
        "results": results,
    }

    stem = Path(queries_file).stem
    out_path = out_dir / f"retrieval_{stem}_{run_start.strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log.info("[%s] Saved %d result(s) to %s", queries_file, len(results), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "queries_files", nargs="*", default=DEFAULT_QUERIES_FILES,
        help="Queries JSON file(s) to run (default: the 3 metadata-tiered copies)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return per query")
    args = parser.parse_args()

    missing = [qf for qf in args.queries_files if not Path(qf).exists()]
    for qf in missing:
        log.warning("Queries file %r not found — skipping", qf)
    queries_files = [qf for qf in args.queries_files if qf not in missing]
    if not queries_files:
        log.error("No queries files to run")
        return

    model_configs = load_model_configs()
    translation_config = load_translation_config()
    rag = RAGPipeline(model_configs, translation_config, dataset_dir=dataset_dir)

    out_dir = Path("retrieval_results")
    out_dir.mkdir(exist_ok=True)

    for qf in queries_files:
        run_queries_file(rag, qf, args.top_k, out_dir)


if __name__ == "__main__":
    main()
