"""Run RQ3's category-split query file(s) against Qdrant and score them with the
same RAGAS context-precision/context-recall judge used for the other retrieval
runs (see main.py's --retrieve and run_metadata_queries.py).

RQ3 asks whether separating documents into per-category collections (code,
prose/documentation, config) improves retrieval versus keeping every embedder's
documents in one merged collection. queries_rq3.json holds the same
project/query/relevant_files/reference_answer entries as queries.json, but each
project's "collections" list points at the "_code"/"_prose"/"_config" per-embedder
splits produced by split_legacy_collections_by_filetype.py instead of the single
merged collection per embedder.

The splits are evaluated as one retriever per embedder, not one per split: each
query is sent to every split collection of an embedder (top-k from each), the
hits are pooled and re-ranked by similarity score, and only the overall top-k
are passed to the RAGAS judge. Scores are comparable across the pooled splits
because all of them were produced by the same embedder with the same distance
metric. This gives one result row per (project, query, embedder), directly
paired with that embedder's merged-collection baseline row.

Each output row's "collection" is the shared split prefix plus "_split" (e.g.
"altoeditorclient_xlmr_split"), "source_collections" lists the split collections
that were queried, and every hit carries a "source_collection" field saying which
split it came from.

--from-results <file> rebuilds the pooled top-k from an earlier per-split run
(one row per split collection, as written by the previous version of this script)
instead of querying Qdrant, so only the RAGAS judge is re-run. This is exact as
long as that run used the same --top-k: each split's own top-k always contains
everything that split can contribute to the pooled top-k.

Each row also carries "query_time_s": the wall-clock time spent retrieving that
row's hits (embedding the query, searching every split collection in Qdrant and
pooling the results), excluding the RAGAS evaluation. The run's total retrieval
time is the sum of these, so evaluation time between queries is not counted
either.

Writes retrieval_results/retrieval_queries_rq3_<timestamp>.json (one file per
input queries file), shaped as:

    {"query_count": N, "total_query_time_s": ..., "mean_query_time_s": ...,
     "results": [<row>, ...]}

Runs rebuilt with --from-results query nothing, so their rows have no
query_time_s and the file has only "results". Older result files are a bare
list of rows; load_rows() accepts both.

Usage:
    python3 run_rq3_queries.py [queries_rq3.json ...] [--top-k 5]
    python3 run_rq3_queries.py --from-results retrieval_results/retrieval_queries_rq3_<ts>.json [--top-k 5]
"""
import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from run_metadata_queries import (
    RAGPipeline,
    evaluate_context_metrics,
    gpu_lock,
    group_key,
    load_model_configs,
    load_queries_config,
    load_rows,
    load_translation_config,
    log,
    route_search,
)

DEFAULT_QUERIES_FILES = ["queries_rq3.json"]
SPLIT_SUFFIXES = ("_code", "_prose", "_config")


def split_base(collection: str) -> str:
    """'altoeditorclient_xlmr_code' -> 'altoeditorclient_xlmr'."""
    for suffix in SPLIT_SUFFIXES:
        if collection.endswith(suffix):
            return collection[: -len(suffix)]
    return collection


def group_split_collections(collections: list[str]) -> dict[str, list[str]]:
    """Group a project's collections by embedder, i.e. by their shared split prefix,
    preserving the order they're listed in."""
    groups: dict[str, list[str]] = {}
    for collection in collections:
        groups.setdefault(split_base(collection), []).append(collection)
    return groups


def merge_top_k(hits_per_collection: dict[str, list[dict]], top_k: int) -> list[dict]:
    """Pool every split's hits, tag each with the split it came from, and keep the
    top_k by score."""
    pooled: dict[tuple, dict] = {}
    for collection, hits in hits_per_collection.items():
        for hit in hits:
            key = (hit["file_path"], hit["chunk_index"])
            tagged = {**hit, "source_collection": collection}
            if key not in pooled or (tagged["score"] or 0.0) > (pooled[key]["score"] or 0.0):
                pooled[key] = tagged
    ranked = sorted(pooled.values(), key=lambda h: h["score"] or 0.0, reverse=True)
    return ranked[:top_k]


def build_tasks(entries: list[dict]) -> list[dict]:
    """One task per (project, query, embedder), each tagged with its original
    position so output order is preserved."""
    tasks = []
    for p_idx, entry in enumerate(entries):
        project = entry["project"]
        groups = group_split_collections(entry["collections"])
        for q_idx, query_entry in enumerate(entry["queries"]):
            for g_idx, (base, collections) in enumerate(groups.items()):
                tasks.append({
                    "order": (p_idx, q_idx, g_idx),
                    "project": project,
                    "base": base,
                    "collections": collections,
                    "query": query_entry["query"],
                    "relevant_files": query_entry.get("relevant_files", []),
                    "reference_answer": query_entry.get("reference_answer"),
                    "metadata": query_entry.get("metadata"),
                })
    return tasks


def score_task(task: dict, hits: list[dict], mode: str, embedder: str | None,
               query_time_s: float | None = None) -> dict:
    project, query = task["project"], task["query"]
    scores = evaluate_context_metrics(query, hits, task["reference_answer"])

    log.info(
        "[%s/%s_split] embedder=%s %r → %d hit(s)  query_time=%s  context_precision=%s  context_recall=%s",
        project, task["base"], embedder, query, len(hits),
        f"{query_time_s:.4f}s" if query_time_s is not None else "n/a",
        scores["context_precision"], scores["context_recall"],
    )
    for rank, hit in enumerate(hits, start=1):
        snippet = (hit["content"] or "").replace("\n", " ")[:120]
        log.info(
            "  #%d  score=%.4f  [%s] %s [chunk %s]  %s",
            rank, hit["score"] or 0.0, hit["source_collection"], hit["file_name"], hit["chunk_index"], snippet,
        )

    result = {
        "order": task["order"],
        "project": project,
        "collection": f"{task['base']}_split",
        "source_collections": task["collections"],
        "mode": mode,
        "embedder": embedder,
        "query": query,
        "relevant_files": task["relevant_files"],
        "reference_answer": task["reference_answer"],
        "metadata": task["metadata"],
        "hits": hits,
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
    }
    if query_time_s is not None:
        result["query_time_s"] = query_time_s
    return result


def process_group(rag: RAGPipeline, group_tasks: list[dict], top_k: int) -> list[dict]:
    results = []
    for task in group_tasks:
        hits_per_collection = {}
        mode, embedder = "dense", None
        # Time only retrieval (query embedding, Qdrant search, pooling); the
        # RAGAS evaluation in score_task is deliberately left outside.
        start = time.perf_counter()
        for collection in task["collections"]:
            hits, mode, coll_embedder = route_search(rag, task["project"], collection, task["query"], top_k)
            embedder = embedder or coll_embedder
            hits_per_collection[collection] = hits
        hits = merge_top_k(hits_per_collection, top_k)
        query_time_s = time.perf_counter() - start
        results.append(score_task(task, hits, mode, embedder, query_time_s))
    return results


def write_results(results: list[dict], out_path: Path, timing: dict | None = None) -> Path:
    results.sort(key=lambda r: r["order"])
    for r in results:
        del r["order"]
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({**(timing or {}), "results": results}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log.info("Saved %d result(s) to %s", len(results), out_path)
    return out_path


def run_queries_file(rag: RAGPipeline, queries_file: str, top_k: int, out_dir: Path) -> Path:
    entries = load_queries_config(queries_file)
    run_start = datetime.now()
    tasks = build_tasks(entries)

    # Every collection in a task belongs to the same embedder, so its first
    # collection is enough to decide which model to load.
    groups: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    for task in tasks:
        groups[group_key(rag, {"project": task["project"], "collection": task["collections"][0]})].append(task)

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

    total_query_time_s = sum(r["query_time_s"] for r in results)
    timing = {
        "query_count": len(results),
        "total_query_time_s": total_query_time_s,
        "mean_query_time_s": total_query_time_s / len(results) if results else None,
    }
    log.info(
        "Total retrieval time for %s: %.4fs over %d quer(y/ies) (evaluation excluded)",
        queries_file, total_query_time_s, len(results),
    )

    stem = Path(queries_file).stem
    out_path = out_dir / f"retrieval_{stem}_{run_start.strftime('%Y%m%d_%H%M%S')}.json"
    return write_results(results, out_path, timing)


def rescore_from_results(results_file: Path, top_k: int, out_dir: Path) -> Path:
    """Rebuild pooled top-k rows from an earlier per-split run without querying Qdrant."""
    rows = load_rows(results_file)
    run_start = datetime.now()

    tasks: dict[tuple[str, str, str], dict] = {}
    hits_by_task: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(dict)
    for idx, row in enumerate(rows):
        base = split_base(row["collection"])
        key = (row["project"], row["query"], base)
        if key not in tasks:
            tasks[key] = {
                "order": (idx,),
                "project": row["project"],
                "base": base,
                "collections": [],
                "query": row["query"],
                "relevant_files": row.get("relevant_files", []),
                "reference_answer": row.get("reference_answer"),
                "metadata": row.get("metadata"),
                "mode": row.get("mode", "dense"),
                "embedder": row.get("embedder"),
            }
        task = tasks[key]
        task["collections"].append(row["collection"])
        task["embedder"] = task["embedder"] or row.get("embedder")
        hits_by_task[key][row["collection"]] = row["hits"]

    results = [
        score_task(task, merge_top_k(hits_by_task[key], top_k), task["mode"], task["embedder"])
        for key, task in tasks.items()
    ]
    out_path = out_dir / f"retrieval_queries_rq3_{run_start.strftime('%Y%m%d_%H%M%S')}.json"
    return write_results(results, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "queries_files", nargs="*", default=DEFAULT_QUERIES_FILES,
        help="Queries JSON file(s) to run (default: queries_rq3.json)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of pooled results kept per query and embedder")
    parser.add_argument(
        "--from-results", type=Path,
        help="Re-pool and re-score an earlier per-split results file instead of querying Qdrant",
    )
    args = parser.parse_args()

    out_dir = Path("retrieval_results")
    out_dir.mkdir(exist_ok=True)

    if args.from_results:
        rescore_from_results(args.from_results, args.top_k, out_dir)
        return

    missing = [qf for qf in args.queries_files if not Path(qf).exists()]
    for qf in missing:
        log.warning("Queries file %r not found — skipping", qf)
    queries_files = [qf for qf in args.queries_files if qf not in missing]
    if not queries_files:
        log.error("No queries files to run")
        return

    model_configs = load_model_configs()
    translation_config = load_translation_config()
    rag = RAGPipeline(model_configs, translation_config, dataset_dir=Path("submodules"))

    for qf in queries_files:
        run_queries_file(rag, qf, args.top_k, out_dir)


if __name__ == "__main__":
    main()
