"""Run RQ3's category-split query file(s) against Qdrant and score them with the
same RAGAS context-precision/context-recall judge used for the other retrieval
runs (see main.py's --retrieve and run_metadata_queries.py).

RQ3 asks whether separating documents into per-category collections (code,
prose/documentation, config) improves retrieval versus keeping every embedder's
documents in one merged collection. queries_rq3.json holds the same
project/query/relevant_files/reference_answer entries as queries.json, but each
project's "collections" list points at the "_code"/"_prose"/"_config" per-embedder
splits produced by split_legacy_collections_by_filetype.py instead of the single
merged collection per embedder — so this script is a thin wrapper around
run_metadata_queries.py's generic runner (project/query/collection flattening,
one-embedder-at-a-time GPU loading, per-collection search routing by name
suffix), just pointed at that file by default.

Writes retrieval_results/retrieval_queries_rq3_<timestamp>.json (one file per
input queries file), in the same schema as the other retrieval runs, so it drops
straight into the existing retrieval_results/graphs.ipynb analysis.

Usage:
    python3 run_rq3_queries.py [queries_rq3.json ...] [--top-k 5]
"""
import argparse
from pathlib import Path

from run_metadata_queries import RAGPipeline, load_model_configs, load_translation_config, log, run_queries_file

DEFAULT_QUERIES_FILES = ["queries_rq3.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "queries_files", nargs="*", default=DEFAULT_QUERIES_FILES,
        help="Queries JSON file(s) to run (default: queries_rq3.json)",
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
    rag = RAGPipeline(model_configs, translation_config, dataset_dir=Path("submodules"))

    out_dir = Path("retrieval_results")
    out_dir.mkdir(exist_ok=True)

    for qf in queries_files:
        run_queries_file(rag, qf, args.top_k, out_dir)


if __name__ == "__main__":
    main()
