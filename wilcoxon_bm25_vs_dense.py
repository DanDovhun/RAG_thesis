"""Wilcoxon signed-rank test comparing BM25 (sparse) retrieval against each
dense embedder on retrieval quality metrics.

BM25 scores are filtered to the `*_bm25_hybrid` collections only (these files
also contain `*_splade_hybrid` entries, mislabeled with the same mode/embedder
as BM25, so they're excluded by collection name). Dense scores come from
retrieval_20260724_193108.json. Run once per file in BM25_FILES.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

METRICS = ["context_precision", "context_recall"]

DENSE_EMBEDDERS = [
    "text-embedding-3-small",
    "Qwen3-Embedding",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "czert-bert-base-cased",
    "SlovakBERT",
]


def load_dense_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """embedder -> metric -> {(project, query): score}"""
    with path.open() as f:
        data = json.load(f)
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    skipped = defaultdict(int)
    for entry in data:
        key = (entry["project"], entry["query"])
        for metric in METRICS:
            value = entry[metric]
            if value is None:
                skipped[(entry["embedder"], metric)] += 1
                continue
            scores[entry["embedder"]][metric][key] = value
    for (embedder, metric), count in sorted(skipped.items()):
        print(f"note: {embedder} / {metric} has {count} null scores, excluded")
    return scores


def load_bm25_scores(path: Path) -> dict[str, dict[str, float]]:
    """metric -> {(project, query): score}, restricted to true BM25 collections."""
    with path.open() as f:
        data = json.load(f)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    skipped = defaultdict(int)
    for entry in data:
        if not entry["collection"].endswith("_bm25_hybrid"):
            continue
        key = (entry["project"], entry["query"])
        for metric in METRICS:
            value = entry[metric]
            if value is None:
                skipped[metric] += 1
                continue
            scores[metric][key] = value
    for metric, count in sorted(skipped.items()):
        print(f"note: BM25 / {metric} has {count} null scores, excluded")
    return scores


def paired_samples(
    a_scores: dict[str, float], b_scores: dict[str, float]
) -> tuple[list[float], list[float]]:
    common_keys = a_scores.keys() & b_scores.keys()
    keys = sorted(common_keys)
    return [a_scores[k] for k in keys], [b_scores[k] for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-file", type=Path,
        default=Path("retrieval_results/retrieval_20260724_193108.json"),
        help="Dense retrieval results JSON file (must contain all DENSE_EMBEDDERS)",
    )
    parser.add_argument(
        "--bm25-files", type=Path, nargs="+",
        default=[
            Path("retrieval_results/retrieval_20260725_154244.json"),
            Path("retrieval_results/retrieval_20260725_154255.json"),
        ],
        help="One or more BM25/SPLADE hybrid results JSON files to compare against --dense-file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_scores = load_dense_scores(args.dense_file)

    for name in DENSE_EMBEDDERS:
        if name not in dense_scores:
            raise SystemExit(f"Embedder {name!r} not found in {args.dense_file}")

    for bm25_file in args.bm25_files:
        bm25_scores = load_bm25_scores(bm25_file)
        print(f"\n##### BM25_FILE = {bm25_file} #####")
        for metric in METRICS:
            print(f"\n=== {metric} ===")
            for embedder in DENSE_EMBEDDERS:
                a, b = paired_samples(bm25_scores[metric], dense_scores[embedder][metric])
                diffs_nonzero = any(x != y for x, y in zip(a, b))
                if not diffs_nonzero:
                    print(f"{'BM25':>6} vs {embedder:<28} n={len(a):<4} all differences are zero, skipping")
                    continue

                result = wilcoxon(a, b, zero_method="wilcox")
                mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
                direction = "BM25" if mean_a > mean_b else embedder
                print(
                    f"{'BM25':>6} vs {embedder:<28} "
                    f"n={len(a):<4} statistic={result.statistic:>10.2f} p={result.pvalue:.4g} "
                    f"p <= 0.05: {result.pvalue <= 0.05} "
                    f"mean(BM25)={mean_a:.4f} mean({embedder})={mean_b:.4f} higher={direction}"
                )


if __name__ == "__main__":
    main()
