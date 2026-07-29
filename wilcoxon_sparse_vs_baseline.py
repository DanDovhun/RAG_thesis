"""Wilcoxon signed-rank test comparing sparse (BM25/SPLADE hybrid) retrieval runs
against a sparse baseline, arm-by-arm (BM25 vs BM25, SPLADE vs SPLADE).

Every entry is labeled embedder="BM25" regardless of arm, so arms are split by
collection suffix (`_bm25_hybrid` / `_splade_hybrid`) instead.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

METRICS = ["context_precision", "context_recall"]
ARM_SUFFIXES = {"BM25": "_bm25_hybrid", "SPLADE": "_splade_hybrid"}


def load_sparse_scores(path: Path, suffix: str) -> dict[str, dict[str, float]]:
    with path.open() as f:
        data = json.load(f)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    skipped = defaultdict(int)
    for entry in data:
        if not entry["collection"].endswith(suffix):
            continue
        key = (entry["project"], entry["query"])
        for metric in METRICS:
            value = entry[metric]
            if value is None:
                skipped[metric] += 1
                continue
            scores[metric][key] = value
    for metric, count in sorted(skipped.items()):
        print(f"note: {suffix} / {metric} has {count} null scores, excluded")
    return scores


def paired_samples(a_scores, b_scores):
    keys = sorted(a_scores.keys() & b_scores.keys())
    return [a_scores[k] for k in keys], [b_scores[k] for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-file", type=Path,
        default=Path("retrieval_results/retrieval_20260725_115511.json"),
    )
    parser.add_argument(
        "--comparison-files", type=Path, nargs="+",
        default=[
            Path("retrieval_results/retrieval_20260725_154244.json"),
            Path("retrieval_results/retrieval_20260725_154255.json"),
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for comparison_file in args.comparison_files:
        print(f"\n##### {comparison_file} vs {args.baseline_file} #####")
        for metric in METRICS:
            print(f"\n=== {metric} ===")
            for arm, suffix in ARM_SUFFIXES.items():
                a_scores = load_sparse_scores(comparison_file, suffix)
                b_scores = load_sparse_scores(args.baseline_file, suffix)
                a, b = paired_samples(a_scores[metric], b_scores[metric])
                if not a:
                    print(f"{arm:<8} n=0 no overlapping (project, query) pairs, skipping")
                    continue
                if all(x == y for x, y in zip(a, b)):
                    print(f"{arm:<8} n={len(a):<4} all differences are zero, skipping")
                    continue

                result = wilcoxon(a, b, zero_method="wilcox")
                mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
                direction = comparison_file.stem if mean_a > mean_b else args.baseline_file.stem
                print(
                    f"{arm:<8} n={len(a):<4} statistic={result.statistic:>10.2f} p={result.pvalue:.4g} "
                    f"p <= 0.05: {result.pvalue <= 0.05} "
                    f"mean(new)={mean_a:.4f} mean(baseline)={mean_b:.4f} higher={direction}"
                )


if __name__ == "__main__":
    main()
