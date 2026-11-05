"""Wilcoxon signed-rank test comparing dense retrieval runs against the
retrieval_20260724_193108.json baseline, embedder-by-embedder.

Each file in COMPARISON_FILES shares the same 6 embedders as BASELINE_FILE, so
scores are paired by (project, query) within each embedder and compared
directly against the corresponding embedder's scores in the baseline. Within a
comparison file and metric, the per-embedder tests form one family,
Bonferroni-corrected (embedders skipped for all-zero differences don't count).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

METRICS = ["context_precision", "context_recall"]

EMBEDDERS = [
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
    # Newer runs write {"results": [...], <timing>}; older runs are a bare list.
    if isinstance(data, dict):
        data = data["results"]
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


def paired_samples(
    a_scores: dict[str, float], b_scores: dict[str, float]
) -> tuple[list[float], list[float]]:
    common_keys = a_scores.keys() & b_scores.keys()
    keys = sorted(common_keys)
    return [a_scores[k] for k in keys], [b_scores[k] for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-file", type=Path,
        default=Path("retrieval_results/retrieval_20260724_193108.json"),
        help="Baseline dense retrieval results JSON file (must contain all EMBEDDERS)",
    )
    parser.add_argument(
        "--comparison-files", type=Path, nargs="+",
        default=[
            Path("retrieval_results/retrieval_20260725_154313.json"),
            Path("retrieval_results/retrieval_20260725_184058.json"),
        ],
        help="One or more dense results JSON files to compare against --baseline-file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_scores = load_dense_scores(args.baseline_file)
    for name in EMBEDDERS:
        if name not in baseline_scores:
            raise SystemExit(f"Embedder {name!r} not found in {args.baseline_file}")

    for comparison_file in args.comparison_files:
        comparison_scores = load_dense_scores(comparison_file)
        print(f"\n##### {comparison_file} vs {args.baseline_file} #####")
        for metric in METRICS:
            print(f"\n=== {metric} ===")
            tests = []
            for embedder in EMBEDDERS:
                if embedder not in comparison_scores:
                    print(f"  {embedder!r} not found in {comparison_file}, skipping")
                    continue
                a, b = paired_samples(
                    comparison_scores[embedder][metric], baseline_scores[embedder][metric]
                )
                diffs_nonzero = any(x != y for x, y in zip(a, b))
                result = wilcoxon(a, b, zero_method="wilcox") if diffs_nonzero else None
                tests.append((embedder, a, b, result))
            m = sum(result is not None for *_, result in tests)

            for embedder, a, b, result in tests:
                if result is None:
                    print(f"{embedder:<28} n={len(a):<4} all differences are zero, skipping")
                    continue
                bonferroni_p = min(result.pvalue * m, 1.0)
                mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
                direction = comparison_file.stem if mean_a > mean_b else args.baseline_file.stem
                print(
                    f"{embedder:<28} "
                    f"n={len(a):<4} statistic={result.statistic:>10.2f} p={result.pvalue:.4g} "
                    f"bonferroni_p(m={m})={bonferroni_p:.4g} "
                    f"bonferroni_p <= 0.05: {bonferroni_p <= 0.05} "
                    f"mean(new)={mean_a:.4f} mean(baseline)={mean_b:.4f} higher={direction}"
                )


if __name__ == "__main__":
    main()
