"""Wilcoxon signed-rank test comparing general-purpose vs. Czech/Slovak-specific
dense embedders on retrieval quality metrics.

Group A (general-purpose): text-embedding-3-small, Qwen3-Embedding,
    bert-base-multilingual-cased, xlm-roberta-base
Group B (language-specific): czert-bert-base-cased, SlovakBERT

For every (A, B) pair and every metric, the two embedders' per-query scores
are aligned by (project, query) and compared with scipy's Wilcoxon
signed-rank test. The 8 pairs within a metric form one family, Bonferroni-corrected
(pairs skipped for having all-zero differences don't count towards it).
"""
import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

from scipy.stats import wilcoxon

METRICS = ["context_precision", "context_recall"]

GENERAL_EMBEDDERS = [
    "text-embedding-3-small",
    "Qwen3-Embedding",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
]
SPECIFIC_EMBEDDERS = [
    "czert-bert-base-cased",
    "SlovakBERT",
]


def load_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
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
    scores: dict[str, dict[str, dict[str, float]]],
    embedder_a: str,
    embedder_b: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    a_scores = scores[embedder_a][metric]
    b_scores = scores[embedder_b][metric]
    common_keys = a_scores.keys() & b_scores.keys()
    if len(common_keys) < len(a_scores):
        missing = len(a_scores) - len(common_keys)
        print(f"  warning: {missing} queries missing overlap for {embedder_a} vs {embedder_b}")
    keys = sorted(common_keys)
    return [a_scores[k] for k in keys], [b_scores[k] for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file", type=Path,
        default=Path("retrieval_results/retrieval_20260724_193108.json"),
        help="Retrieval results JSON file to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.results_file)

    available = set(scores.keys())
    for name in GENERAL_EMBEDDERS + SPECIFIC_EMBEDDERS:
        if name not in available:
            raise SystemExit(f"Embedder {name!r} not found in {args.results_file}. Found: {sorted(available)}")

    for metric in METRICS:
        print(f"\n=== {metric} ===")
        tests = []
        for general, specific in product(GENERAL_EMBEDDERS, SPECIFIC_EMBEDDERS):
            a, b = paired_samples(scores, general, specific, metric)
            diffs_nonzero = any(x != y for x, y in zip(a, b))
            result = wilcoxon(a, b, zero_method="wilcox") if diffs_nonzero else None
            tests.append((general, specific, a, b, result))
        m = sum(result is not None for *_, result in tests)

        for general, specific, a, b, result in tests:
            if result is None:
                print(f"{general:>28} vs {specific:<20} n={len(a):<4} all differences are zero, skipping")
                continue
            bonferroni_p = min(result.pvalue * m, 1.0)
            mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
            direction = general if mean_a > mean_b else specific
            print(
                f"{general:>28} vs {specific:<20} "
                f"n={len(a):<4} statistic={result.statistic:>10.2f} p={result.pvalue:.4g} "
                f"bonferroni_p(m={m})={bonferroni_p:.4g} "
                f'bonferroni_p <= 0.05: {bonferroni_p <= 0.05} '
                f"mean({general})={mean_a:.4f} mean({specific})={mean_b:.4f} higher={direction}"
            )


if __name__ == "__main__":
    main()
