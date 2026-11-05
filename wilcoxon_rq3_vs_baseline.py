"""Wilcoxon signed-rank test for RQ3: does splitting an embedder's documents into
per-category collections (code / prose / config) improve retrieval over keeping
them in one merged collection?

Baseline: single merged collection per embedder (e.g. "<project>_e5sksmall_hybrid").
Comparison: same embedder's category-split collections from run_rq3_queries.py,
where each query hits the code/prose/config splits, the hits are pooled and only
the overall top-k is scored -- one row per embedder, with collection
"<project>_e5sksmall_hybrid_split".

Runs are matched to embedders by collection-name suffix. A query with zero retrieved
hits has null context_precision/context_recall in the source data; those are
real retrieval failures, not missing data, so they are scored as 0.0 rather
than excluded from the sample.

Groups with no known baseline embedder name (slovakbert, czert) are reported
with a note instead of being silently skipped.

Within a metric, the per-embedder split-vs-merged tests form one family,
Bonferroni-corrected (embedders skipped for missing baseline, missing overlap or
all-zero differences don't count).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

METRICS = ["context_precision", "context_recall"]
# collection-name suffix -> embedder name as recorded in the baseline file's
# "embedder" field, or None if that embedder has no merged-collection baseline
# to compare against.
EMBEDDER_GROUPS = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "qwen3": "Qwen3-Embedding",
    "te3s": "text-embedding-3-small",
    "e5sksmall_hybrid": "e5-sk-small",
    "simcsecz_hybrid": "simcse-small-e-czech",
    "slovakbert": None,
    "czert": None,
}


def load_baseline_scores(paths: list[Path], embedder: str) -> dict[str, dict[tuple, float]]:
    """metric -> {(project, query): score}, for a single embedder's merged collection,
    taken from whichever of the baseline files contains that embedder."""
    data = []
    for path in paths:
        with path.open() as f:
            loaded = json.load(f)
        # main.py now writes {"results": [...], <timing>}; older runs are a bare list.
        data.extend(loaded["results"] if isinstance(loaded, dict) else loaded)
    scores: dict[str, dict[tuple, float]] = defaultdict(dict)
    zero_hit = defaultdict(int)
    for entry in data:
        if entry["embedder"] != embedder:
            continue
        key = (entry["project"], entry["query"])
        for metric in METRICS:
            value = entry[metric]
            if value is None:
                zero_hit[metric] += 1
                value = 0.0
            scores[metric][key] = value
    for metric, count in sorted(zero_hit.items()):
        print(f"note: baseline {embedder} / {metric} has {count} zero-hit queries, scored as 0.0")
    return scores


def load_split_scores(path: Path, suffix: str) -> dict[str, dict[tuple, float]]:
    """metric -> {(project, query): score}, for collections ending in the given suffix."""
    with path.open() as f:
        data = json.load(f)
    # run_rq3_queries.py writes {"results": [...], <timing>}; older runs are a bare list.
    if isinstance(data, dict):
        data = data["results"]
    scores: dict[str, dict[tuple, float]] = defaultdict(dict)
    zero_hit = defaultdict(int)
    for entry in data:
        if not entry["collection"].endswith(suffix):
            continue
        key = (entry["project"], entry["query"])
        for metric in METRICS:
            value = entry[metric]
            if value is None:
                zero_hit[metric] += 1
                value = 0.0
            scores[metric][key] = value
    for metric, count in sorted(zero_hit.items()):
        print(f"note: {suffix} / {metric} has {count} zero-hit queries, scored as 0.0")
    return scores


def paired_samples(a_scores, b_scores):
    keys = sorted(a_scores.keys() & b_scores.keys())
    return [a_scores[k] for k in keys], [b_scores[k] for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baseline-files", type=Path, nargs="+",
        default=[
            Path("retrieval_results/retrieval_20260724_193108.json"),  # xlmr, mbert, qwen3, te3s
            Path("retrieval_results/retrieval_20260803_101516.json"),  # e5-sk-small, simcse-small-e-czech
        ],
    )
    parser.add_argument(
        "--rq3-file", type=Path,
        default=Path("retrieval_results/retrieval_queries_rq3_20261004_083740.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"##### {args.rq3_file} vs {', '.join(map(str, args.baseline_files))} #####")

    # Run every test first so the Bonferroni family size (per metric, across
    # embedders) is known before anything is printed.
    tests: dict[str, dict[str, tuple]] = {}
    for suffix, embedder in EMBEDDER_GROUPS.items():
        if embedder is None:
            continue
        baseline_scores = load_baseline_scores(args.baseline_files, embedder)
        if not baseline_scores:
            continue
        split_scores = load_split_scores(args.rq3_file, f"{suffix}_split")
        tests[suffix] = {}
        for metric in METRICS:
            a, b = paired_samples(split_scores[metric], baseline_scores[metric])
            testable = a and any(x != y for x, y in zip(a, b))
            result = wilcoxon(a, b, zero_method="wilcox") if testable else None
            tests[suffix][metric] = (a, b, result)
    family_sizes = {
        metric: sum(per_metric[metric][2] is not None for per_metric in tests.values())
        for metric in METRICS
    }

    for suffix, embedder in EMBEDDER_GROUPS.items():
        print(f"\n===== embedder: {embedder if embedder else suffix} =====")
        if embedder is None:
            print(f"note: no baseline embedder mapped for '{suffix}', skipping comparison")
            continue
        if suffix not in tests:
            print(f"note: no rows for '{embedder}' in the baseline file(s), skipping comparison")
            continue
        print("\n--- pooled splits vs merged baseline ---")
        for metric in METRICS:
            a, b, result = tests[suffix][metric]
            if not a:
                print(f"{metric:<18} n=0 no overlapping (project, query) pairs, skipping")
                continue
            if result is None:
                print(f"{metric:<18} n={len(a):<4} all differences are zero, skipping")
                continue

            m = family_sizes[metric]
            bonferroni_p = min(result.pvalue * m, 1.0)
            mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
            direction = "split" if mean_a > mean_b else "baseline"
            print(
                f"{metric:<18} n={len(a):<4} statistic={result.statistic:>10.2f} p={result.pvalue:.4g} "
                f"bonferroni_p(m={m})={bonferroni_p:.4g} "
                f"bonferroni_p <= 0.05: {bonferroni_p <= 0.05} "
                f"mean(split)={mean_a:.4f} mean(baseline)={mean_b:.4f} higher={direction}"
            )


if __name__ == "__main__":
    main()
