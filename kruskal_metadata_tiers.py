"""Friedman test comparing the three metadata tiers (curated / half / full)
produced by run_metadata_queries.py, embedder-by-embedder.

Each tier file scores the same queries against the same collections, just with a
different amount of chunk metadata embedded (curated subset, half, or full). For
each embedder, scores are aligned by (project, query) across all three tiers, so
the three tiers are related (paired) samples rather than independent groups.
scipy's Friedman test -- the repeated-measures analogue of Kruskal-Wallis -- is
used for the omnibus comparison. When it is significant, the Conover post-hoc
test for unreplicated blocked data (scikit-posthocs' posthoc_conover_friedman,
Bonferroni-adjusted over the 3 tier pairs) follows up. Conover is preferred over
Nemenyi here: it works on the same within-query ranks as the Friedman statistic
and is more powerful, which matters with only 3 tiers and many blocks (queries).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

METRICS = ["context_precision", "context_recall"]
TIERS = ["curated", "half", "full"]


def load_scores(path: Path) -> dict[str, dict[str, dict[tuple[str, str], float]]]:
    """embedder -> metric -> {(project, query): score}"""
    with path.open() as f:
        data = json.load(f)
    # run_metadata_queries.py writes {"results": [...], <timing>}; older runs are a bare list.
    if isinstance(data, dict):
        data = data["results"]
    scores: dict[str, dict[str, dict[tuple[str, str], float]]] = defaultdict(lambda: defaultdict(dict))
    skipped = defaultdict(int)
    for entry in data:
        key = (entry["project"], entry["query"])
        for metric in METRICS:
            value = entry[metric]
            if value is None:
                skipped[(entry["embedder"], metric)] += 1
                continue
            scores[entry["embedder"]][metric][key] = value
    for (embedder, metric), count in sorted(skipped.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        print(f"note: {embedder} / {metric} has {count} null scores, excluded")
    return scores


def aligned_groups(
    tier_scores: dict[str, dict[str, float]], embedder: str, metric: str
) -> list[list[float]]:
    """One list of scores per tier, restricted to (project, query) keys common to
    all tiers so every tier is scored on the same set of queries."""
    per_tier = [tier_scores[tier][embedder][metric] for tier in TIERS]
    common_keys = set(per_tier[0])
    for d in per_tier[1:]:
        common_keys &= d.keys()
    keys = sorted(common_keys)
    return [[d[k] for k in keys] for d in per_tier]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--curated-file", type=Path,
        default=Path("retrieval_results/retrieval_queries_meta_curated_20260803_132647.json"),
    )
    parser.add_argument(
        "--half-file", type=Path,
        default=Path("retrieval_results/retrieval_queries_meta_half_20260803_133448.json"),
    )
    parser.add_argument(
        "--full-file", type=Path,
        default=Path("retrieval_results/retrieval_queries_meta_full_20260803_132506.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tier_files = {"curated": args.curated_file, "half": args.half_file, "full": args.full_file}
    tier_scores = {tier: load_scores(path) for tier, path in tier_files.items()}

    embedders = sorted(set.union(*(set(s.keys()) for s in tier_scores.values())) - {None})

    for metric in METRICS:
        print(f"\n=== {metric} ===")
        for embedder in embedders:
            if any(embedder not in tier_scores[tier] for tier in TIERS):
                print(f"{embedder:<28} missing from at least one tier, skipping")
                continue

            groups = aligned_groups(tier_scores, embedder, metric)
            n = len(groups[0])
            if n == 0:
                print(f"{embedder:<28} no overlapping queries across tiers, skipping")
                continue
            if all(len(set(g)) == 1 and g[0] == groups[0][0] for g in groups):
                print(f"{embedder:<28} n={n:<4} identical across all tiers, skipping")
                continue

            result = friedmanchisquare(*groups)
            means = {tier: sum(g) / len(g) for tier, g in zip(TIERS, groups)}
            means_str = " ".join(f"mean({tier})={means[tier]:.4f}" for tier in TIERS)
            print(
                f"{embedder:<28} n={n:<4} chi2={result.statistic:>8.3f} p={result.pvalue:.4g} "
                f"p <= {args.alpha}: {result.pvalue <= args.alpha} {means_str}"
            )

            if result.pvalue <= args.alpha:
                # rows = blocks (queries), columns = groups (tiers)
                post_hoc = sp.posthoc_conover_friedman(np.column_stack(groups), p_adjust="bonferroni")
                for i, tier_a in enumerate(TIERS):
                    for j, tier_b in enumerate(TIERS[i + 1:], start=i + 1):
                        bonferroni_p = post_hoc.iloc[i, j]
                        print(
                            f"    conover {tier_a:<8} vs {tier_b:<8} "
                            f"bonferroni_p={bonferroni_p:.4g} "
                            f"significant: {bonferroni_p <= args.alpha}"
                        )


if __name__ == "__main__":
    main()
