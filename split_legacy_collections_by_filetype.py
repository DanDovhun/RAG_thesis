"""Split Qdrant collections into per-file-type collections.

Eligible source collections are ones with no "_en"/"_sparse" suffix (and not
already a "_code"/"_prose"/"_config" split target) — this covers both the
"legacy" collections created before the "_hybrid" naming convention existed
(bare "<name>", no suffix at all) and the live "<name>_hybrid" collections.

For each eligible collection "<name>", every point is moved (copied then removed
from the source) into exactly one of "<name>_code", "<name>_prose", "<name>_config"
based on its `file_type` payload field (the lower-cased file extension
main.py/pipeline.py stamps on every chunk at index time):

    prose:  .md, .pdf, .txt, .csv
    code:   .html, .rb, .java, .go, .php, .js, .tx, .cs, .cshtml, .groovy, .bat, .sh
    config: .yaml, .json, and anything not listed above (catch-all)

Destination collections are created (if missing) with the exact same vector config
as their source, so this works uniformly across the different embedding dims/aliases
(czert, mbert, qwen3, slovakbert, te3s, xlmr, ...) without needing to resolve each
source alias back to a model config — several of these aliases (czert, slovakbert)
belong to embedders no longer in embedder_config.toml.

Every point's file_type maps into exactly one bucket, so a fully-processed source
collection ends up empty. By design this script leaves that empty collection in
place (does not delete it).

Migration is done batch-by-batch and only deletes a batch of source points after its
upsert into the destination succeeds, so re-running after an interruption is safe:
already-moved points are simply absent from the next scroll of the source.

Usage:
    python split_legacy_collections_by_filetype.py [--dry-run] [--batch-size 256]
"""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag_pipeline import QDRANT_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROSE_EXTS = {".md", ".pdf", ".txt", ".csv"}
CODE_EXTS = {
    ".html", ".rb", ".java", ".go", ".php", ".js", ".tx", ".cs",
    ".cshtml", ".groovy", ".bat", ".sh",
}
# CONFIG_EXTS is deliberately not a closed set — ".yaml"/".json" plus every
# extension not claimed by PROSE_EXTS/CODE_EXTS falls here (see classify()).
CONFIG_EXTS = {".yaml", ".json"}

# "_hybrid" is a usable source suffix (the live collections main.py/pipeline.py
# populate) — only "_en" (machine-translated variants) and "_sparse" are excluded.
EXCLUDED_SUFFIXES = ("_en", "_sparse")
BUCKET_SUFFIXES = ("_code", "_prose", "_config")


def is_eligible_source(name: str) -> bool:
    if name.endswith(BUCKET_SUFFIXES):
        return False
    return not name.endswith(EXCLUDED_SUFFIXES)


def classify(file_type: str | None) -> str:
    ext = (file_type or "").lower()
    if ext in PROSE_EXTS:
        return "prose"
    if ext in CODE_EXTS:
        return "code"
    return "config"  # .yaml, .json, and any other/unrecognized extension


def ensure_target_collection(client: QdrantClient, source: str, target: str) -> None:
    if client.collection_exists(target):
        return
    info = client.get_collection(source)
    params = info.config.params
    log.info("Creating collection %r (mirroring %r vector config)", target, source)
    client.create_collection(
        collection_name=target,
        vectors_config=params.vectors,
        sparse_vectors_config=params.sparse_vectors or None,
    )


def migrate_collection(client: QdrantClient, source: str, batch_size: int, dry_run: bool) -> dict[str, int]:
    targets = {bucket: f"{source}_{bucket}" for bucket in ("code", "prose", "config")}
    ensured: set[str] = set()  # buckets whose target collection is confirmed to exist

    moved = {"code": 0, "prose": 0, "config": 0}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        buckets: dict[str, list[PointStruct]] = {"code": [], "prose": [], "config": []}
        for point in points:
            bucket = classify((point.payload or {}).get("file_type"))
            buckets[bucket].append(
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
            )

        if dry_run:
            for bucket, pts in buckets.items():
                moved[bucket] += len(pts)
        else:
            for bucket, pts in buckets.items():
                if pts:
                    if bucket not in ensured:
                        ensure_target_collection(client, source, targets[bucket])
                        ensured.add(bucket)
                    client.upsert(collection_name=targets[bucket], points=pts)
                    moved[bucket] += len(pts)
            client.delete(collection_name=source, points_selector=[p.id for p in points])

        if offset is None:
            break

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing/deleting anything")
    parser.add_argument("--batch-size", type=int, default=256, help="Points per scroll/upsert/delete batch")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    all_names = [c.name for c in client.get_collections().collections]
    sources = sorted(n for n in all_names if is_eligible_source(n))

    log.info("%d eligible source collection(s) to split (of %d total)", len(sources), len(all_names))
    if args.dry_run:
        log.info("DRY RUN — no data will be written or deleted")

    grand_total = {"code": 0, "prose": 0, "config": 0}
    for name in sources:
        moved = migrate_collection(client, name, args.batch_size, args.dry_run)
        log.info(
            "[%s] code=%d  prose=%d  config=%d",
            name, moved["code"], moved["prose"], moved["config"],
        )
        for bucket in grand_total:
            grand_total[bucket] += moved[bucket]

    log.info(
        "Done. Total moved — code=%d  prose=%d  config=%d",
        grand_total["code"], grand_total["prose"], grand_total["config"],
    )


if __name__ == "__main__":
    main()
