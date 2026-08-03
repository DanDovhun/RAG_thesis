"""One-off maintenance script: create per-field Qdrant payload indexes on every
collection so document metadata (file_path, model_name, language, ...) is
individually indexed/filterable instead of sitting unindexed inside each
point's payload blob.

Safe to re-run: an already-indexed field is skipped, not recreated.
"""

import logging

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"

# Metadata fields written by rag_pipeline.pipeline (index_document /
# _upsert_sparse_chunks / _index_translation), excluding "content" itself.
FIELD_SCHEMAS = {
    "file_path": PayloadSchemaType.KEYWORD,
    "file_name": PayloadSchemaType.KEYWORD,
    "file_type": PayloadSchemaType.KEYWORD,
    "chunk_index": PayloadSchemaType.INTEGER,
    "chunk_count": PayloadSchemaType.INTEGER,
    "language": PayloadSchemaType.KEYWORD,
    "authors": PayloadSchemaType.KEYWORD,
    "created_at": PayloadSchemaType.DATETIME,
    "modified_at": PayloadSchemaType.DATETIME,
    "model_name": PayloadSchemaType.KEYWORD,
    "model_id": PayloadSchemaType.KEYWORD,
    "source_language": PayloadSchemaType.KEYWORD,
    "translated": PayloadSchemaType.BOOL,
}


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    collections = [c["name"] for c in requests.get(f"{QDRANT_URL}/collections").json()["result"]["collections"]]
    log.info("Found %d collections", len(collections))

    for collection in collections:
        info = client.get_collection(collection)
        existing = set(info.payload_schema or {})
        for field_name, schema in FIELD_SCHEMAS.items():
            if field_name in existing:
                continue
            try:
                client.create_payload_index(
                    collection_name=collection, field_name=field_name, field_schema=schema,
                )
                log.info("[%s] indexed %s (%s)", collection, field_name, schema.value)
            except UnexpectedResponse as e:
                log.warning("[%s] failed to index %s: %s", collection, field_name, e)

    log.info("Done")


if __name__ == "__main__":
    main()
