import requests
from qdrant_client import QdrantClient
from pathlib import Path

qdrant_url = "http://localhost:6333/"

snapshots_root = Path("snapshot_backup")
snapshots_root.mkdir(parents=True, exist_ok=True)

client = QdrantClient(url=qdrant_url)

collections = requests.get(qdrant_url + "collections").json().get("result").get("collections")
for coll in collections:
    collection = coll.get("name")
    collection_dir = snapshots_root / collection
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Take a fresh snapshot of the current collection state.
    client.create_snapshot(collection_name=collection)

    # Download every snapshot that exists for this collection, not just the one just created.
    for snapshot in client.list_snapshots(collection_name=collection):
        download_path = collection_dir / snapshot.name
        if download_path.exists():
            continue

        print(f"{qdrant_url}collections/{collection}/snapshots/{snapshot.name}")
        response = requests.get(
            f"{qdrant_url}collections/{collection}/snapshots/{snapshot.name}",
            stream=True
        )
        response.raise_for_status()

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
