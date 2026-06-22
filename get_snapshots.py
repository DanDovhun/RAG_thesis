import requests
from qdrant_client import QdrantClient
from pathlib import Path
from datetime import datetime

qdrant_url = "http://localhost:6333/"

snapshots_root = Path("snapshots")
snapshots_root.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
download_path = snapshots_root / timestamp
#download_path.mkdir(parents=True, exist_ok=True)

client = QdrantClient(url=qdrant_url)

collections = requests.get(qdrant_url + "collections").json().get("result").get("collections")
for coll in collections:
    collection = coll.get("name")
    snapshot = client.create_snapshot(collection_name=collection)
    print(snapshot)
    print(f"{qdrant_url}collections/{collection}/snapshots/{snapshot}")
    response = requests.get(
        f"{qdrant_url}collections/{collection}/snapshots/{snapshot.name}",
        stream=True
    )
    response.raise_for_status()

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)