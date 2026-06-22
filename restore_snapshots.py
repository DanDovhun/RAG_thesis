import requests
import argparse
from pathlib import Path

qdrant_url = "http://localhost:6333/"

snapshots_root = Path("snapshots")

def list_snapshots():
    files = sorted(snapshots_root.iterdir())
    return files

def restore_snapshot(snapshot_path: Path, collection_name: str):
    print(f"Restoring '{collection_name}' from {snapshot_path} ...")
    with open(snapshot_path, "rb") as f:
        response = requests.put(
            f"{qdrant_url}collections/{collection_name}/snapshots/upload",
            params={"priority": "snapshot"},
            files={"snapshot": (snapshot_path.name, f, "application/octet-stream")},
        )
    response.raise_for_status()
    print(f"Done: {response.json()}")

def main():
    parser = argparse.ArgumentParser(description="Restore Qdrant collections from local snapshots")
    parser.add_argument("--snapshot", help="Path to snapshot file (defaults to most recent in snapshots/)")
    parser.add_argument("--collection", required=True, help="Target collection name to restore into")
    args = parser.parse_args()

    if args.snapshot:
        snapshot_path = Path(args.snapshot)
    else:
        available = list_snapshots()
        if not available:
            print("No snapshots found in snapshots/")
            return
        snapshot_path = available[-1]
        print(f"Using most recent snapshot: {snapshot_path}")

    if not snapshot_path.exists():
        print(f"Snapshot file not found: {snapshot_path}")
        return

    restore_snapshot(snapshot_path, args.collection)

if __name__ == "__main__":
    main()
