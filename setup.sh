python3 -m venv .venv
source .venv/bin/activate

pushd docker
docker compose up -d
popd

pip install -r requirements.txt