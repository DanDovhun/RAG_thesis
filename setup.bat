@echo off
python -m venv .venv
call .venv\Scripts\activate.bat

pushd docker
docker compose up -d
popd

pip install -r requirements.txt
