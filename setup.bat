@echo off
python -m venv .venv

docker compose up -d

.venv\Scripts\pip install -r requirements.txt
