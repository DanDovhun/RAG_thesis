# Data Selection/Curation for RAG-LLM
## Articles:
https://qdrant.tech/articles/neural-search-tutorial/
https://qdrant.tech/articles/what-are-embeddings/

# Setup
## Get submodules
After cloning the repositories and the submodules are empty run: `git submodule update --init`

## .env
Setup a `.env` file containing these variables:
```
QDRANT_URL="http://localhost:6333"
PGVECTOR_CONN = "postgresql://postgres:postgres@localhost:5433/haystack"
POSTGRES_DB="vectordb"
MODELS_DIR="models"
QDRANT_STORAGE="/qdrant_storage"    
PGVECTOR_STORAGE="/pgvector_storage"
DEEPL_API_KEY="<API_KEY>"
```

## Python environment
To create a virtual environment and install dependencies run: `setup.bat` on Windows and `chmod +x setup.sh && ./setup.sh` on MacOS and Linux. 

To then activate the created virtual environment to use the script run `call .venv\Scripts\activate.bat` on Windows or `source .venv/bin/activate` on MacOS and Linux.

## Local Dense Embedders:
Please see [MODELS.md](docs/MODELS.md)

### Starting Database Docker Containers
Both `setup.sh` and `setup.bat` should already installed and launched both containers. In case you want to launch them without the entire setup process (e.g. you just restarted your computer) run:
```sh
docker compose up -d
```

### Installing HuggingFace embedders:
`python3 download_models.py`