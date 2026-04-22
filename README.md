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
```

## Python environment
To create a virtual environment and install dependencies run: `setup.bat` on Windows and `chmod +x setup.sh && ./setup.sh` on MacOS and Linux. 

To then activate the created virtual environment to use the script run `call .venv\Scripts\activate.bat` on Windows or `source .venv/bin/activate` on MacOS and Linux.

### Starting PostgreSQL PGVector and Qdrant Docker Environments:
Both `setup.sh` and `setup.bat` should already installed and launched both containers. In case you want to launch them without the entire setup process (e.g. you just restarted your computer) run:
```sh
pushd docker
docker compose up -d
popd
```