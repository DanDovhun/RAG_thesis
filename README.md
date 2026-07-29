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
OPENAI_API_KEY="<API_KEY>"
```
`OPENAI_API_KEY` (or `OPENAI_KEY`) is required for the OpenAI dense embedder and for generating reference answers / scoring retrieval with RAGAS (see below).

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

### Get submodules:
Since `git clone` and `git pull` will only download files and folders directly lined to RAG_thesis repository and only nameplate folders for the submodule, run:
```sh
git submodule update --init --recursive
```

### Installing HuggingFace embedders:
`python3 download_models.py`

## Ollama
https://github.com/mythrantic/ollama-docker

# Running
## Indexing documents
Embed and upload all submodule documents into Qdrant/Pgvector for every enabled model in `embedder_config.toml`:
```sh
python3 main.py
```
Flags:
- `--reset` — delete all Qdrant collections first
- `--translate` — also machine-translate prose documents to English via DeepL and index the translations
- `--sparse-only` — index only BM25 sparse vectors, skipping the (slower) dense embedders

## Generating reference answers
RAGAS's context precision/recall scoring (below) needs a ground-truth answer per query. Auto-generate them once (an LLM reads each query's `relevant_files` and writes a concise answer) and save them into the queries file:
```sh
python3 main.py --generate-answers
```
Flags:
- `--queries-file` — JSON file to read and update in place (default `queries.json`)

This calls the OpenAI API once per query and writes a `reference_answer` field back into each query entry. Queries that already have a `reference_answer` are skipped, so it's safe to re-run after adding new queries. Requires `OPENAI_API_KEY`/`OPENAI_KEY`.

## Running retrieval
Run the queries defined in `queries.json` against their collections and save the results:
```sh
python3 main.py --retrieve
```
Flags:
- `--queries-file` — JSON file of `[{project, collections, queries}, ...]` to run (default `queries.json`)
- `--mode` — retrieval mode: `dense` (default), `bm25`, `splade`, or `hybrid` (RRF fusion of dense + BM25)
- `--top-k` — number of hits to return per query (default `5`)

Each query's retrieved chunks are also scored with RAGAS's standard LLM-judged Context Precision and Context Recall metrics: an LLM compares each chunk's actual text against the query and that query's `reference_answer` (see above) — context_precision rewards relevant chunks ranked earlier, context_recall checks whether all the information in the reference answer was actually retrieved. Both come back as `None` if the query has no `reference_answer` yet or `OPENAI_API_KEY`/`OPENAI_KEY` isn't set. This makes one or more OpenAI API calls per query per collection, so a full run over `queries.json` adds up — expect real cost/latency, not just a local computation. Scores are logged per query and saved alongside the hits. Results are written to `retrieval_results/retrieval_<timestamp>.json`.
