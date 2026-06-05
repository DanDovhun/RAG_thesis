# Data Selection/Curation for RAG-LLM

## About
This project investigates how to prepare and curate data for Retrieval-Augmented Generation (RAG) systems, with a focus on **low-resource languages** such as Slovak, Czech, Slovene, and Croatian. The core challenge is that many languages are underrepresented in both LLM training data and embedding models, leading to degraded retrieval accuracy when project documentation is written in those languages.

The research explores three main directions:
- **Metadata enrichment** — whether augmenting document chunks with structured metadata (authors, timestamps, document type, language) improves retrieval relevance and LLM response accuracy.
- **Embedding strategies** — comparing language-specific fine-tuned embedders (BERT, Qwen) against general-purpose models, and evaluating machine translation of source documents into English as a way to leverage stronger English-language embedders.
- **Collection separation** — whether physically partitioning documents into separate vector database collections by category (code, documentation, requirements) reduces false positives and improves retrieval efficiency compared to a single mixed collection.

The pipeline uses [Haystack](https://haystack.deepset.ai/) for document processing and chunking, [Qdrant](https://qdrant.tech/) and PostgreSQL [PGVector](https://github.com/pgvector/pgvector) as vector databases, and evaluates results using the [RAGAS](https://docs.ragas.io/) framework. Dataset documents are sourced from publicly available open-source GitHub repositories under permissive licenses (MIT, BSD, LGPL).

## Articles
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

### Installing HuggingFace embedders:
`python3 download_models.py`
