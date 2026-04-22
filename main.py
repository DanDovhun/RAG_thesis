from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

import os

repo_dirs = ["govbox-pro", "navody.digital", "slovensko-sk-api"]
file_types = [".md", ".pdf", ".yaml", ".html", ".txt"]

PGVECTOR_CONN = "postgresql://postgres:postgres@localhost:5433/haystack"


class RAGPipeline:
    def __init__(self):
        self.pgvector_store = PgvectorDocumentStore(
            connection_string=PGVECTOR_CONN,
            embedding_dimension=384,
            vector_function="cosine_similarity",
            recreate_table=False,
        )
        self.qdrant_store = QdrantDocumentStore(
            url="http://localhost:6333",
            index="documents",
            embedding_dim=384,
            similarity="cosine",
        )

        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
        self.indexing_pipeline.add_component("pgvector_writer", DocumentWriter(document_store=self.pgvector_store))
        self.indexing_pipeline.add_component("qdrant_writer", DocumentWriter(document_store=self.qdrant_store))
        self.indexing_pipeline.connect("embedder.documents", "pgvector_writer.documents")
        self.indexing_pipeline.connect("embedder.documents", "qdrant_writer.documents")

        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
        self.query_pipeline.add_component("pgvector_retriever", PgvectorEmbeddingRetriever(document_store=self.pgvector_store))
        self.query_pipeline.add_component("qdrant_retriever", QdrantEmbeddingRetriever(document_store=self.qdrant_store))
        self.query_pipeline.add_component("joiner", DocumentJoiner())
        self.query_pipeline.connect("text_embedder.embedding", "pgvector_retriever.query_embedding")
        self.query_pipeline.connect("text_embedder.embedding", "qdrant_retriever.query_embedding")
        self.query_pipeline.connect("pgvector_retriever.documents", "joiner.documents")
        self.query_pipeline.connect("qdrant_retriever.documents", "joiner.documents")


def read_files(start_doc: Path):
    """
    Function will be used later to add documents into the vector databases
    """
    extensions = []
    start_doc = Path(start_doc)

    for doc in start_doc.rglob("*"):
        extensions.append(doc.suffix) if doc.suffix not in extensions else None
        print(f"{doc}: {doc.is_dir()}") if doc.suffix in file_types or doc.stem == "License" else None

    print(extensions)


for repo in repo_dirs:
    read_files(repo)
