from pathlib import Path

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.converters import HTMLToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

PGVECTOR_CONN = "postgresql://postgres:postgres@localhost:5433/haystack"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # swap for whichever model you have pulled

PROMPT_TEMPLATE = """
Given the following context documents, answer the question as accurately as possible.
If the answer is not contained in the documents, say so explicitly.

Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:
"""

dataset_dir = Path("submodules")
file_types = [
    ".md", ".pdf", ".yaml", ".html", ".txt", ".rb", ".java", ".json", ".go", ".php", ".js", ".tx", ".cs", ".cshtml", ".csv", ".groovy", ".bat", ".sh"
]

class RAGPipeline:
    def __init__(self):
        # --- Document stores ---
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

        # --- Indexing pipeline ---
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component(
            "embedder", SentenceTransformersDocumentEmbedder()
        )
        self.indexing_pipeline.add_component(
            "pgvector_writer", DocumentWriter(document_store=self.pgvector_store)
        )
        self.indexing_pipeline.add_component(
            "qdrant_writer", DocumentWriter(document_store=self.qdrant_store)
        )
        self.indexing_pipeline.connect("embedder.documents", "pgvector_writer.documents")
        self.indexing_pipeline.connect("embedder.documents", "qdrant_writer.documents")

        # --- Query + generation pipeline ---
        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component(
            "text_embedder", SentenceTransformersTextEmbedder()
        )
        self.query_pipeline.add_component(
            "pgvector_retriever",
            PgvectorEmbeddingRetriever(document_store=self.pgvector_store),
        )
        self.query_pipeline.add_component(
            "qdrant_retriever",
            QdrantEmbeddingRetriever(document_store=self.qdrant_store),
        )
        self.query_pipeline.add_component("joiner", DocumentJoiner())
        self.query_pipeline.add_component(
            "prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE)
        )

        # Embedder → retrievers
        self.query_pipeline.connect(
            "text_embedder.embedding", "pgvector_retriever.query_embedding"
        )
        self.query_pipeline.connect(
            "text_embedder.embedding", "qdrant_retriever.query_embedding"
        )
        # Retrievers → joiner
        self.query_pipeline.connect("pgvector_retriever.documents", "joiner.documents")
        self.query_pipeline.connect("qdrant_retriever.documents", "joiner.documents")
        self.query_pipeline.add_component(
            "llm",
            OllamaGenerator(model=OLLAMA_MODEL, url=OLLAMA_URL),
        )

        # Joiner → prompt builder → LLM
        self.query_pipeline.connect("joiner.documents", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder.prompt", "llm.prompt")

    def index_document(self, filepath: Path) -> None:
        filepath = Path(filepath)
        if filepath.suffix == ".html":
            converter = HTMLToDocument()
        elif filepath.suffix == ".pdf":
            converter = PyPDFToDocument()
        else:
            converter = TextFileToDocument()

        docs = converter.run(sources=[filepath])["documents"]
        if docs:
            self.indexing_pipeline.run({"embedder": {"documents": docs}})

    def query(self, question: str, ollama_model: str = OLLAMA_MODEL) -> str:
        self.query_pipeline.get_component("llm").model = ollama_model
        result = self.query_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
        })
        replies = result["llm"]["replies"]
        return replies[0] if replies else ""

pipeline = RAGPipeline()

def iter_folders(dir):
    extensions_found = []
    dir = Path(dir)

    for doc in dir.iterdir():
        if doc.is_dir():
            print(f"Entering {doc}")
            extensions_found += iter_folders(doc)

        else:
            print(f"Processing: {doc}")
            
    
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

exts = iter_folders(dataset_dir)
