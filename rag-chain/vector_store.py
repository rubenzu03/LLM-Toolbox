from pathlib import Path
import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from document_loader import load_documents

CHROMA_PATH = Path(__file__).resolve().parent.parent / "vector_store"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_FUNCTION = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        os.remove(CHROMA_PATH)
    
    db = Chroma.from_documents(chunks, EMBEDDING_FUNCTION, collection_name=COLLECTION_NAME, persist_directory=CHROMA_PATH)
    db.persist()
    