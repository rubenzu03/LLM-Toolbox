import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = Path(__file__).resolve().parent.parent / "vector_store"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PATH),
    )


def save_to_chroma(chunks: list[Document]):
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
    )
    