import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = Path(__file__).resolve().parent.parent / "vector_store"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_collection")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "nomic-embed-text")

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PATH),
    )


def save_to_chroma(chunks: list[Document]):
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
    )
