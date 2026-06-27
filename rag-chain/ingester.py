import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCUMENTS_PATH = os.environ.get("DOCUMENTS_PATH", "/documents/")


def chunk_text(documents: list[Document]):
    if not documents:
        print("No documents to chunk.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks from {len(documents)} document(s)")

    return chunks
