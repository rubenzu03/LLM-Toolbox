import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCUMENTS_PATH = os.environ.get("DOCUMENTS_PATH", "/documents/")

def chunk_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    document = chunks[0]
    print(f"Document content: {document.page_content}")
    print(f"Document metadata: {document.metadata}")
    
    return chunks
