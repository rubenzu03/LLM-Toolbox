import os
import pypdf
from langchain_core.documents import Document

DOCUMENT_PATH = os.environ.get("DOCUMENTS_PATH", "/documents/")

def load_documents():
    documents = []
    for filename in os.listdir(DOCUMENT_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DOCUMENT_PATH, filename)
            with open(file_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                documents.append(Document(page_content=text, metadata={"source": filename}))
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENT_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
                
    return documents