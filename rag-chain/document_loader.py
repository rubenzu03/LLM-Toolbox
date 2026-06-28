import os
import contextlib
from pathlib import Path

import pypdf
from langchain_core.documents import Document

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = _PROJECT_ROOT / os.environ.get("DOCUMENTS_PATH", "documents")


def load_documents():
    """Load documents from the specified DOCUMENT_PATH, supporting PDF and TXT files."""
    documents = []
    for item in DOCUMENT_PATH.iterdir():
        if item.suffix == ".pdf":
            with open(item, "rb") as f:
                with contextlib.redirect_stderr(open(os.devnull, "w", encoding="utf-8")):
                    pdf = pypdf.PdfReader(f, strict=False)
                    text = "".join(page.extract_text() for page in pdf.pages)
                documents.append(
                    Document(page_content=text, metadata={"source": item.name})
                )
        elif item.suffix == ".txt":
            text = item.read_text(encoding="utf-8")
            documents.append(
                Document(page_content=text, metadata={"source": item.name})
            )
    return documents
