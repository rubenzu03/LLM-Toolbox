from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import os

CODEBASE_PATH = os.environ.get("CODEBASE_PATH", "codebase")
EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".html": Language.HTML,
}


def load_codebase_files():
    codebase_files = []
    for extension, lang in EXTENSIONS.items():
        loader = DirectoryLoader(
            CODEBASE_PATH,
            glob=f"**/*{extension}",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            exclude=[
                "**/node_modules/**",
                "**/.git/**",
                "**/venv/**",
                "**/__pycache__/**",
            ],
        )
        for doc in loader.load():
            doc.metadata["language"] = lang
            doc.metadata["filename"] = os.path.basename(doc.metadata["source"])
            codebase_files.append(doc)
    return codebase_files

def split_documents(documents):
    chunks = []
    for doc in documents:
        lang = doc.metadata.get("language")
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language(lang), chunk_size=1000, chunk_overlap=200
            )
        except ValueError:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks.extend(splitter.split_documents([doc]))
    return chunks