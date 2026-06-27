import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from vector_store import CHROMA_PATH, COLLECTION_NAME
from prompts import CODE_SYSTEM_PROMPT

CODEBASE_PATH = os.environ.get("CODEBASE_PATH", "codebase")


def _get_retriever():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL_NAME", "nomic-embed-text")
        ),
        persist_directory=str(CHROMA_PATH),
    ).as_retriever(search_kwargs={"k": 5})


@tool
def search_codebase(query: str) -> str:
    """Search the codebase vector store for relevant code snippets."""
    retriever = _get_retriever()
    results = retriever.invoke(query)
    return "\n\n---\n\n".join(
        f"File: {r.metadata['source']}\n{r.page_content}" for r in results
    )


@tool
def read_full_file(filepath: str) -> str:
    """Read the full contents of a file from the codebase."""
    full_path = os.path.join(CODEBASE_PATH, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"


@tool
def list_project_files(extension: str = ""):
    """List project files, optionally filtered by extension (e.g. '.py')."""
    files = []
    for root, _, filenames in os.walk(CODEBASE_PATH):
        for f in filenames:
            if not extension or f.endswith(extension):
                files.append(os.path.relpath(os.path.join(root, f), CODEBASE_PATH))
    return "\n".join(files[:200])


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file in the codebase. Creates directories if needed."""
    full_path = os.path.normpath(os.path.join(CODEBASE_PATH, filepath))
    if not full_path.startswith(os.path.normpath(CODEBASE_PATH)):
        return f"Error: path escapes codebase directory: {filepath}"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} bytes to {filepath}"


tools = [search_codebase, read_full_file, list_project_files, write_file]


def build_coding_agent(model: BaseChatModel, checkpointer=None):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=CODE_SYSTEM_PROMPT,
        name="coding_agent",
        checkpointer=checkpointer,
    )
