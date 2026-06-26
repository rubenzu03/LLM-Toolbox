import os

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecturo
from langchain import hub

from vector_store import get_vector_store

vectorstore = get_vector_store()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

CODEBASE_PATH = os.environ.get("CODEBASE_PATH", "codebase")

@tool
def search_codebase(query: str) -> str:
    results = retriever.invoke(query)
    return "\n\n---\n\n".join(
        f"Archivo: {r.metadata['source']}\n{r.page_content}" for r in results
    )

@tool
def read_full_file(filepath: str) -> str:
    full_path = os.path.join(CODEBASE_PATH, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Archivo no encontrado: {filepath}"