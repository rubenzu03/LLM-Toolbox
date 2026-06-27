import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from document_loader import DOCUMENT_PATH, load_documents
from ingester import chunk_text
from ollama_model_factory import create_model, get_installed_models
from vector_store import save_to_chroma
from coding_agent.coding_agent import build_coding_agent
from chat_agent import build_chat_agent

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_SESSION"] = "RAG Chain"


def ingest():
    documents = load_documents()
    if not documents:
        print(f"No documents found in {DOCUMENT_PATH}")
        return False
    chunks = chunk_text(documents)
    save_to_chroma(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")
    return True


def main():
    ingest()

    print("Installed Ollama models:")
    models = get_installed_models()
    if not models:
        print("No Ollama models found. Install one with: ollama pull <model>")
        return
    model_name = models[0]
    print(f"Using model: {model_name}")
    llm = create_model(model_name)

    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    chat_agent = build_chat_agent(llm, checkpointer)
    coding_agent = None
    thread_id = "default"

    print("\nChat mode. Commands:  /code  /chat  /new  exit")
    mode = "chat"
    while True:
        query = input(f"({mode}) > ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if query == "/new":
            thread_id = os.urandom(4).hex()
            print(f"New session: {thread_id}")
            continue
        if query == "/chat":
            if chat_agent is None:
                chat_agent = build_chat_agent(llm, checkpointer)
            mode = "chat"
            continue
        if query == "/code":
            if coding_agent is None:
                coding_agent = build_coding_agent(llm, checkpointer)
                print("Coding agent ready.")
            mode = "code"
            continue
        if not query:
            continue

        agent = coding_agent if mode == "code" else chat_agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        print(f"\n{result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
