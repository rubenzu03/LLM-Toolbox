import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from document_loader import DOCUMENT_PATH, load_documents
from ingester import chunk_text
from ollama_model_factory import create_model, get_installed_models
from vector_store import get_vector_store, save_to_chroma
from coding_agent.coding_agent import build_coding_agent
from prompts import RAG_PROMPT_TEMPLATE

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_SESSION"] = "RAG Chain"


def build_chain(model_name: str | None = None):
    if model_name is None:
        models = get_installed_models()
        if not models:
            print("No Ollama models found. Install one with: ollama pull <model>")
            return None, None
        model_name = models[0]
        print(f"Using model: {model_name}")

    llm = create_model(model_name)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs: list[Document]):
        print(f"\nRetrieved {len(docs)} document(s):")
        for d in docs:
            print(f"  - {d.metadata.get('source', 'unknown')}")
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, llm


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

    chain, llm = build_chain()
    if chain is None:
        return

    coding_agent = None

    print("\nRAG Query. Commands: /code  /rag  exit")
    mode = "rag"
    while True:
        query = input(f"({mode}) > ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if query == "/rag":
            mode = "rag"
            continue
        if query == "/code":
            if coding_agent is None:
                coding_agent = build_coding_agent(llm)
                print("Coding agent ready.")
            mode = "code"
            continue
        if not query:
            continue

        if mode == "code":
            result = coding_agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            print(f"\n{result['messages'][-1].content}\n")
        else:
            result = chain.invoke(query)
            print(f"\nAnswer: {result}\n")


if __name__ == "__main__":
    main()
