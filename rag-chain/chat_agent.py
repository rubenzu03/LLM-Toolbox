from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document

from vector_store import get_vector_store
from prompts import RAG_PROMPT_TEMPLATE


@tool
def retrieve_context(query: str) -> str:
    """Search the document store for relevant context to answer the user's question."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    print(f"\nRetrieved {len(docs)} document(s):")
    for d in docs:
        print(f"  - {d.metadata.get('source', 'unknown')}")
    return "\n\n".join(d.page_content for d in docs)


tools = [retrieve_context]


def build_chat_agent(model: BaseChatModel, checkpointer=None):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=RAG_PROMPT_TEMPLATE
        + "\n\nWhen you need information to answer the question, use the retrieve_context tool.",
        name="chat_agent",
        checkpointer=checkpointer,
    )
