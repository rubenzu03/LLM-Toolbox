from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from vector_store import get_vector_store
from prompts import RAG_PROMPT_TEMPLATE
from web_researcher.web_researcher import search_web, fetch_webpage


@tool
def retrieve_context(query: str) -> str:
    """Search only the LOCAL document store (ingested PDFs and text files). Do NOT use this for current events, news, sports, weather, real-time info, or anything time-sensitive."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    print(f"\nRetrieved {len(docs)} document(s):")
    for d in docs:
        print(f"  - {d.metadata.get('source', 'unknown')}")
    result = "\n\n".join(d.page_content for d in docs)
    result += (
        "\n\n---\n"
        "Note: If the above retrieved content does not answer the question, "
        "use the search_web tool to search the internet instead."
    )
    return result


tools = [retrieve_context]


def build_chat_agent(model: BaseChatModel, checkpointer=None, enable_web_search=False):
    if enable_web_search:
        agent_tools = [search_web, fetch_webpage, retrieve_context]
        prompt = (
            "You have access to search_web (internet search) and retrieve_context (local documents).\n\n"
            "IMPORTANT - Tool selection rules:\n"
            "- For CURRENT EVENTS, NEWS, SPORTS, WEATHER, RECENT INFO: you MUST call search_web.\n"
            "- For questions about ingested LOCAL documents/PDFs: call retrieve_context.\n"
            "- If retrieve_context returns nothing useful, call search_web as backup.\n"
            "- Never say you don't know without first calling search_web for current topics."
        )
    else:
        agent_tools = [retrieve_context]
        prompt = RAG_PROMPT_TEMPLATE + (
            "\n\nWhen you need information to answer the question, use the retrieve_context tool."
        )
    return create_agent(
        model=model,
        tools=agent_tools,
        system_prompt=prompt,
        name="chat_agent",
        checkpointer=checkpointer,
    )
