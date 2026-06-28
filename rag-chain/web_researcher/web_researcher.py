import bs4
import requests
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from ddgs import DDGS

from vector_store import get_vector_store


WEB_AGENT_PROMPT = """You are a web research assistant with tools to search the web and
fetch pages. Only use these tools when you genuinely lack the information.

If the user asks a question you can answer from your own knowledge (e.g. coding questions,
general explanations, conversions), answer directly without calling any tools.

Only use search_web or fetch_webpage when the user asks about current events, specific URLs,
or topics you're unsure about. Use save_web_content_to_knowledge_base only when the user
explicitly asks to save something.
"""


@tool
def search_web(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except requests.RequestException as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("href", "No URL")
        snippet = r.get("body", "No snippet")
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n\n".join(lines)


@tool
def fetch_webpage(url: str) -> str:
    """Fetch and extract readable text content from a URL."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text()
        return text[:8000] + ("..." if len(text) > 8000 else "")
    except requests.RequestException as e:
        return f"Failed to fetch {url}: {e}"


@tool
def save_web_content_to_knowledge_base(url: str, title: str, content: str) -> str:
    """Save web content to the Chroma vector knowledge base for future retrieval."""
    doc = Document(
        page_content=content,
        metadata={"source": url, "title": title},
    )
    vector_store = get_vector_store()
    vector_store.add_documents([doc])
    return f"Saved '{title}' ({url}) to knowledge base."


def build_web_researcher_agent(model: BaseChatModel, checkpointer=None):
    """Build the web researcher agent with the provided model and optional checkpointer."""
    return create_agent(
        model=model,
        tools=[search_web, fetch_webpage, save_web_content_to_knowledge_base],
        system_prompt=WEB_AGENT_PROMPT,
        name="web_researcher",
        checkpointer=checkpointer,
    )
