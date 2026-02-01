"""
Web Search Augmentation module
Provides web search context to supplement AI model responses with current information.
Uses Tavily API as primary, Google Custom Search as fallback.
"""
import os
import requests
from typing import Optional

# API Keys from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")


def get_web_search_context(query: str) -> Optional[str]:
    """
    Fetch current web search results to augment LLM queries.
    Tavily first (optimized for LLMs), Google Custom Search fallback.

    Args:
        query: The user's search query

    Returns:
        Formatted search context string, or None if search fails
    """

    # Try Tavily first (optimized for LLMs)
    if TAVILY_API_KEY:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": True
                },
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                search_context = "CURRENT WEB SEARCH RESULTS (use this for up-to-date information):\n\n"

                # Include Tavily's direct answer if available
                if data.get('answer'):
                    search_context += f"DIRECT ANSWER: {data['answer']}\n\n"

                # Include search results
                for i, result in enumerate(data.get('results', [])[:5], 1):
                    title = result.get('title', '')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    search_context += f"{i}. {title}\n   {content}\n"
                    if url:
                        search_context += f"   Source: {url}\n"
                    search_context += "\n"

                return search_context
        except Exception as e:
            print(f"Tavily search error: {e}")

    # Fall back to Google Custom Search
    if GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_SEARCH_ENGINE_ID,
                'q': query,
                'num': 5
            }
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                if items:
                    search_context = "CURRENT WEB SEARCH RESULTS (use this for up-to-date information):\n\n"
                    for i, item in enumerate(items[:5], 1):
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        link = item.get('link', '')
                        search_context += f"{i}. {title}\n   {snippet}\n"
                        if link:
                            search_context += f"   Source: {link}\n"
                        search_context += "\n"
                    return search_context
        except Exception as e:
            print(f"Google search error: {e}")

    return None


def is_web_search_available() -> bool:
    """Check if web search is configured and available."""
    return bool(TAVILY_API_KEY) or bool(GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID)
