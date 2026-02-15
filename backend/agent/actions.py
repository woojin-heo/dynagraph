from .tools import tavily_search, wikipedia_search

# tool mapping
ACTION_TOOLS = {
    "SEARCH_TAVILY": tavily_search,
    "SEARCH_WIKIPEDIA": wikipedia_search,
    "SEARCH_DOCUMENT": None,
}