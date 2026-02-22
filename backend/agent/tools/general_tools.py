import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

load_dotenv()

@tool
def search_wikipedia(query: str, load_max_docs: int = 3) -> str:
    '''
    Search Wikipedia for the given query and return the results.
    Args:
        query: The query to search for.
        load_max_docs: The maximum number of documents to load.
    Returns:
        A string containing the search results.
    '''
    docs = WikipediaLoader(query=query, load_max_docs=load_max_docs).load()
    formatted_search_results = [
        f'<Document source="{doc.metadata["source"]}">\n{doc.page_content}\n</Document>'
        for doc in docs
    ]
    return "\n\n-----\n\n".join(formatted_search_results)

@tool
def search_tavily(query: str, load_max_docs: int = 3) -> str:
    '''
    Search Tavily for the given query and return the results.
    Args:
        query: The query to search for.
        load_max_docs: The maximum number of documents to load.
    Returns:
        A string containing the search results.
    '''
    tavily = TavilySearch(max_results=load_max_docs, include_images=False, include_image_descriptions=False)
    search_docs = tavily.run(query)
    formatted_search_results = [
        f'<Document source="{doc["url"]}"> title: {doc["title"]}\n{doc["content"]}\n</Document>'
        for doc in search_docs['results']
    ]
    return "\n\n-----\n\n".join(formatted_search_results)