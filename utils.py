"""
Utility functions for the quiz answer generator project.
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from ddgs import DDGS
from langchain.tools import tool


DIR_ROOT = Path(__file__).parent
FILE_QUESTIONS_CSV = DIR_ROOT / "questions.csv"
DIR_PROMPTS = DIR_ROOT / "prompts"


def load_questions(csv_path: str = FILE_QUESTIONS_CSV) -> pd.DataFrame:
    """
    Load questions from CSV file.
    
    Args:
        csv_path: Path to the questions CSV file
        
    Returns:
        DataFrame with Category, Question, and Weight columns
    """
    df = pd.read_csv(csv_path)
    return df


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo and return top results.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries with 'title', 'url', and 'snippet' keys
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                'title': r.get('title', ''),
                'url': r.get('href', ''),
                'snippet': r.get('body', '')
            }
            for r in results
        ]

def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    Format search results into a readable string.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string with search results
    """
    if not results:
        return "No search results available."
    
    formatted = "Relevant Web search results:\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. [{result['title']}]({result['url']})\n"
        formatted += f"   {result['snippet']}\n"
    return formatted


@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for LLM to search the web and get formatted results.
    This combines search_web and format_search_results into a single tool.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        Markdown-formatted enumeration of search results where,
        for each result, the first line is the title of a Web page with an hyper-ref to the page's URL, 
        and the second line is a snippet/summary of the content.
    """
    results = search_web(query, max_results=max_results)
    return format_search_results(results)


def load_prompt_template(name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        name: Name of the prompt template file inside `prompts/`, without extension (e.g., "basic" for `basic.txt`)

    Returns:
        Template string
    """
    template_path = DIR_PROMPTS / f"{name}.txt"
    return template_path.read_text(encoding='utf-8')


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment.
    
    Returns:
        API key string
        
    Raises:
        ValueError if API key is not set
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-key-here'"
        )
    return api_key
