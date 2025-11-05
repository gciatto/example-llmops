"""
Utility functions for the quiz answer generator project.
"""
import os
import pandas as pd
from typing import List, Dict
from duckduckgo_search import DDGS


def load_questions(csv_path: str = "questions.csv") -> pd.DataFrame:
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
    try:
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
    except Exception as e:
        print(f"Error searching web: {e}")
        return []


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
    
    formatted = "Relevant web search results:\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. {result['title']}\n"
        formatted += f"   {result['snippet']}\n"
        formatted += f"   Source: {result['url']}\n\n"
    return formatted


def load_prompt_template(template_path: str) -> str:
    """
    Load a prompt template from a file.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Template string
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


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
