"""Web scraping utilities for extracting text content from URLs."""

import requests
from bs4 import BeautifulSoup
from typing import List
import time


def scrape_url_content(url: str, timeout: int = 10) -> List[str]:
    """Scrape text content from a URL and return a list of paragraphs.
    
    Args:
        url: The URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        List of text paragraphs extracted from the page
        
    Raises:
        Exception: If the request fails or content cannot be extracted
    """
    try:
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Extract text from paragraph tags and other content elements
        paragraphs = []
        
        # Try to find the main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        
        # Extract paragraphs
        for element in main_content.find_all(['p', 'div', 'section']):
            text = element.get_text(separator=' ', strip=True)
            # Only keep paragraphs with substantial content (more than 50 chars)
            if text and len(text) > 50:
                paragraphs.append(text)
        
        # If we didn't find much content, try a more aggressive extraction
        if len(paragraphs) < 3:
            text = main_content.get_text(separator='\n', strip=True)
            # Split by newlines and filter
            paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 50]
        
        return paragraphs
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch URL {url}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to parse content from {url}: {str(e)}")


def chunk_text_for_llm(paragraphs: List[str], max_chars: int = 8000) -> List[str]:
    """Chunk text paragraphs into sizes suitable for LLM context windows.
    
    Args:
        paragraphs: List of text paragraphs
        max_chars: Maximum characters per chunk (default: 8000 to leave room for prompt)
        
    Returns:
        List of text chunks, each under max_chars
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        para_length = len(paragraph)
        
        # If adding this paragraph would exceed the limit, save current chunk and start new one
        if current_length + para_length > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # If a single paragraph is too long, split it
        if para_length > max_chars:
            # If we have content in current chunk, save it first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split the long paragraph into smaller pieces
            words = paragraph.split()
            temp_chunk = []
            temp_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if temp_length + word_length > max_chars and temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                    temp_chunk = []
                    temp_length = 0
                temp_chunk.append(word)
                temp_length += word_length
            
            if temp_chunk:
                chunks.append(' '.join(temp_chunk))
        else:
            current_chunk.append(paragraph)
            current_length += para_length + 2  # +2 for newlines
    
    # Add any remaining content
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
