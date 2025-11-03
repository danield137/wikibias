"""Wikipedia parsing utilities for fetching and processing Wikipedia content."""

import re
import requests
import bs4

# User agent for Wikipedia API requests
UA = {"User-Agent": "wiki-citations/0.1"}


def get_text_and_refs(title: str):
    """Fetch Wikipedia page with inline citations preserved.

    Args:
        title: Wikipedia page title

    Returns:
        tuple: (paragraphs, refs) where:
            - paragraphs is a list of paragraph text strings
            - refs is a list of reference dicts with keys: index, text, url
    """
    r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/html/{title}", headers=UA)
    r.raise_for_status()
    soup = bs4.BeautifulSoup(r.text, "html.parser")
    main = soup.select_one("main") or soup

    # Drop non-prose containers
    for sel in [
        "table",
        "figure",
        "aside",
        "div.hatnote",
        "div.navbox",
        "div.sidebar",
        "table.infobox",
        "table.metadata",
    ]:
        for n in main.select(sel):
            n.decompose()

    # Normalize inline ref markers
    for sup in main.select("sup.reference"):
        sup.replace_with(sup.get_text(strip=True))

    # Extract paragraphs
    paragraphs = []
    secs = main.select("section[data-mw-section-id]")
    if secs:
        for sec in secs:
            for p in sec.find_all("p", recursive=False):
                t = p.get_text(" ", strip=True)
                if t:
                    paragraphs.append(t)
    else:
        for p in main.select(".mw-parser-output > p"):
            t = p.get_text(" ", strip=True)
            if t:
                paragraphs.append(t)

    # Extract references
    refs = []
    for li in main.select("ol.references > li"):
        # Validate this is a proper citation note by checking id
        li_id = li.get("id")
        if not li_id or not isinstance(li_id, str) or not li_id.startswith("cite_note-"):
            continue  # Skip if no proper id attribute
        
        # Extract index from data-mw-footnote-number attribute
        footnote_number = li.get("data-mw-footnote-number")
        reference = False
        try:
            key = footnote_number
        except (ValueError, TypeError):
            continue  # Skip if we can't parse the index

        try:
            int(footnote_number)
            reference = True
        except:
            pass

        text = (li.select_one("span.reference-text") or li).get_text(" ", strip=True)
        ext = None
        # Look for anchor in cite element
        cite = li.select_one("cite")
        if cite:
            for a in cite.select("a[href^='http']"):
                href = a.get("href")
                if href and isinstance(href, str) and "wikipedia.org" not in href:
                    ext = href
                    break
        # Fallback to searching in the entire li if no cite element found
        if not ext:
            for a in li.select("a[href^='http']"):
                href = a.get("href")
                if href and isinstance(href, str) and "wikipedia.org" not in href:
                    ext = href
                    break
        refs.append({"key": key, "text": text, "url": ext, "kind": "reference" if reference else "note"})

    return paragraphs, refs


def split_into_sentences(text: str) -> list:
    """Simple sentence splitter using regex.

    Args:
        text: Text to split into sentences

    Returns:
        list: List of sentence strings
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_citation_indices(sentence: str) -> list:
    """Extract citation markers like [1], [2] from sentence.

    Args:
        sentence: Sentence text containing citation markers

    Returns:
        list: List of citation indices as integers
    """
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, sentence)
    return [int(m) for m in matches]


def map_sentence_citations(paragraph: str, refs: list) -> list:
    """Map each sentence to its citations.

    Args:
        paragraph: Paragraph text
        refs: List of reference dicts from get_text_and_refs

    Returns:
        list: List of dicts with keys 'text' (sentence) and 'citations' (list of ref dicts)
    """
    sentences = split_into_sentences(paragraph)
    sentence_citations = []

    for sentence in sentences:
        citation_indices = extract_citation_indices(sentence)
        citations = []

        for idx in citation_indices:
            # Find matching reference (refs are 1-indexed)
            matching_ref = next((r for r in refs if r["index"] == idx), None)
            if matching_ref:
                citations.append(matching_ref)

        sentence_citations.append({"text": sentence, "citations": citations})

    return sentence_citations


def fetch_citation_content(url: str, max_length: int = 5000) -> dict:
    """Fetch and extract content from a citation URL.

    Args:
        url: The URL to fetch
        max_length: Maximum character length of extracted content

    Returns:
        dict: Contains 'success', 'content', 'title', and 'error' (if failed)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find main content area
        main_content = None
        for selector in ["article", "main", '[role="main"]', ".content", ".article-body"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        # Extract text
        text = main_content.get_text(separator=" ", strip=True)

        # Get title
        title = soup.title.string if soup.title else ""

        # Limit length
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return {"success": True, "content": text, "title": title, "url": url}

    except requests.Timeout:
        return {"success": False, "error": "Request timeout", "url": url}
    except requests.RequestException as e:
        return {"success": False, "error": f"Request error: {str(e)}", "url": url}
    except Exception as e:
        return {"success": False, "error": f"Parsing error: {str(e)}", "url": url}


def fetch_citation_contents(citation_map: dict) -> dict:
    """Fetch content for all citations with URLs.

    Args:
        citation_map: Dict mapping citation indices to reference details

    Returns:
        dict: Mapping from citation index to fetched content
    """
    citation_contents = {}

    for idx, citation in citation_map.items():
        if citation and citation.get("url"):
            print(f"  Fetching citation {idx}: {citation['url'][:50]}...")
            content = fetch_citation_content(citation["url"])
            citation_contents[idx] = content
        else:
            citation_contents[idx] = {"success": False, "error": "No URL available"}

    return citation_contents
