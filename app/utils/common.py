import hashlib
import re
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urljoin, urlparse
import warnings
import logging
from typing import Dict

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def scrape_url(url: str) -> str:
    """Scrape content from a single URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return ""

def clean_text(html: str) -> str:
    """Clean HTML and extract meaningful text."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from meaningful tags
        texts = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "span"])
        clean_texts = []
        
        for element in texts:
            text = element.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short texts
                clean_texts.append(text)
        
        return "\n".join(clean_texts)
    except Exception as e:
        logging.error(f"Failed to clean text: {e}")
        return ""

def chunk_text(text: str) -> list[str]:
    """Split text into chunks for embedding."""
    try:
        if not text.strip():
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        logging.error(f"Failed to chunk text: {e}")
        return []

def crawl_website(start_url: str, max_pages: int = 10) -> Dict[str, str]:
    """Crawl a website starting from the given URL."""
    visited = set()
    data = {}
    urls_to_visit = [start_url]
    
    while urls_to_visit and len(visited) < max_pages:
        current_url = urls_to_visit.pop(0)
        
        if current_url in visited:
            continue
            
        try:
            logging.info(f"Crawling: {current_url}")
            html = scrape_url(current_url)
            
            if not html:
                continue
                
            # Check if it's HTML content
            if "text/html" not in requests.head(current_url, timeout=5).headers.get("Content-Type", ""):
                continue
                
            visited.add(current_url)
            data[current_url] = html
            
            # Extract links for further crawling
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(current_url, link["href"])
                
                # Only crawl links from the same domain
                if (urlparse(full_url).netloc == urlparse(start_url).netloc 
                    and full_url not in visited 
                    and full_url not in urls_to_visit
                    and len(visited) < max_pages):
                    urls_to_visit.append(full_url)
                    
        except Exception as e:
            logging.error(f"Error crawling {current_url}: {e}")
            continue
    
    logging.info(f"Crawled {len(data)} pages")
    return data

def extract_website_name(url: str) -> str:
    """Extract a clean website name from URL to use as collection name."""
    try:
        # Parse the URL to get the domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Remove common TLD extensions and keep the main domain name
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            # Take the main domain name (before the TLD)
            website_name = domain_parts[0]
        else:
            website_name = domain
        
        # Clean the name: remove special characters and replace with underscores
        website_name = re.sub(r'[^a-zA-Z0-9]', '_', website_name)
        
        # Ensure it's not empty and has reasonable length
        if not website_name or len(website_name) < 2:
            # Fallback to hash if extraction fails
            return hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Limit length and ensure it starts with a letter (Qdrant collection name requirements)
        website_name = website_name[:20]  # Limit to 20 characters
        if not website_name[0].isalpha():
            website_name = 'site_' + website_name
            
        return website_name
        
    except Exception as e:
        logging.warning(f"Failed to extract website name from {url}: {e}")
        # Fallback to hash
        return hashlib.md5(url.encode()).hexdigest()[:8]