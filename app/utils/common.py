import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urljoin, urlparse
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def scrape_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    texts = soup.find_all(["p", "h1", "h2", "h3", "li"])
    return "\n".join(t.get_text(strip=True) for t in texts if t.get_text(strip=True))

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def crawl_website(start_url: str, visited=None) -> dict[str, str]:
    if visited is None:
        visited = set()

    data = {}
    try:
        response = requests.get(start_url, timeout=10)
        if "text/html" not in response.headers.get("Content-Type", ""):
            return data

        html = response.text
        visited.add(start_url)
        data[start_url] = html  

        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = urljoin(start_url, link["href"])
            # Only crawl links from the same domain and that are not visited yet
            if (
                urlparse(full_url).netloc == urlparse(start_url).netloc
                and full_url not in visited
            ):
                nested_data = crawl_website(full_url, visited)
                data.update(nested_data)
    except Exception:
        pass

    return data
