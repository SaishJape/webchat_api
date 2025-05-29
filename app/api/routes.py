from fastapi import APIRouter, HTTPException
from app.db.models import ScrapeRequest
from app.services.gemini import get_embeddings
from app.utils.common import crawl_website, clean_text, chunk_text
from app.db.qdrant import ingest_to_qdrant, query_qdrant, qdrant
import hashlib

router = APIRouter()


@router.post("/scrape-and-ingest")
def scrape_and_ingest(req: ScrapeRequest):
    try:
        pages = crawl_website(req.url)  
        all_chunks = []

        for url, html in pages.items():
            if html and isinstance(html, str) and len(html) > 0:
                cleaned_text = clean_text(html)
                if cleaned_text.strip():
                    chunks = chunk_text(cleaned_text)
                    all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No valid text found to ingest")

        embeddings = get_embeddings(all_chunks)
        collection_name = hashlib.md5(req.url.encode()).hexdigest()

        ingest_to_qdrant(collection_name, all_chunks, embeddings)

        return {"message": "Ingestion successful", "collection_name": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

