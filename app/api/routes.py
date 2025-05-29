from fastapi import APIRouter, HTTPException
from app.db.models import QARequest, ScrapeRequest
from app.services.gemini import ask_gemini
from app.services.embeddings import get_embeddings, get_question_embedding
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


@router.post("/ask-question")
def ask_question(req: QARequest):
    try:
        question_embedding = get_question_embedding(req.question)
        results = query_qdrant(req.collection_name, question_embedding)

        context = "\n\n".join([res['payload']['text'] for res in results]) if results else ""
        
        answer = ask_gemini(context, req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def qdrant_search(collection_name: str, query_embedding: list[float], top_k: int = 5):
    # Perform vector similarity search in Qdrant
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    # Extract text chunks from payload
    return [
        {
            "id": hit.id,
            "text": hit.payload.get("text", "")  # Ensure text is stored under 'text' key
        }
        for hit in search_result
        if hit.payload and "text" in hit.payload
    ]

