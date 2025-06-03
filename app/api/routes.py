from fastapi import APIRouter, HTTPException
from app.db.models import QARequest, ScrapeRequest
from app.services.gemini import ask_gemini, analyze_user_query, ask_gemini_enhanced
from app.services.embeddings import get_embeddings, get_question_embedding
from app.utils.common import crawl_website, clean_text, chunk_text, extract_website_name
from app.db.qdrant import ingest_to_qdrant, query_qdrant, enhanced_query_qdrant
import hashlib
import logging

router = APIRouter()

@router.post("/scrape-and-ingest")
async def scrape_and_ingest(req: ScrapeRequest):
    """Scrape a website and ingest the content into Qdrant."""
    try:
        logging.info(f"Starting scrape and ingest for URL: {req.url}")
        
        # Crawl the website
        pages = crawl_website(str(req.url))
        
        if not pages:
            raise HTTPException(status_code=400, detail="No pages could be scraped from the provided URL")
        
        all_chunks = []
        
        # Process each page
        for url, html in pages.items():
            if html and isinstance(html, str) and len(html) > 0:
                cleaned_text = clean_text(html)
                if cleaned_text.strip():
                    chunks = chunk_text(cleaned_text)
                    all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(
                status_code=400, 
                detail="No valid text content found to ingest from the website"
            )

        logging.info(f"Generated {len(all_chunks)} text chunks")
        
        # Generate embeddings
        embeddings = get_embeddings(all_chunks)
        
        # Create collection name based on website name
        collection_name = extract_website_name(str(req.url))
        logging.info(f"Using collection name: {collection_name}")
        
        # Ingest to Qdrant
        ingest_to_qdrant(collection_name, all_chunks, embeddings)

        return {
            "message": "Ingestion successful",
            "collection_name": collection_name,
            "pages_scraped": len(pages),
            "chunks_created": len(all_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in scrape_and_ingest: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/ask-question")
async def ask_question(req: QARequest):
    """Ask a question with enhanced query analysis and retrieval."""
    try:
        logging.info(f"Processing question: {req.question}")
        
        # Analyze user query using Gemini
        query_analysis = analyze_user_query(req.question)
        logging.info(f"Query analysis: {query_analysis}")
        
        # Generate question embedding
        question_embedding = get_question_embedding(req.question)
        
        # Enhanced query with keywords
        results = enhanced_query_qdrant(
            req.collection_name, 
            question_embedding, 
            query_analysis.get('keywords', [req.question])
        )
        
        # Prepare enhanced context
        if results:
            context_chunks = []
            for result in results:
                if result.get('payload') and result['payload'].get('text'):
                    # Add relevance info to context
                    text = result['payload']['text']
                    score = result.get('score', 0)
                    context_chunks.append(f"[Relevance: {score:.3f}] {text}")
            
            context = "\n\n".join(context_chunks)
        else:
            context = ""
        
        # Enhanced prompt for Gemini
        enhanced_answer = ask_gemini_enhanced(context, req.question, query_analysis)
        
        return {
            "answer": enhanced_answer,
            "query_analysis": query_analysis,
            "sources_found": len(results),
            "context_used": bool(context.strip()),
            "top_relevance_score": results[0].get('score', 0) if results else 0
        }
        
    except Exception as e:
        logging.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

