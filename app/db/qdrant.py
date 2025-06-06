from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, OptimizersConfigDiff, CollectionStatus
from typing import List, Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
import json

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client with proper configuration."""
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            timeout=60,
            prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
        )
        # Test connection
        client.get_collections()
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise

# Initialize Qdrant client
try:
    qdrant = get_qdrant_client()
    logger.info("Successfully connected to Qdrant server")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant server: {e}")
    raise

VECTOR_SIZE = 384

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_collection_if_not_exists(collection_name: str) -> None:
    """Create a Qdrant collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = qdrant.get_collections()
        existing_names = [col.name for col in collections.collections]
        
        if collection_name not in existing_names:
            # Create collection with proper configuration
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    on_disk=True  # Store vectors on disk to save memory
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    max_optimization_threads=4,
                    memmap_threshold=20000,
                    indexing_threshold=20000
                ),
                replication_factor=1,  # Single node setup
                write_consistency_factor=1,  # Single node setup
                init_from=None  # Don't initialize from another collection
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} already exists")
            
        # Verify collection was created/accessed
        try:
            # Use get_collection instead of get_collection_info
            collection_info = qdrant.get_collection(collection_name)
            if not collection_info:
                raise Exception(f"Failed to verify collection {collection_name}")
            
            # Check if collection is ready
            if collection_info.status != CollectionStatus.GREEN:
                logger.warning(f"Collection {collection_name} status is {collection_info.status}")
                
            logger.info(f"Collection {collection_name} verified successfully")
            
        except Exception as e:
            logger.error(f"Error verifying collection: {e}")
            # Don't raise here, as the collection might still be usable
            
    except Exception as e:
        logger.error(f"Failed to create/verify collection: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def ingest_to_qdrant(collection_name: str, texts: List[str], embeddings: List[List[float]]) -> None:
    """Ingest text chunks and embeddings into Qdrant."""
    try:
        # Validate inputs
        if not texts or not embeddings:
            raise ValueError("Empty texts or embeddings provided")
            
        if len(texts) != len(embeddings):
            raise ValueError(f"Mismatched lengths: {len(texts)} texts vs {len(embeddings)} embeddings")
            
        # Validate embedding dimensions
        for i, embedding in enumerate(embeddings):
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(f"Invalid embedding dimension at index {i}: {len(embedding)} vs {VECTOR_SIZE}")
        
        # Ensure collection exists
        create_collection_if_not_exists(collection_name)
        
        # Prepare points with metadata
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            if not text.strip():
                continue
                
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": {
                        "chunk_index": i,
                        "text_length": len(text),
                        "created_at": datetime.now().isoformat()
                    }
                }
            )
            points.append(point)
        
        if not points:
            raise ValueError("No valid points to insert")
            
        # Batch process points
        batch_size = 100
        total_ingested = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True,  # Wait for operation to complete
                    ordering=None  # No specific ordering required
                )
                total_ingested += len(batch)
                logger.info(f"Successfully ingested batch {i//batch_size + 1} ({len(batch)} points)")
            except Exception as e:
                logger.error(f"Failed to ingest batch {i//batch_size + 1}: {e}")
                # Continue with next batch instead of raising
                continue
                
        if total_ingested == 0:
            raise Exception("Failed to ingest any points")
            
        logger.info(f"Successfully ingested {total_ingested} points to collection {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to ingest to Qdrant: {e}")
        raise

def query_qdrant(collection_name: str, query_vector: List[float], limit: int = 3) -> List[dict]:
    """Query top relevant chunks from Qdrant using cosine similarity."""
    try:
        hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in hits
        ]
    except Exception as e:
        logging.error(f"Failed to query Qdrant: {e}")
        return []
    
def enhanced_query_qdrant(collection_name: str, query_vector: List[float], keywords: List[str], limit: int = 5) -> List[dict]:
    """Enhanced query with keyword filtering and multiple search strategies."""
    try:
        # Primary vector search
        hits = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit * 2,  # Get more results initially
            with_payload=True
        )
        
        results = []
        for hit in hits:
            score = hit.score
            payload = hit.payload
            
            # Boost score if keywords found in text
            if payload and payload.get('text'):
                text_lower = payload['text'].lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                if keyword_matches > 0:
                    score += (keyword_matches * 0.1)  # Boost score
            
            results.append({
                "id": hit.id,
                "score": score,
                "payload": payload,
                "keyword_matches": keyword_matches if 'keyword_matches' in locals() else 0
            })
        
        # Sort by enhanced score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
        
    except Exception as e:
        logging.error(f"Failed to query Qdrant: {e}")
        return []

# def process_query_with_gemini(user_query: str) -> Dict[str, Any]:
#     """
#     Process user query with Gemini to extract key information and generate search parameters.
#     """
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
        
#         prompt = f"""
#         Analyze the following user query and extract key information for database search:
#         Query: {user_query}
        
#         Please provide:
#         1. Main search terms
#         2. Any specific requirements or constraints
#         3. Context or domain information
        
#         Format the response as a JSON with these fields:
#         - search_terms: list of key terms
#         - requirements: list of specific requirements
#         - context: string describing the context
#         """
        
#         response = model.generate_content(prompt)
#         processed_text = response.text.strip()
#         print("query/quation response from Gemini (processed_text) : ", processed_text)

#         # Try parsing the response as JSON
#         search_params = json.loads(processed_text)
#         print("process_query_with_gemini send to other (search_params) : ", search_params )
#         return search_params

#     except Exception as e:
#         logger.error(f"Error processing query with Gemini: {e}")
#         # Fallback to basic query processing
#         return {
#             "search_terms": [user_query],
#             "requirements": [],
#             "context": ""
#         }

# final ...............................................
# import json
# import re

# def process_query_with_gemini(user_query: str) -> Dict[str, Any]:
#     """
#     Process user query with Gemini to extract key information and generate search parameters.
#     Translates query if needed.
#     """
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
        
#         prompt = f"""
#         You are a multilingual assistant. The user query might be in any language.

#         Step 1: Translate the query to English if it's not already.
#         Step 2: Extract key information for database search.

#         Query: {user_query}

#         Output this JSON structure:
#         {{
#           "search_terms": [ ... ],
#           "requirements": [ ... ],
#           "context": "..." 
#         }}
#         """

#         response = model.generate_content(prompt)
#         processed_text = response.text.strip()
#         print("Gemini raw response:", processed_text)

#         # Clean ```json block if present
#         match = re.search(r'\{.*\}', processed_text, re.DOTALL)
#         if match:
#             cleaned_json = match.group(0)
#             search_params = json.loads(cleaned_json)
#         else:
#             raise ValueError("No valid JSON object found.")

#         print("Parsed search parameters:", search_params)
#         return search_params

#     except Exception as e:
#         logger.error(f"Error processing query with Gemini: {e}")
#         return {
#             "search_terms": [user_query],
#             "requirements": [],
#             "context": ""
#         }



# def generate_response_with_gemini(query: str, search_results: List[dict]) -> str:
#     """
#     Generate a natural language response using Gemini based on search results.
#     """
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
        
#         # Prepare context from search results
#         context = "\n".join([
#             f"Result {i+1}: {result['payload'].get('text', '')}"
#             for i, result in enumerate(search_results)
#         ])
        
#         prompt = f"""
#         Based on the following search results, provide a comprehensive and natural response to the user's query.
        
#         User Query: {query}
        
#         Search Results:
#         {context}
        
#         Please provide a well-structured response that:
#         1. Directly addresses the user's query
#         2. Incorporates relevant information from the search results
#         3. Maintains a natural, conversational tone
#         4. Is clear and concise
#         """
        
#         response = model.generate_content(prompt)
#         print("generate_response_with_gemini response in own function :", response.text)
#         return response.text
        
    # except Exception as e:
    #     logger.error(f"Error generating response with Gemini: {e}")
    #     # Fallback to basic response
    #     return "I apologize, but I'm having trouble generating a proper response at the moment."

# def enhanced_query_with_gemini(collection_name: str, user_query: str, query_vector: List[float], limit: int = 5) -> Dict[str, Any]:
#     """
#     Enhanced query process that uses Gemini for query understanding and response generation.
#     """
#     try:    
#         # Process query with Gemini
#         processed_query = process_query_with_gemini(user_query)

#         print("processed_query (process_query_with_gemini function in called function) : ", processed_query)
        
#         # Perform vector search
#         search_results = query_qdrant(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=limit
#         )
        
#         print("get data from the qdrant database (search_results) : ", search_results)

#         # Generate natural language response
#         # response = generate_response_with_gemini(user_query, search_results)
#         # print("generate_response_with_gemini response in other function :", response)
        
#         return {
#             "processed_query": processed_query,
#             "search_results": search_results
#             # "response": response
#         }
        
#     except Exception as e:
#         logger.error(f"Error in enhanced query with Gemini: {e}")
#         return {
#             "error": str(e),
#             "response": "I apologize, but I encountered an error while processing your query."
#         }






# final ...................................................
# def enhanced_query_with_gemini(
#     collection_name: str,
#     user_query: str,
#     query_vector: List[float],
#     limit: int = 5
# ) -> Dict[str, Any]:
#     """
#     Enhanced query process that uses Gemini for query understanding and response generation.
#     """
#     try:
#         # Step 1: Process query with Gemini
#         processed_query = process_query_with_gemini(user_query)
#         logger.debug(f"Processed query: {processed_query}")

#         # Step 2: Perform vector search in Qdrant
#         search_results = query_qdrant(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=limit
#         )
#         logger.debug(f"Search results from Qdrant: {search_results}")

#         # Step 3: Extract context text from search results
#         context_chunks = []
#         for result in search_results:
#             if result.get("payload") and result["payload"].get("text"):
#                 score = result.get("score", 0)
#                 text = result["payload"]["text"]
#                 context_chunks.append(f"[Relevance: {score:.3f}] {text}")
        
#         print("context_chunks (context text from search results) : ", context_chunks)

#         context_text = "\n\n".join(context_chunks)

#         print("context_text (context text from search results) : ", context_text)

#         return {
#             "processed_query": processed_query,
#             "search_results": search_results,
#             "context_text": context_text
#         }

#     except Exception as e:
#         logger.error(f"Error in enhanced_query_with_gemini: {e}")
#         return {
#             "error": str(e),
#             "response": "I apologize, but I encountered an error while processing your query."
#         }
