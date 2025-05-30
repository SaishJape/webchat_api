from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Qdrant client
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"), 
    port=int(os.getenv("QDRANT_PORT", 6333))
)

VECTOR_SIZE = 384

def create_collection_if_not_exists(collection_name: str) -> None:
    """Create a Qdrant collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = qdrant.get_collections()
        existing_names = [col.name for col in collections.collections]
        
        if collection_name not in existing_names:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logging.info(f"Created collection: {collection_name}")
        else:
            logging.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logging.error(f"Failed to create collection: {e}")
        raise

def ingest_to_qdrant(collection_name: str, texts: List[str], embeddings: List[List[float]]) -> None:
    """Ingest text chunks and embeddings into Qdrant."""
    create_collection_if_not_exists(collection_name)

    if not texts or not embeddings or len(texts) != len(embeddings):
        logging.warning("Empty or mismatched texts/embeddings. Skipping Qdrant ingestion.")
        return

    points = [
        PointStruct(id=i, vector=embedding, payload={"text": text})
        for i, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    if not points:
        logging.warning("No points to insert into Qdrant.")
        return

    try:
        qdrant.upsert(collection_name=collection_name, points=points)
        logging.info(f"Successfully ingested {len(points)} points to collection {collection_name}")
    except Exception as e:
        logging.error(f"Failed to ingest to Qdrant: {e}")
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
    