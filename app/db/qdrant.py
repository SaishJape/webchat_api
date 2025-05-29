from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
import logging
import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(host= os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("PORT",6333))

VECTOR_SIZE = 384

def create_collection_if_not_exists(collection_name: str) -> None:
    try:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            logging.error(f"Failed to create collection: {e}")
            raise

def ingest_to_qdrant(collection_name: str, texts: List[str], embeddings: List[List[float]]) -> None:
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

    qdrant.upsert(collection_name=collection_name, points=points)


def query_qdrant(collection_name: str, query_vector: List[float]) -> List[dict]:
    """Query top 3 relevant chunks from Qdrant using cosine similarity."""
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        
        limit=3
    )
    return [hit.dict() for hit in hits]