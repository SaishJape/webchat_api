from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
import logging

qdrant = QdrantClient(host="localhost", port=6333)

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
