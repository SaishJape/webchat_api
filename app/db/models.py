from pydantic import BaseModel

class ScrapeRequest(BaseModel):
    url: str

class QARequest(BaseModel):
    question: str
    collection_name: str
    history: list[tuple[str, str]] = []  
