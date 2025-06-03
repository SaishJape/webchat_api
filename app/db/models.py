from pydantic import BaseModel, HttpUrl

class ScrapeRequest(BaseModel):
    url: HttpUrl

class QARequest(BaseModel):
    question: str
    collection_name: str