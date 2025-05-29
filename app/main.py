from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Web Scraping Q&A Chatbot")
app.include_router(router)
