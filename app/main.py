from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

from app.api.routes import router
from app.utils.logger import setup_logger

# Setup logging
setup_logger()

app = FastAPI(
    title="Web Scraping Q&A Chatbot",
    description="An intelligent chatbot that scrapes and answers questions from websites using Qdrant and Gemini.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes from the API router
app.include_router(router)

# Custom 404 handler
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Oops! The resource was not found."},
    )

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Services initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutdown: Cleaning up services.")

# Entry point for running with uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)