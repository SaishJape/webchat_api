from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router

app = FastAPI(
    title="Web Scraping Q&A Chatbot",
    description="An intelligent chatbot that scrapes and answers questions from websites using Qdrant and Gemini.",
    version="1.0.0"
)

# CORS Middleware (optional but useful for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes from the API router
app.include_router(router)

# Custom 404 handler (optional)
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Oops! The resource was not found."},
    )

# Startup and Shutdown Events (for DB, scraping schedulers, etc.)
@app.on_event("startup")
async def startup_event():
    print("Application startup: Initialize services or DB connections here.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown: Clean up services or DB connections here.")
