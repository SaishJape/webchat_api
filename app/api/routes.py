from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from app.db.models import QARequest, ScrapeRequest, UserCreate, UserLogin, User, Token
from app.services.gemini import ask_gemini, enhanced_query_with_gemini, translate_to_english
from app.services.embeddings import get_embeddings, get_question_embedding
from app.utils.common import crawl_website, clean_text, chunk_text, extract_website_name
from app.db.qdrant import ingest_to_qdrant, query_qdrant, enhanced_query_qdrant
from app.auth.auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.db.mysql import get_db
import mysql.connector
import logging
from typing import Dict, Optional, List
import asyncio
from datetime import datetime, timedelta
import hashlib
import uuid
import traceback
import re
import io
import PyPDF2
import xml.etree.ElementTree as ET
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Store scraping progress with last update time
scraping_progress: Dict[str, dict] = {}

# Allowed file types
ALLOWED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'svg': 'image/svg+xml',
    'txt': 'text/plain',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def get_user_by_email(db, email: str):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    return user

def get_user_by_id(db, user_id: str):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    return user

def create_user(db, user_data: dict):
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO users (id, email, username, password_hash)
        VALUES (%s, %s, %s, %s)
    """, (
        user_data['id'],
        user_data['email'],
        user_data['username'],
        user_data['password_hash']
    ))
    db.commit()
    cursor.close()

def preprocess_text(text: str) -> str:
    """Preprocess text to ensure it's clean and properly formatted."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Create overlapping chunks from text with fixed size."""
    if not text:
        return []
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks

def process_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += preprocess_text(page_text) + "\n"
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing PDF file: {str(e)}")

def process_svg(file_content: bytes) -> str:
    """Extract text from SVG file."""
    try:
        svg_content = file_content.decode('utf-8')
        root = ET.fromstring(svg_content)
        # Extract text elements from SVG
        text_elements = root.findall(".//{http://www.w3.org/2000/svg}text")
        text = "\n".join([elem.text for elem in text_elements if elem.text])
        return preprocess_text(text)
    except Exception as e:
        logger.error(f"Error processing SVG: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing SVG file: {str(e)}")

def process_text_file(file_content: bytes) -> str:
    """Process text-based files."""
    try:
        text = file_content.decode('utf-8')
        return preprocess_text(text)
    except Exception as e:
        logger.error(f"Error processing text file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing text file: {str(e)}")

@router.post("/signup", response_model=User)
async def signup(user: UserCreate, db = Depends(get_db)):
    """Create a new user account."""
    try:
        logger.info(f"Signup attempt for email: {user.email}")
        
        # Check if user already exists
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        existing_user = cursor.fetchone()
        cursor.close()
        
        if existing_user:
            logger.warning(f"Signup failed: Email already registered: {user.email}")
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)
        
        # Insert new user
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO users (id, email, username, password_hash)
            VALUES (%s, %s, %s, %s)
        """, (
            user_id,
            user.email,
            user.username,
            hashed_password
        ))
        db.commit()
        cursor.close()
        
        logger.info(f"User created successfully: {user.email}")
        return {
            'id': user_id,
            'email': user.email,
            'username': user.username,
            'created_at': datetime.now(),
            'is_active': True
        }
    except HTTPException as he:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise he
    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An error occurred during signup. Please try again."
        )

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db = Depends(get_db)
):
    """Login endpoint for OAuth2 password flow"""
    try:
        logger.info(f"Login attempt for username: {form_data.username}")
        
        # Find user
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (form_data.username,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"Login failed: User not found with email {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not verify_password(form_data.password, user['password_hash']):
            logger.warning(f"Login failed: Invalid password for user {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['id']}, expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )

@router.post("/login/json", response_model=Token)
async def login_json(user_data: UserLogin, db = Depends(get_db)):
    """Login endpoint for JSON data"""
    try:
        logger.info(f"Login attempt for email: {user_data.email}")
        
        # Find user
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user_data.email,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"Login failed: User not found with email {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not verify_password(user_data.password, user['password_hash']):
            logger.warning(f"Login failed: Invalid password for user {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['id']}, expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user {user_data.email}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )

@router.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Upload and process files (PDF, SVG, etc.) and store in user's collection."""
    try:
        # Use user's ID as collection name
        collection_name = current_user['id']
        
        logger.info(f"Starting file upload process for file: {file.filename}")
        
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file type: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}"
            )

        # Read file content
        try:
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file received")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Process file based on type
        text_content = ""
        try:
            if file_extension == 'pdf':
                text_content = process_pdf(file_content)
            elif file_extension == 'svg':
                text_content = process_svg(file_content)
            else:
                text_content = process_text_file(file_content)
        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing file content: {str(e)}")

        if not text_content.strip():
            logger.warning("No text content extracted from file")
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")

        # Create chunks from the text
        try:
            chunks = create_chunks(text_content, chunk_size=1000, overlap=200)
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Validate chunks
            valid_chunks = [chunk for chunk in chunks if chunk.strip()]
            if len(valid_chunks) != len(chunks):
                logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} empty chunks")
                chunks = valid_chunks
                
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid text chunks could be created from the file")
                
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating text chunks: {str(e)}")

        # Generate embeddings
        try:
            embeddings = get_embeddings(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                raise Exception("Embedding generation failed or produced mismatched results")
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

        # Ingest to Qdrant
        try:
            ingest_to_qdrant(collection_name, chunks, embeddings)
            logger.info(f"Successfully ingested {len(chunks)} chunks to collection {collection_name}")
        except Exception as e:
            logger.error(f"Error ingesting to Qdrant: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error storing data: {str(e)}")

        return {
            "status": "success",
            "message": "File processed and stored successfully",
            "collection_name": collection_name,
            "chunks_created": len(chunks),
            "file_name": file.filename
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_and_process: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/scrape-and-ingest")
async def scrape_and_ingest(
    req: ScrapeRequest,
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Scrape a website and ingest the content into user's collection."""
    try:
        # Use user's ID as collection name
        collection_name = current_user['id']
        
        logger.info(f"Starting scrape and ingest for URL: {req.url}")
        
        # Generate a unique ID for this scraping task
        task_id = hashlib.md5(f"{req.url}_{datetime.now().timestamp()}".encode()).hexdigest()
        
        # Initialize progress tracking
        scraping_progress[task_id] = {
            "status": "crawling",
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "pages_scraped": 0,
            "chunks_created": 0,
            "error": None,
            "is_completed": False
        }
        
        # Start background task
        background_tasks.add_task(process_scraping, req.url, task_id, collection_name)
        
        return {"task_id": task_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error starting scrape process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scraping-progress/{task_id}")
async def get_scraping_progress(task_id: str):
    """Get the current progress of a scraping task."""
    if task_id not in scraping_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if we should return cached response
    current_time = datetime.now()
    last_update = scraping_progress[task_id]["last_update"]
    time_diff = (current_time - last_update).total_seconds()
    
    # If the task is completed or there's an error, return immediately
    if scraping_progress[task_id]["is_completed"] or scraping_progress[task_id]["error"]:
        return scraping_progress[task_id]
    
    # If less than 2 seconds have passed since last update, return cached response
    if time_diff < 2:
        return scraping_progress[task_id]
    
    # Update last update time
    scraping_progress[task_id]["last_update"] = current_time
    return scraping_progress[task_id]

async def process_scraping(url: str, task_id: str, collection_name: str):
    """Background task to process scraping and ingestion."""
    try:
        # Update status to crawling
        update_progress(task_id, "crawling")
        
        # Crawl the website
        pages = crawl_website(str(url))
        
        if not pages:
            update_progress(task_id, "error", error="No pages could be scraped from the provided URL")
            return
        
        update_progress(task_id, "crawling", pages_scraped=len(pages))
        
        # Update status to processing
        update_progress(task_id, "processing")
        
        all_chunks = []
        
        # Process each page
        for url, html in pages.items():
            if html and isinstance(html, str) and len(html) > 0:
                cleaned_text = clean_text(html)
                if cleaned_text.strip():
                    # Create chunks with size 64
                    chunks = create_chunks(cleaned_text, chunk_size=64, overlap=10)
                    all_chunks.extend(chunks)
        
        if not all_chunks:
            update_progress(task_id, "error", error="No valid text content found to ingest from the website")
            return
        
        update_progress(task_id, "processing", chunks_created=len(all_chunks))
        
        # Update status to generating embeddings
        update_progress(task_id, "generating_embeddings")
        
        # Generate embeddings
        embeddings = get_embeddings(all_chunks)
        
        # Update status to storing
        update_progress(task_id, "storing")
        
        # Ingest to Qdrant using user's collection
        ingest_to_qdrant(collection_name, all_chunks, embeddings)
        
        # Update status to completed
        update_progress(task_id, "completed", 
                       result={
                           "collection_name": collection_name,
                           "pages_scraped": len(pages),
                           "chunks_created": len(all_chunks)
                       })
        
    except Exception as e:
        logger.error(f"Error in scraping process: {e}")
        update_progress(task_id, "error", error=str(e))

def update_progress(task_id: str, status: str, **kwargs):
    """Update progress with new status and optional data."""
    if task_id not in scraping_progress:
        return
        
    scraping_progress[task_id].update({
        "status": status,
        "last_update": datetime.now(),
        **kwargs
    })
    
    if status in ["completed", "error"]:
        scraping_progress[task_id]["is_completed"] = True

# @router.post("/ask-question")
# async def ask_question(
#     req: QARequest,
#     # current_user = Depends(get_current_active_user),
#     db = Depends(get_db)
# ):
#     """Ask a question using user's collection and return clean chatbot-ready JSON."""
#     try:
#         logging.info(f"Processing question: {req.question}")

#         # Get question embedding
#         question_embedding = get_question_embedding(req.question)

#         # Use enhanced query with Gemini for better processing
#         enhanced_results = enhanced_query_with_gemini(
#             collection_name=req.collection_name,
#             user_query=req.question,
#             query_vector=question_embedding,
#             limit=5
#         )

#         # Extract context from search results
#         context_chunks = []
#         if enhanced_results.get('search_results'):
#             for result in enhanced_results['search_results']:
#                 if result.get('payload') and result['payload'].get('text'):
#                     text = result['payload']['text']
#                     score = result.get('score', 0)
#                     context_chunks.append(f"[Relevance: {score:.3f}] {text}")
#         context = "\n\n".join(context_chunks)

#         print("enhanced_results.get('context_text', {}) -> ", enhanced_results.get('context_text', {}))
#         # Get structured answer from Gemini
#         gemini_output = ask_gemini(context, req.question, enhanced_results.get('context_text', {}))

#         return gemini_output

#     except Exception as e:
#         logging.error(f"Error in ask_question: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@router.post("/ask-question")
async def ask_question(req: QARequest, db=Depends(get_db)):
    try:
        logging.info(f"Processing question: {req.question}")
        
        translated_query = translate_to_english(req.question)

        # Step 1: Get embedding
        # question_embedding = get_question_embedding(req.question)
        question_embedding = get_question_embedding(translated_query)

        # Step 2: Enhanced query to get search results and context
        enhanced_results = enhanced_query_with_gemini(
            collection_name=req.collection_name,
            user_query=translated_query,
            # user_query=req.question,
            query_vector=question_embedding,
            limit=5
        )

        # Step 3: Ask Gemini with the full context and results
        final_response = ask_gemini(
            enhanced_results.get("context_text", ""),  # context string
            req.question,                              # user question
            enhanced_results.get("processed_query", {}),  # parsed search info
            enhanced_results                            # full result dict
        )

        return final_response

    except Exception as e:
        logging.error(f"Error in ask_question: {e}")
        return {
            "response": "Something went wrong while answering your question.",
            "buttons": False,
            "button_type": None,
            "button_data": None
        }
