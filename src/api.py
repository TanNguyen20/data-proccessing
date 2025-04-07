import os
import tempfile
import logging
import traceback
from typing import List, Dict, Any, Optional
from functools import wraps

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

from .main import AIProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-AI Provider Data Processing API",
    description="API for processing data using multiple AI providers (OpenAI, Gemini, xAI)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Base request model with common fields
class BaseAIRequest(BaseModel):
    provider: str = "xai"
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

# Pydantic models for request/response
class TextRequest(BaseAIRequest):
    text: str

class FacebookPostRequest(BaseAIRequest):
    post_url: HttpUrl

class FacebookPageRequest(BaseAIRequest):
    page_url: HttpUrl

class TableDataRequest(BaseAIRequest):
    table_data: Dict[str, Any]

class ChatRequest(BaseAIRequest):
    messages: List[Dict[str, str]]
    stream: bool = False

class ExcelUrlRequest(BaseAIRequest):
    excel_url: HttpUrl
    prompt: Optional[str] = None

# Error handling decorator
def handle_ai_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Get the full traceback
            error_traceback = traceback.format_exc()
            
            # Log detailed error information
            logger.error(
                f"Error in {func.__name__}:\n"
                f"Args: {args}\n"
                f"Kwargs: {kwargs}\n"
                f"Error: {str(e)}\n"
                f"Traceback:\n{error_traceback}"
            )
            
            # For HTTP exceptions, preserve the status code
            if isinstance(e, HTTPException):
                raise HTTPException(
                    status_code=e.status_code,
                    detail={
                        "message": str(e),
                        "error_type": e.__class__.__name__,
                        "traceback": error_traceback
                    }
                )
            
            # For other exceptions, return 500 with detailed error
            raise HTTPException(
                status_code=500,
                detail={
                    "message": str(e),
                    "error_type": e.__class__.__name__,
                    "traceback": error_traceback
                }
            )
    return wrapper

# Dependency to get AI processor
def get_processor(provider: str = Query("xai", description="AI provider to use")):
    try:
        return AIProcessor(provider=provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-AI Provider Data Processing API"}

@app.get("/providers")
async def get_providers():
    return {
        "providers": ["openai", "gemini", "xai"],
        "default": "xai"
    }

@app.post("/process/text")
@handle_ai_errors
async def process_text(request: TextRequest, processor: AIProcessor = Depends(get_processor)):
    result = processor.process_text(
        request.text,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return {"result": result}

@app.post("/process/facebook/post")
@handle_ai_errors
async def process_facebook_post(request: FacebookPostRequest, processor: AIProcessor = Depends(get_processor)):
    result = processor.process_facebook_post(
        str(request.post_url),
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return result

@app.post("/process/facebook/page")
@handle_ai_errors
async def process_facebook_page(request: FacebookPageRequest, processor: AIProcessor = Depends(get_processor)):
    result = processor.process_facebook_page(
        str(request.page_url),
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return result

@app.post("/process/table")
@handle_ai_errors
async def process_table(request: TableDataRequest, processor: AIProcessor = Depends(get_processor)):
    result = processor.process_table(
        request.table_data,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return {"result": result}

@app.post("/process/image")
@handle_ai_errors
async def process_image(
        file: UploadFile = File(...),
        provider: str = Query("xai", description="AI provider to use"),
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        processor = AIProcessor(provider=provider)
        result = processor.process_image(
            temp_file_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return {"result": result}
    finally:
        os.unlink(temp_file_path)

@app.post("/chat")
@handle_ai_errors
async def chat(request: ChatRequest, processor: AIProcessor = Depends(get_processor)):
    if request.stream:
        async def generate():
            for line in processor.stream_chat(
                    request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
            ):
                if line:
                    yield f"data: {line.decode('utf-8')}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        result = processor.chat(
            request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"result": result}

@app.get("/embedding")
@handle_ai_errors
async def get_embedding(
        text: str = Query(..., description="Text to get embedding for"),
        provider: str = Query("xai", description="AI provider to use")
):
    processor = AIProcessor(provider=provider)
    embedding = processor.get_embedding(text)
    return {"embedding": embedding}

@app.post("/process-excel-url/", response_model=Dict[str, Any])
@handle_ai_errors
async def process_excel_url(request: ExcelUrlRequest):
    processor = AIProcessor()
    result = await processor.process_excel_url(
        excel_url=str(request.excel_url),
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        prompt=request.prompt
    )
    return result

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # Get the full traceback
    error_traceback = traceback.format_exc()
    
    # Log the error
    logger.error(
        f"Unhandled exception:\n"
        f"Path: {request.url.path}\n"
        f"Method: {request.method}\n"
        f"Error: {str(exc)}\n"
        f"Traceback:\n{error_traceback}"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "message": str(exc),
                "error_type": exc.__class__.__name__,
                "traceback": error_traceback
            }
        }
    )
