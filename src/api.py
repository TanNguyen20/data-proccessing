import logging
import os
import tempfile
import traceback
from functools import wraps
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

from .main import AIProcessor
from .utils.page_extractor import extract_table_as_json
from .database import DatabaseNameGenerator, insert_json_data, insert_many_json_data

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


class TableDataRequest(BaseAIRequest):
    table_data: Dict[str, Any]


class ChatRequest(BaseAIRequest):
    messages: List[Dict[str, str]]
    stream: bool = False


class ExcelUrlRequest(BaseAIRequest):
    excel_url: HttpUrl
    prompt: Optional[str] = None


class URLRequest(BaseModel):
    url: str


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


@app.get("/embedding")
@handle_ai_errors
async def get_embedding(
        text: str = Query(..., description="Text to get embedding for"),
        provider: str = Query("xai", description="AI provider to use")
):
    processor = AIProcessor(provider=provider)
    embedding = processor.get_embedding(text)
    return {"embedding": embedding}


@app.post("/process-excel-file")
@handle_ai_errors
async def process_excel_file(
        excel_file: UploadFile = File(...),
        provider: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 1000
):
    """
    Process an Excel file uploaded by the user and save results to MongoDB.
    
    Args:
        excel_file: The Excel file to process
        provider: The AI provider to use (xai, openai, gemini)
        prompt: Optional prompt to guide the analysis
        temperature: Temperature for the AI model (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict containing:
        - processed_data: The processed Excel data
        - analysis: Overall analysis of the data
        - collection_name: Name of the MongoDB collection where data is stored
        - document_ids: List of IDs of the stored documents
    """
    try:
        # Get the file extension from the original filename
        file_extension = os.path.splitext(excel_file.filename)[1].lower()
        
        # Validate file extension
        if file_extension not in ['.xlsx', '.xls', '.csv']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: .xlsx, .xls, .csv"
            )

        # Save the uploaded file temporarily with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await excel_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Prepare kwargs for the processor
        kwargs = {}
        if provider:
            kwargs["provider"] = provider
        if prompt:
            kwargs["prompt"] = prompt
        if temperature:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            # Process the Excel file
            processor = AIProcessor(provider=provider if provider else "xai")
            result = await processor.process_excel(temp_file_path, **kwargs)

            # Generate collection name from filename
            collection_name = await DatabaseNameGenerator.generate_table_name_from_file_content(
                excel_file,
                db_type='mongodb',
                max_length=30
            )

            # Create documents for each row with metadata
            documents = []
            for row in result.get('data', []):
                document = {
                    "filename": excel_file.filename,
                    "processed_at": datetime.utcnow(),
                    "analysis": result.get('analysis'),
                    "data": row
                }
                documents.append(document)

            # Store all documents in MongoDB
            doc_ids = insert_many_json_data(collection_name, documents)

            # Add MongoDB info to the result
            result.update({
                "collection_name": collection_name,
                "document_ids": doc_ids
            })

            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing Excel file: {str(e)}"
            )
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-excel-url/")
@handle_ai_errors
async def process_excel_url(request: ExcelUrlRequest):
    """
    Process an Excel file from a URL and save results to MongoDB.
    
    Args:
        request: The request containing:
            - excel_url: The URL of the Excel file or Google Sheet
            - provider: The AI provider to use (xai, openai, gemini)
            - model: The model to use
            - prompt: Optional prompt to guide the analysis
            - temperature: Temperature for the AI model (0.0 to 1.0)
            - max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict containing:
        - data: The processed Excel data
        - analysis: Overall analysis of the data
        - collection_name: Name of the MongoDB collection where data is stored
        - document_ids: List of IDs of the stored documents
    """
    try:
        # Get the file extension from the URL
        file_extension = os.path.splitext(str(request.excel_url))[1].lower()
        
        # Validate file extension for direct Excel URLs
        if not str(request.excel_url).startswith(('https://docs.google.com/spreadsheets')):
            if file_extension not in ['.xlsx', '.xls', '.csv']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Supported formats: .xlsx, .xls, .csv"
                )

        # Prepare kwargs for the processor
        kwargs = {}
        if request.model:
            kwargs["model"] = request.model
        if request.prompt:
            kwargs["prompt"] = request.prompt
        if request.temperature:
            kwargs["temperature"] = request.temperature
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        # Process the Excel URL
        processor = AIProcessor(provider=request.provider)
        result = await processor.process_excel_url(
            str(request.excel_url),
            **kwargs
        )

        # Generate collection name from URL
        collection_name = await DatabaseNameGenerator.generate_table_name_from_url(
            str(request.excel_url),
            db_type='mongodb',
            max_length=30,
            use_domain=True
        )

        # Create documents for each row with metadata
        documents = []
        for row in result.get('data', []):
            document = {
                "url": str(request.excel_url),
                "processed_at": datetime.utcnow(),
                "analysis": result.get('analysis'),
                "data": row
            }
            documents.append(document)

        # Store all documents in MongoDB
        doc_ids = insert_many_json_data(collection_name, documents)

        # Add MongoDB info to the result
        result.update({
            "collection_name": collection_name,
            "document_ids": doc_ids
        })

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Excel URL: {str(e)}"
        )


@app.post("/process-pdf")
@handle_ai_errors
async def process_pdf(
        file: UploadFile = File(...),
        provider: str = Query("xai", description="AI provider to use"),
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
):
    """
    Process PDF file with table data using AI and save results to MongoDB.
    
    Args:
        file: The PDF file to process
        provider: The AI provider to use (xai, openai, gemini)
        model: Optional specific model to use
        prompt: Optional prompt to guide the analysis
        temperature: Temperature for the AI model (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict containing:
        - table_data: Structured table data from the PDF
        - analysis: Overall analysis of the document
        - page_count: Number of pages in the PDF
        - collection_name: Name of the MongoDB collection where data is stored
        - document_ids: List of IDs of the stored documents
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        processor = AIProcessor(provider=provider)
        result = await processor.process_pdf(
            file,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Generate collection name from filename
        collection_name = await DatabaseNameGenerator.generate_table_name_from_file_content(
            file,
            db_type='mongodb',
            max_length=30
        )

        # Create documents for each table row with metadata
        documents = []
        for row in result.get('table_data', []):
            document = {
                "filename": file.filename,
                "processed_at": datetime.utcnow(),
                "analysis": result.get('analysis'),
                "page_count": result.get('page_count'),
                "data": row
            }
            documents.append(document)

        # Store all documents in MongoDB
        doc_ids = insert_many_json_data(collection_name, documents)

        # Add MongoDB info to the result
        result.update({
            "collection_name": collection_name,
            "document_ids": doc_ids
        })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-pdf-from-url")
@handle_ai_errors
async def process_pdf_from_url(
        url: str = Query(..., description="URL of the PDF file to process"),
        provider: str = Query("xai", description="AI provider to use"),
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        verify_ssl: bool = Query(True, description="Whether to verify SSL certificates")
):
    """
    Process PDF file from URL with table data using AI and save results to MongoDB.
    
    Args:
        url: URL of the PDF file to process
        provider: The AI provider to use (xai, openai, gemini)
        model: Optional specific model to use
        prompt: Optional prompt to guide the analysis
        temperature: Temperature for the AI model (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        verify_ssl: Whether to verify SSL certificates (default: True)
        
    Returns:
        Dict containing:
        - table_data: Structured table data from the PDF
        - analysis: Overall analysis of the document
        - page_count: Number of pages in the PDF
        - row_count: Total number of rows extracted
        - headers_by_page: Dictionary of headers found on each page
        - source_url: Original URL of the PDF file
        - collection_name: Name of the MongoDB collection where data is stored
        - document_ids: List of IDs of the stored documents
    """
    try:
        # Validate URL
        if not url.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="URL must point to a PDF file")

        processor = AIProcessor(provider=provider)
        result = await processor.process_pdf_from_url(
            url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            verify_ssl=verify_ssl
        )

        # Generate collection name from URL
        collection_name = await DatabaseNameGenerator.generate_table_name_from_url(
            url,
            db_type='mongodb',
            max_length=30,
            use_domain=True
        )

        # Create documents for each table row with metadata
        documents = []
        for row in result.get('table_data', []):
            document = {
                "url": url,
                "processed_at": datetime.utcnow(),
                "analysis": result.get('analysis'),
                "page_count": result.get('page_count'),
                "data": row
            }
            documents.append(document)

        # Store all documents in MongoDB
        doc_ids = insert_many_json_data(collection_name, documents)

        # Add MongoDB info to the result
        result.update({
            "collection_name": collection_name,
            "document_ids": doc_ids
        })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-table-from-page")
async def extract_table(request: URLRequest):
    try:
        # Extract table data
        table_data = await extract_table_as_json(request.url)
        
        # Generate collection name from URL
        collection_name = await DatabaseNameGenerator.generate_table_name_from_url(
            request.url,
            db_type='mongodb',
            max_length=30,
            use_domain=True
        )
        
        # Create a document for each row with metadata
        documents = []
        for row in table_data:
            document = {
                "url": request.url,
                "extracted_at": datetime.utcnow(),
                "data": row
            }
            documents.append(document)
        
        # Store all documents in MongoDB
        doc_ids = insert_many_json_data(collection_name, documents)
        
        return {
            "data": table_data,
            "collection_name": collection_name,
            "document_ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "message": str(exc),
            "error_type": exc.__class__.__name__,
            "traceback": traceback.format_exc()
        }
    )
