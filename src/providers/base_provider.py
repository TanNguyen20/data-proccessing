from abc import ABC, abstractmethod
from typing import Any, Dict

from fastapi import UploadFile


class BaseAIProvider(ABC):
    """Base class for AI providers"""

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the AI provider with necessary credentials"""
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on the given prompt
        
        Args:
            prompt (str): The prompt to generate text from
            **kwargs: Additional arguments like temperature, max_tokens, etc.
            
        Returns:
            str: The generated text
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list:
        """Get list of available models for the provider
        
        Returns:
            list: List of model names available for use with this provider.
                 Each provider implementation should return its specific list of models:
                 - OpenAI: List of available models from OpenAI API
                 - Gemini: List of available Gemini models
                 - xAI: List of available xAI models
        """
        pass

    @abstractmethod
    async def process_excel(self, file_path: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file using the AI provider"""
        pass

    @abstractmethod
    async def process_excel_url(self, excel_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using the AI provider"""
        pass

    @abstractmethod
    async def process_pdf(self, pdf_file: UploadFile, **kwargs) -> Dict[str, Any]:
        """Process PDF file using the AI provider
        
        Args:
            pdf_file: The uploaded PDF file
            **kwargs: Additional arguments like model, temperature, etc.
            
        Returns:
            Dict containing:
                - table_data: List of extracted and formatted table data
                - analysis: Overall document analysis
                - page_count: Number of pages in the PDF
                - row_count: Total number of rows extracted
                - headers_by_page: Dictionary of headers found on each page
        """
        pass

    @abstractmethod
    async def process_pdf_from_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Process PDF file from URL using the AI provider
        
        Args:
            url: URL of the PDF file to process
            **kwargs: Additional arguments like model, temperature, etc.
            
        Returns:
            Dict containing:
                - table_data: List of extracted and formatted table data
                - analysis: Overall document analysis
                - page_count: Number of pages in the PDF
                - row_count: Total number of rows extracted
                - headers_by_page: Dictionary of headers found on each page
                - source_url: Original URL of the PDF file
        """
        pass

    @abstractmethod
    async def process_image(self, image_file: UploadFile, **kwargs) -> Dict[str, Any]:
        """Process image file using the AI provider
        
        Args:
            image_file: The uploaded image file
            **kwargs: Additional arguments like model, temperature, etc.
            
        Returns:
            Dict containing:
                - tables: List of detected tables with their coordinates and content
                - text: Extracted text from the image
                - analysis: Overall image analysis
                - image_metadata: Basic metadata about the image (dimensions, format, etc.)
        """
        pass

    @abstractmethod
    async def process_table_by_ai(self, file: UploadFile, **kwargs) -> Dict[str, Any]:
        """Process any file using AI to extract table data.
        
        This method supports two approaches for processing files:
        1. Direct file processing: The file content is processed directly by the AI model
        2. File upload approach: The file is uploaded to the AI provider's system first (e.g., OpenAI's file upload API)
        
        Args:
            file: The file to process (supports any file type)
            **kwargs: Additional arguments like:
                - model: Optional model name to use
                - temperature: Temperature for the AI model (0.0 to 1.0)
                - max_tokens: Maximum number of tokens to generate
                - use_file_upload: Optional boolean to force using file upload approach if supported
            
        Returns:
            Dict containing:
                - table_data: List of extracted table rows, where each row is a dictionary
                  with column headers as keys and cell values as values
                - analysis: Overall analysis of the document content
                
        The table_data structure will be:
        [
            {
                "column1": "value1",
                "column2": "value2",
                ...
            },
            ...
        ]
        
        Raises:
            ValueError: If the response is not properly formatted or missing required data
            Exception: For any other processing errors
        """
        pass
