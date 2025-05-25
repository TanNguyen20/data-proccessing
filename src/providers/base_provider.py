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
    def get_embedding(self, text: str) -> list:
        """Get embeddings for the given text"""
        pass

    @abstractmethod
    def get_available_models(self) -> list:
        """Get list of available models for the provider"""
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
