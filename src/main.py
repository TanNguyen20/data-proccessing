from typing import Dict, Any

from fastapi import UploadFile

from .config import DEFAULT_PROVIDER
from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider
from .providers.xai_provider import XAIProvider


class AIProcessor:
    def __init__(self, provider: str = DEFAULT_PROVIDER):
        self.provider = self._initialize_provider(provider)

    def _initialize_provider(self, provider_name: str):
        """Initialize the specified AI provider"""
        provider_name = provider_name.strip().lower()  # Normalize provider name
        providers = {
            'openai': OpenAIProvider,
            'gemini': GeminiProvider,
            'xai': XAIProvider
        }

        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}")

        provider = providers[provider_name]()
        provider.initialize()
        return provider

    async def process_excel(self, excel_file: UploadFile, **kwargs) -> Dict[str, Any]:
        """Process Excel file using AI"""
        try:
            # Check if provider is specified in kwargs and is different from current provider
            if 'provider' in kwargs and kwargs['provider'] != self.provider.__class__.__name__.lower().replace(
                    'provider', ''):
                # Initialize a new provider with the specified provider name
                self.provider = self._initialize_provider(kwargs['provider'])
                # Remove provider from kwargs to avoid passing it to the provider method
                del kwargs['provider']

            return await self.provider.process_excel(excel_file, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")

    async def process_pdf(self, pdf_file: UploadFile, **kwargs) -> Dict[str, Any]:
        """Process PDF file using AI
        
        Args:
            pdf_file: The uploaded PDF file
            **kwargs: Additional arguments like model, temperature, prompt etc.
            
        Returns:
            Dict containing the processed PDF data and analysis
        """
        try:
            # Check if provider is specified in kwargs and is different from current provider
            if 'provider' in kwargs and kwargs['provider'] != self.provider.__class__.__name__.lower().replace(
                    'provider', ''):
                # Initialize a new provider with the specified provider name
                self.provider = self._initialize_provider(kwargs['provider'])
                # Remove provider from kwargs to avoid passing it to the provider method
                del kwargs['provider']

            return await self.provider.process_pdf(pdf_file, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing PDF file: {str(e)}")

    async def process_pdf_from_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Process PDF file from URL using AI
        
        Args:
            url: URL of the PDF file to process
            **kwargs: Additional arguments like model, temperature, prompt etc.
            
        Returns:
            Dict containing:
                - table_data: List of extracted and formatted table data
                - analysis: Overall document analysis
                - page_count: Number of pages in the PDF
                - row_count: Total number of rows extracted
                - headers_by_page: Dictionary of headers found on each page
                - source_url: Original URL of the PDF file
        """
        try:
            # Check if provider is specified in kwargs and is different from current provider
            if 'provider' in kwargs and kwargs['provider'] != self.provider.__class__.__name__.lower().replace(
                    'provider', ''):
                # Initialize a new provider with the specified provider name
                self.provider = self._initialize_provider(kwargs['provider'])
                # Remove provider from kwargs to avoid passing it to the provider method
                del kwargs['provider']

            # Ensure verify_ssl is passed through
            verify_ssl = kwargs.get('verify_ssl', True)
            kwargs['verify_ssl'] = verify_ssl

            return await self.provider.process_pdf_from_url(url, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing PDF from URL: {str(e)}")

    async def process_excel_url(self, excel_url: str, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using AI"""
        try:
            # Check if provider is specified in kwargs and is different from current provider
            if 'provider' in kwargs and kwargs['provider'] != self.provider.__class__.__name__.lower().replace(
                    'provider', ''):
                # Initialize a new provider with the specified provider name
                self.provider = self._initialize_provider(kwargs['provider'])
                # Remove provider from kwargs to avoid passing it to the provider method
                del kwargs['provider']

            return await self.provider.process_excel_url(excel_url, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing Excel URL: {str(e)}")

    def get_available_models(self) -> list:
        """Get list of available models for the current provider
        
        Returns:
            list: List of available model names for the current provider
        """
        try:
            return self.provider.get_available_models()
        except Exception as e:
            raise Exception(f"Error getting available models: {str(e)}")


# Initialize the AI processor
processor = AIProcessor()
