from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAIProvider(ABC):
    """Base class for AI providers"""

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the AI provider with necessary credentials"""
        pass

    @abstractmethod
    def process_text(self, text: str, **kwargs) -> str:
        """Process text using the AI provider"""
        pass

    @abstractmethod
    def process_image(self, image_path: str, **kwargs) -> str:
        """Process image using the AI provider"""
        pass

    @abstractmethod
    def process_table(self, table_data: Any, **kwargs) -> Dict:
        """Process table data using the AI provider"""
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
