from typing import Union, Dict, List, Any

from .config import DEFAULT_PROVIDER
from .extractors.excel_extractor import ExcelExtractor
from .extractors.facebook_extractor import FacebookExtractor
from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider
from .providers.xai_provider import XAIProvider


class AIProcessor:
    def __init__(self, provider: str = DEFAULT_PROVIDER):
        self.provider = self._initialize_provider(provider)
        self.facebook_extractor = FacebookExtractor()

    def _initialize_provider(self, provider_name: str):
        """Initialize the specified AI provider"""
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

    async def process_excel(self, file_path_or_url: str, **kwargs) -> Union[Dict, List[Dict]]:
        """Process Excel file and analyze its contents using AI"""
        try:
            return await self.provider.process_excel(file_path_or_url, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing Excel: {str(e)}")

    async def process_image(self, image_path: str, **kwargs) -> str:
        """Process image using AI vision capabilities"""
        try:
            return await self.provider.process_image(image_path, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    async def process_text(self, text: str, **kwargs) -> str:
        """Process text using AI"""
        try:
            return await self.provider.process_text(text, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing text: {str(e)}")

    async def process_facebook_post(self, post_url: str, **kwargs) -> Dict:
        """Process Facebook post data using AI"""
        try:
            return await self.provider.process_facebook_post(post_url, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing Facebook post: {str(e)}")

    async def process_facebook_page(self, page_url: str, **kwargs) -> Dict:
        """Process Facebook page data using AI"""
        try:
            return await self.provider.process_facebook_page(page_url, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing Facebook page: {str(e)}")

    async def process_excel_url(self, excel_url: str, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using AI"""
        try:
            if 'provider' in kwargs:
                del kwargs['provider']
            return await self.provider.process_excel_url(excel_url, **kwargs)
        except Exception as e:
            raise Exception(f"Error processing Excel URL: {str(e)}")


async def main():
    # Example usage
    processor = AIProcessor(provider='xai')  # Using xAI as the default provider

    # Process Excel file
    result = await processor.process_excel('path/to/excel.xlsx')
    print("Excel Analysis:", result)

    # Process image
    image_analysis = await processor.process_image('path/to/image.jpg')
    print("Image Analysis:", image_analysis)

    # Process text
    text_analysis = await processor.process_text("Analyze this text")
    print("Text Analysis:", text_analysis)

    # Process Facebook post
    facebook_post_analysis = await processor.process_facebook_post('https://www.facebook.com/example/post/123456789')
    print("Facebook Post Analysis:", facebook_post_analysis)

    # Process Facebook page
    facebook_page_analysis = await processor.process_facebook_page('https://www.facebook.com/example')
    print("Facebook Page Analysis:", facebook_page_analysis)
    
    # Process Excel URL
    excel_url_analysis = await processor.process_excel_url('https://example.com/data.xlsx')
    print("Excel URL Analysis:", excel_url_analysis)


if __name__ == "__main__":
    main()
