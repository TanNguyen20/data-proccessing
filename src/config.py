import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
XAI_API_KEY = os.getenv('XAI_API_KEY')

# Provider Settings
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'xai')  # Can be 'openai', 'gemini', or 'xai'

# xAI Settings
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_DEFAULT_MODEL = os.getenv('XAI_DEFAULT_MODEL', "grok-1")
XAI_VISION_MODEL = os.getenv('XAI_VISION_MODEL', "grok-1-vision")
XAI_EMBEDDING_MODEL = os.getenv('XAI_EMBEDDING_MODEL', "text-embedding-3-small")

# OpenAI Settings
OPENAI_DEFAULT_MODEL = os.getenv('OPENAI_DEFAULT_MODEL', "gpt-3.5-turbo")
OPENAI_VISION_MODEL = os.getenv('OPENAI_VISION_MODEL', "gpt-4-vision-preview")
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', "text-embedding-ada-002")

# Gemini Settings
GEMINI_DEFAULT_MODEL = os.getenv('GEMINI_DEFAULT_MODEL', "gemini-pro")
GEMINI_VISION_MODEL = os.getenv('GEMINI_VISION_MODEL', "gemini-pro-vision")

# File Processing Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_EXCEL_FORMATS = ['.xlsx', '.xls', '.csv']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']
SUPPORTED_PDF_FORMATS = ['.pdf']

# Web Scraping Settings
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
REQUEST_TIMEOUT = 30  # seconds

# OCR Settings
TESSERACT_CMD = 'tesseract'  # Path to tesseract executable
