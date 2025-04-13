import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
XAI_API_KEY = os.getenv('XAI_API_KEY')

# Provider Settings
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'xai')

# API Base URLs
XAI_BASE_URL = "https://api.x.ai/v1"
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', "https://api.openai.com/v1")

# OpenAI Model Settings
OPENAI_DEFAULT_MODEL = os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-3.5-turbo')
OPENAI_VISION_MODEL = os.getenv('OPENAI_VISION_MODEL', 'gpt-4-vision-preview')
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')

# Gemini Model Settings
GEMINI_DEFAULT_MODEL = os.getenv('GEMINI_DEFAULT_MODEL', 'gemini-1.5-pro')
GEMINI_VISION_MODEL = os.getenv('GEMINI_VISION_MODEL', 'gemini-pro-vision')
GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'embedding-001')

# xAI Model Settings
XAI_DEFAULT_MODEL = os.getenv('XAI_DEFAULT_MODEL', 'grok-1')
XAI_VISION_MODEL = os.getenv('XAI_VISION_MODEL', 'grok-1-vision')
XAI_EMBEDDING_MODEL = os.getenv('XAI_EMBEDDING_MODEL', 'text-embedding-3-small')

# File Processing Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = {
    'excel': ['.xlsx', '.xls', '.csv'],
    'image': ['.jpg', '.jpeg', '.png'],
    'pdf': ['.pdf']
}

# Column Pattern Settings
COLUMN_PATTERNS = {
    'class_code_prefixes': ['DH', 'TH'],
    'phone_prefixes': ['0', '+84', '*'],
    'phone_lengths': [9, 10, 11],
    'student_id_length_range': (7, 9),
    'sequential_max_gap': 5,
    'multi_word_ratio_threshold': 0.8,
    'numeric_ratio_threshold': 0.9,
    'pattern_match_threshold': 0.8
}

# Web Scraping Settings
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
REQUEST_TIMEOUT = 30  # seconds

# OCR Settings
TESSERACT_CMD = 'tesseract'  # Path to tesseract executable
