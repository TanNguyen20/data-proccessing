# Multi-AI Provider Data Processing Project

This project provides a flexible framework for processing data from various sources using multiple AI providers (OpenAI, Google Gemini, and xAI). It includes a FastAPI-based API for easy integration and data processing capabilities.

## Features

- Support for multiple AI providers:
    - OpenAI
    - Google Gemini
    - xAI (Grok)
- Data extraction and processing from multiple sources:
    - Excel files (local and online)
    - PDF files (local and online)
    - Table data extraction from web pages
- RESTful API endpoints for all processing capabilities
- MongoDB integration for data storage
- Error handling and logging
- CORS support for cross-origin requests

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys (use `.env.template` as reference):
   ```
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   XAI_API_KEY=your_xai_key
   MONGODB_URI=your_mongodb_connection_string
   ```

## Project Structure

```
├── src/
│   ├── providers/        # AI provider implementations
│   │   ├── base_provider.py    # Abstract base class for providers
│   │   ├── openai_provider.py
│   │   ├── gemini_provider.py
│   │   └── xai_provider.py
│   ├── extractors/       # Data extraction modules
│   │   └── excel_extractor.py
│   ├── utils/           # Utility functions
│   ├── config.py        # Configuration management
│   ├── database.py      # MongoDB integration
│   ├── api.py          # FastAPI routes and endpoints
│   └── main.py         # Main processing logic
├── run_api.py          # FastAPI application entry point
├── requirements.txt    # Project dependencies
└── .env.template       # Template for environment variables
```

## API Endpoints

Start the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000` with the following endpoints:

### Excel Processing
- `POST /process-excel-file` - Process uploaded Excel file
- `POST /process-excel-url` - Process Excel file from URL

### PDF Processing
- `POST /process-pdf` - Process uploaded PDF file
- `POST /process-pdf-from-url` - Process PDF file from URL

### Table Extraction
- `POST /extract-table-from-page` - Extract table data from web page

### Provider Information
- `GET /providers` - List available AI providers

All endpoints support the following common parameters:
- `provider`: AI provider to use (xai, openai, gemini)
- `model`: Specific model to use (optional)
- `temperature`: Temperature for the AI model (0.0 to 1.0)
- `max_tokens`: Maximum number of tokens to generate

## Python Library Usage

```python
from src.main import AIProcessor

# Initialize the processor with your preferred provider
processor = AIProcessor(provider='xai')  # Options: 'openai', 'gemini', 'xai'

# Process Excel file
result = await processor.process_excel('path/to/excel.xlsx')
print(result)

# Process PDF
result = await processor.process_pdf('path/to/document.pdf')
print(result)
```

## Dependencies

Key dependencies include:
- `openai>=1.12.0` - OpenAI API client
- `google-generativeai>=0.3.2` - Google Gemini API client
- `pandas>=2.2.3` - Data manipulation
- `fastapi>=0.110.0` - API framework
- `pdfplumber>=0.10.3` - PDF processing
- `pymongo>=4.6.1` - MongoDB integration
- `python-dotenv>=1.0.0` - Environment variable management
- And more (see requirements.txt for complete list)

## License

MIT License 