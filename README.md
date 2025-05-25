# Multi-AI Provider Data Processing Project

This project provides a flexible framework for processing data from various sources using multiple AI providers (OpenAI, Google Gemini, and xAI). It includes a FastAPI-based API for easy integration and data processing capabilities.

## Features

- Support for multiple AI providers:
    - OpenAI
    - Google Gemini
    - xAI (Grok)
- Data extraction and processing from multiple sources:
    - Excel files (local and online)
    - Table images (OCR)
    - Facebook links
    - Google search results
    - PDF files
- RESTful API endpoints for all processing capabilities
- Streaming chat completions
- Text embeddings
- Image analysis
- Table data extraction and processing

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
│   └── main.py          # Main entry point
├── run_api.py           # FastAPI application entry point
├── requirements.txt     # Project dependencies
└── .env.template        # Template for environment variables
```

## Usage

### API Usage

Start the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000` with the following endpoints:
- `/process/text` - Process text data
- `/process/image` - Process image data
- `/process/excel` - Process Excel files
- `/process/pdf` - Process PDF files
- `/process/table` - Process table data

### Python Library Usage

```python
from src.main import AIProcessor

# Initialize the processor with your preferred provider
processor = AIProcessor(provider='xai')  # Options: 'openai', 'gemini', 'xai'

# Process Excel file
result = processor.process_excel('path/to/excel.xlsx', extract_tables=True)
print(result)

# Process image
result = processor.process_image('path/to/image.jpg')
print(result)

# Process PDF
result = processor.process_pdf(pdf_file)
print(result)
```

### Using Specific Providers

You can also use specific providers directly:

```python
from src.providers.xai_provider import XAIProvider

# Initialize the provider
provider = XAIProvider()
provider.initialize()

# Get embeddings
embeddings = provider.get_embedding("Sample text")
print(embeddings)

# Get available models
models = provider.get_available_models()
print(models)
```

## Dependencies

Key dependencies include:
- `openai>=1.12.0` - OpenAI API client
- `google-generativeai>=0.3.2` - Google Gemini API client
- `pandas>=2.2.3` - Data manipulation
- `fastapi>=0.110.0` - API framework
- `pdfplumber>=0.10.3` - PDF processing
- `pytesseract>=0.3.10` - OCR capabilities
- `playwright>=1.52.0` - Web automation
- And more (see requirements.txt for complete list)

## License

MIT License 