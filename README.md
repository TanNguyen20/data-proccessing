# Multi-AI Provider Data Processing Project

This project provides a flexible framework for processing data from various sources using multiple AI providers (OpenAI,
Google Gemini, and xAI).

## Features

- Support for multiple AI providers:
    - OpenAI
    - Google Gemini
    - xAI (Grok)
- Data extraction from multiple sources:
    - Excel files (local and online)
    - Table images (OCR)
    - Facebook links
    - Google search results
    - PDF files
- Streaming chat completions
- Text embeddings
- Image analysis

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   XAI_API_KEY=your_xai_key
   ```

## Project Structure

```
├── src/
│   ├── providers/        # AI provider implementations
│   │   ├── base_provider.py
│   │   ├── openai_provider.py
│   │   ├── gemini_provider.py
│   │   ├── xai_provider.py
│   ├── extractors/       # Data extraction modules
│   │   └── excel_extractor.py
│   ├── utils/            # Utility functions
│   ├── config.py         # Configuration management
│   └── main.py           # Main entry point
├── requirements.txt      # Project dependencies
└── .env.template         # Templete for environment variables 
```

## Usage

### Basic Usage

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

# Process text
result = processor.process_text("Analyze this text")
print(result)
```

### Using Specific Providers

You can also use specific providers directly:

```python
from src.providers.xai_provider import XAIProvider

# Initialize the provider
provider = XAIProvider()
provider.initialize()

# Process text
result = provider.process_text("What is artificial intelligence?")
print(result)

# Streaming chat
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Tell me about the history of AI."}
]
for line in provider.stream_chat(messages):
    if line:
        # Process streaming response
        print(line.decode('utf-8'))
```

## License

MIT License 