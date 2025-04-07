# xAI Provider

This module provides an implementation of the xAI API for the Multi-AI Provider Data Processing Project.

## Features

- Text processing using the Grok-1 model
- Streaming chat completions
- Image processing using the Grok-1 Vision model
- Table data analysis
- Text embeddings using text-embedding-3-small and text-embedding-3-large models

## Setup

1. Get an API key from [xAI](https://x.ai/)
2. Add your API key to the `.env` file:
   ```
   XAI_API_KEY=your_xai_api_key_here
   ```

## Usage

### Basic Usage

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

# Process image
result = provider.process_image("path/to/image.jpg")
print(result)

# Process table data
table_data = {
    "columns": ["Name", "Age"],
    "data": [{"Name": "John", "Age": 30}, {"Name": "Alice", "Age": 25}]
}
result = provider.process_table(table_data)
print(result)

# Get embeddings
embeddings = provider.get_embedding("Sample text")
print(embeddings)
```

### Using with AIProcessor

```python
from src.main import AIProcessor

# Initialize the processor with xAI provider
processor = AIProcessor(provider='xai')

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

## Available Models

- `grok-1`: Text processing model
- `grok-1-vision`: Vision model for image processing
- `text-embedding-3-small`: Small embedding model
- `text-embedding-3-large`: Large embedding model

## API Parameters

The xAI provider supports the following parameters for API calls:

- `model`: The model to use (default: grok-1)
- `temperature`: Controls randomness (default: 0.7)
- `max_tokens`: Maximum number of tokens to generate (default: 1000)
- `stream`: Whether to stream the response (for chat completions)

## Error Handling

The provider includes basic error handling for API errors. If an error occurs, it will raise an exception with the error
message from the xAI API.

## Limitations

- The xAI API has rate limits that may affect usage
- Some features may require specific API access levels
- Image processing is limited to supported image formats

## References

- [xAI API Documentation](https://docs.x.ai/docs/guides/chat#prerequisites) 