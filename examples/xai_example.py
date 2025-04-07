import json
import os
import sys

from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.providers.xai_provider import XAIProvider


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the xAI provider
    provider = XAIProvider()
    provider.initialize()

    # Example 1: Text processing
    text = "What are the key features of artificial intelligence?"
    print("Processing text:", text)
    result = provider.process_text(text)
    print("Result:", result)
    print("\n" + "-" * 50 + "\n")

    # Example 2: Streaming chat
    print("Streaming chat example:")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Tell me about the history of AI in 3 sentences."}
    ]

    print("Response: ", end="", flush=True)
    for line in provider.stream_chat(messages):
        if line:
            try:
                line_data = json.loads(line.decode('utf-8').replace('data: ', ''))
                if 'choices' in line_data and len(line_data['choices']) > 0:
                    delta = line_data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end="", flush=True)
            except json.JSONDecodeError:
                continue
    print("\n" + "-" * 50 + "\n")

    # Example 3: Image processing (if you have an image file)
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    if os.path.exists(image_path):
        print("Processing image:", image_path)
        result = provider.process_image(image_path)
        print("Result:", result)
        print("\n" + "-" * 50 + "\n")

    # Example 4: Table processing
    table_data = {
        "columns": ["Name", "Age", "City"],
        "data": [
            {"Name": "John", "Age": 30, "City": "New York"},
            {"Name": "Alice", "Age": 25, "City": "Los Angeles"},
            {"Name": "Bob", "Age": 35, "City": "Chicago"}
        ]
    }
    print("Processing table:", table_data)
    result = provider.process_table(table_data)
    print("Result:", result)
    print("\n" + "-" * 50 + "\n")

    # Example 5: Getting embeddings
    text = "This is a sample text for embedding."
    print("Getting embeddings for:", text)
    embeddings = provider.get_embedding(text)
    print("Embeddings:", embeddings[:5], "...")  # Print first 5 values

    # Example 6: List available models
    models = provider.get_available_models()
    print("\nAvailable models:", models)


if __name__ == "__main__":
    main()
