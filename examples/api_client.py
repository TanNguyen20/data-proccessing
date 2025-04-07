import json
import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API URL from environment variable or use default
API_URL = os.getenv("API_URL", "http://localhost:8000")


def process_text(text, provider="xai", model=None, temperature=0.7, max_tokens=1000):
    """Process text using the API"""
    url = f"{API_URL}/process/text"
    payload = {
        "text": text,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def process_facebook_post(post_url, provider="xai", model=None, temperature=0.7, max_tokens=1000):
    """Process Facebook post using the API"""
    url = f"{API_URL}/process/facebook/post"
    payload = {
        "post_url": post_url,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def process_facebook_page(page_url, provider="xai", model=None, temperature=0.7, max_tokens=1000):
    """Process Facebook page using the API"""
    url = f"{API_URL}/process/facebook/page"
    payload = {
        "page_url": page_url,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def process_image(image_path, provider="xai", model=None, temperature=0.7, max_tokens=1000):
    """Process image using the API"""
    url = f"{API_URL}/process/image"

    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        params = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(url, files=files, params=params)
        response.raise_for_status()
        return response.json()


def process_table(table_data, provider="xai", model=None, temperature=0.7, max_tokens=1000):
    """Process table data using the API"""
    url = f"{API_URL}/process/table"
    payload = {
        "table_data": table_data,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def chat(messages, provider="xai", model=None, temperature=0.7, max_tokens=1000, stream=False):
    """Chat with the AI using the API"""
    url = f"{API_URL}/chat"
    payload = {
        "messages": messages,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }

    if stream:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data = line_text[6:]  # Remove 'data: ' prefix
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        print()  # New line at the end
        return None
    else:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def get_embedding(text, provider="xai"):
    """Get embedding for text using the API"""
    url = f"{API_URL}/embedding"
    params = {
        "text": text,
        "provider": provider
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def main():
    # Example 1: Process text
    print("Example 1: Process text")
    try:
        result = process_text("What are the key features of artificial intelligence?")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing text: {e}")

    # Example 2: Process Facebook post
    print("\nExample 2: Process Facebook post")
    try:
        result = process_facebook_post("https://www.facebook.com/example/post/123456789")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing Facebook post: {e}")

    # Example 3: Process Facebook page
    print("\nExample 3: Process Facebook page")
    try:
        result = process_facebook_page("https://www.facebook.com/example")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing Facebook page: {e}")

    # Example 4: Process table data
    print("\nExample 4: Process table data")
    try:
        table_data = {
            "columns": ["Name", "Age", "City"],
            "data": [
                {"Name": "John", "Age": 30, "City": "New York"},
                {"Name": "Alice", "Age": 25, "City": "Los Angeles"},
                {"Name": "Bob", "Age": 35, "City": "Chicago"}
            ]
        }
        result = process_table(table_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing table data: {e}")

    # Example 5: Chat with streaming
    print("\nExample 5: Chat with streaming")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Tell me about the history of AI in 3 sentences."}
        ]
        chat(messages, stream=True)
    except Exception as e:
        print(f"Error in chat: {e}")

    # Example 6: Get embedding
    print("\nExample 6: Get embedding")
    try:
        result = get_embedding("This is a sample text for embedding.")
        print(f"Embedding (first 5 values): {result['embedding'][:5]}...")
    except Exception as e:
        print(f"Error getting embedding: {e}")


if __name__ == "__main__":
    main()
