import json
import os
import sys

from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import AIProcessor
from src.extractors.facebook_extractor import FacebookExtractor


def process_facebook_post(processor, post_url):
    """Process a Facebook post using the AI processor"""
    print(f"\nProcessing Facebook post: {post_url}")
    try:
        result = processor.process_facebook_post(post_url)
        print("Facebook Post Analysis Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing Facebook post: {e}")


def process_facebook_page(processor, page_url):
    """Process a Facebook page using the AI processor"""
    print(f"\nProcessing Facebook page: {page_url}")
    try:
        result = processor.process_facebook_page(page_url)
        print("Facebook Page Analysis Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing Facebook page: {e}")


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the processor with xAI provider
    processor = AIProcessor(provider='xai')

    # Example 1: Process Facebook post
    post_url = "https://www.facebook.com/example/post/123456789"  # Replace with a real Facebook post URL
    process_facebook_post(processor, post_url)

    # Example 2: Process Facebook page
    page_url = "https://www.facebook.com/example"  # Replace with a real Facebook page URL
    process_facebook_page(processor, page_url)

    # Example 3: Use Facebook extractor directly
    print("\nUsing Facebook extractor directly:")
    extractor = FacebookExtractor()

    try:
        post_data = extractor.extract_post_data(post_url)
        print("Facebook Post Data:")
        print(json.dumps(post_data, indent=2))
    except Exception as e:
        print(f"Error extracting Facebook post data: {e}")

    try:
        page_data = extractor.extract_page_data(page_url)
        print("\nFacebook Page Data:")
        print(json.dumps(page_data, indent=2))
    except Exception as e:
        print(f"Error extracting Facebook page data: {e}")


if __name__ == "__main__":
    main()
