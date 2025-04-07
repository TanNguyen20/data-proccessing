import json
import os
import sys

from dotenv import load_dotenv

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import AIProcessor


def process_excel_file(processor, file_path):
    """Process an Excel file using the AI processor"""
    print(f"\nProcessing Excel file: {file_path}")
    try:
        result = processor.process_excel(file_path, extract_tables=True)
        print("Excel Analysis Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing Excel file: {e}")


def process_image(processor, image_path):
    """Process an image using the AI processor"""
    print(f"\nProcessing image: {image_path}")
    try:
        result = processor.process_image(image_path)
        print("Image Analysis Result:")
        print(result)
    except Exception as e:
        print(f"Error processing image: {e}")


def process_text(processor, text):
    """Process text using the AI processor"""
    print(f"\nProcessing text: {text}")
    try:
        result = processor.process_text(text)
        print("Text Analysis Result:")
        print(result)
    except Exception as e:
        print(f"Error processing text: {e}")


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the processor with xAI provider
    processor = AIProcessor(provider='xai')

    # Example 1: Process Excel file
    excel_file = "data/sample.xlsx"  # Replace with your Excel file path
    if os.path.exists(excel_file):
        process_excel_file(processor, excel_file)
    else:
        print(f"Excel file not found: {excel_file}")

    # Example 2: Process image
    image_file = "data/sample.jpg"  # Replace with your image file path
    if os.path.exists(image_file):
        process_image(processor, image_file)
    else:
        print(f"Image file not found: {image_file}")

    # Example 3: Process text
    text = "Analyze the following data: The company's revenue increased by 25% in Q1 2023, but decreased by 10% in Q2 2023. Employee count remained stable at 500 throughout the year."
    process_text(processor, text)

    # Example 4: Process table data
    table_data = {
        "columns": ["Product", "Q1 Sales", "Q2 Sales", "Q3 Sales", "Q4 Sales"],
        "data": [
            {"Product": "Widget A", "Q1 Sales": 100, "Q2 Sales": 120, "Q3 Sales": 150, "Q4 Sales": 180},
            {"Product": "Widget B", "Q1 Sales": 200, "Q2 Sales": 220, "Q3 Sales": 250, "Q4 Sales": 280},
            {"Product": "Widget C", "Q1 Sales": 300, "Q2 Sales": 320, "Q3 Sales": 350, "Q4 Sales": 380}
        ]
    }
    print("\nProcessing table data:")
    try:
        result = processor.process_text(f"Analyze this sales data and provide insights: {json.dumps(table_data)}")
        print("Table Analysis Result:")
        print(result)
    except Exception as e:
        print(f"Error processing table data: {e}")


if __name__ == "__main__":
    main()
