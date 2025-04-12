import os
import tempfile

import cv2
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


class TableTransformer:
    """
    A wrapper for the Microsoft Table Transformer model from Hugging Face.
    This model is specifically designed for detecting tables in documents.
    """

    def __init__(self, model_name="microsoft/table-transformer-detection", device=None):
        """
        Initialize the Table Transformer model.
        
        Args:
            model_name: The name of the model on Hugging Face Hub
            device: The device to run the model on (cuda, cpu, or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and processor
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        # Define the class labels
        self.id2label = self.model.config.id2label

    def detect_tables(self, image_path):
        """
        Detect tables in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            A list of dictionaries containing table bounding boxes and confidence scores
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to the same device as the model
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to COCO format
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]

        # Extract table detections
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.id2label[label.item()] == "table":
                tables.append({
                    "box": box.tolist(),
                    "score": score.item()
                })

        return tables

    def extract_table_regions(self, image_path, output_dir=None):
        """
        Extract table regions from an image and save them as separate images.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save the extracted table images (if None, uses a temporary directory)
            
        Returns:
            A list of paths to the extracted table images
        """
        # Detect tables
        tables = self.detect_tables(image_path)

        if not tables:
            return []

        # Load the original image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Create output directory if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Extract and save table regions
        table_images = []
        for i, table in enumerate(tables):
            # Get bounding box coordinates
            x1, y1, x2, y2 = table["box"]

            # Convert normalized coordinates to pixel coordinates
            x1, y1 = int(x1 * width), int(y1 * height)
            x2, y2 = int(x2 * width), int(y2 * height)

            # Extract the table region
            table_region = image[y1:y2, x1:x2]

            # Save the table region
            output_path = os.path.join(output_dir, f"table_{i + 1}.png")
            cv2.imwrite(output_path, table_region)

            table_images.append({
                "path": output_path,
                "box": table["box"],
                "score": table["score"]
            })

        return table_images

    def process_document(self, document_path, output_dir=None):
        """
        Process a document (PDF, image, etc.) to detect and extract tables.
        
        Args:
            document_path: Path to the document file
            output_dir: Directory to save the extracted table images
            
        Returns:
            A list of dictionaries containing information about detected tables
        """
        # Check if the document is a PDF
        if document_path.lower().endswith('.pdf'):
            # Convert PDF to image (first page only)
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(document_path, first_page=1, last_page=1)

                if not images:
                    return []

                # Save the first page as a temporary image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    images[0].save(temp_file.name, 'PNG')
                    temp_path = temp_file.name

                # Process the image
                result = self.extract_table_regions(temp_path, output_dir)

                # Clean up the temporary file
                os.unlink(temp_path)

                return result
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                return []
        else:
            # Process as an image
            return self.extract_table_regions(document_path, output_dir)
