from typing import List, Tuple

import torch
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection


class TableDetector:
    """Table detection using Microsoft's Table Transformer model"""

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.confidence_threshold = 0.7

    def detect_tables(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect tables in an image and return their bounding boxes
        
        Args:
            image: PIL Image object
            
        Returns:
            List of tuples containing bounding box coordinates (x_min, y_min, x_max, y_max)
        """
        # Prepare image for the model
        inputs = self.processor(images=image, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to normalized coordinates
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]

        # Extract bounding boxes
        boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i) for i in box.tolist()]
            boxes.append(tuple(box))

        return boxes

    def extract_tables(self, image: Image.Image) -> List[Image.Image]:
        """
        Extract table regions from the image
        
        Args:
            image: PIL Image object
            
        Returns:
            List of PIL Image objects containing individual tables
        """
        # Detect tables
        boxes = self.detect_tables(image)

        # Crop and return table regions
        tables = []
        for box in boxes:
            table_region = image.crop(box)
            tables.append(table_region)

        return tables

    def process_image(self, image_path: str) -> List[Image.Image]:
        """
        Process an image file and extract tables
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of PIL Image objects containing detected tables
        """
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Extract tables
            return self.extract_tables(image)

        except Exception as e:
            print(f"Error processing image: {e}")
            return []
