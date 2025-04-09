from typing import Any, Dict, List
import base64
import json
import pandas as pd
import requests
from io import BytesIO
import tempfile
import os
import numpy as np
import re

import openai

from .base_provider import BaseAIProvider
from ..config import OPENAI_API_KEY, OPENAI_DEFAULT_MODEL, OPENAI_VISION_MODEL, OPENAI_EMBEDDING_MODEL


class OpenAIProvider(BaseAIProvider):
    def __init__(self):
        self.client = None
        self.available_models = []

    def initialize(self) -> None:
        """Initialize OpenAI client"""
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.available_models = self.get_available_models()

    def process_text(self, text: str, **kwargs) -> str:
        """Process text using OpenAI"""
        model = kwargs.get('model', OPENAI_DEFAULT_MODEL)
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        return response.choices[0].message.content

    def process_image(self, image_path: str, **kwargs) -> str:
        """Process image using OpenAI's vision capabilities"""
        model = kwargs.get('model', OPENAI_VISION_MODEL)
        with open(image_path, 'rb') as image_file:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": kwargs.get('prompt', "What's in this image?")},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                                }
                            }
                        ]
                    }
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
        return response.choices[0].message.content

    def process_table(self, table_data: Any, **kwargs) -> Dict:
        """Process table data using OpenAI"""
        # Convert table data to string representation
        table_str = str(table_data)
        prompt = kwargs.get('prompt', f"Analyze this table data and return a structured JSON response: {table_str}")

        response = self.client.chat.completions.create(
            model=kwargs.get('model', OPENAI_DEFAULT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using OpenAI's embedding model"""
        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models"""
        models = self.client.models.list()
        return [model.id for model in models.data]
        
    async def process_excel(self, file_path: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file using OpenAI"""
        try:
            # Read Excel file into a pandas DataFrame
            df = pd.read_excel(file_path)
            
            # Clean the data by replacing special float values
            df = df.replace([np.inf, -np.inf], None)  # Replace infinity with None directly
            df = df.replace({col: {np.nan: None} for col in df.select_dtypes(include=['float64']).columns})  # Replace NaN with None for float columns
            
            # Convert float columns to handle out-of-range values
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else None)
            
            # Convert DataFrame to a string representation
            excel_str = df.to_string()
            
            # Create a prompt that includes the Excel data
            analysis_prompt = prompt or "Analyze this Excel data and provide insights."
            full_prompt = f"{analysis_prompt}\n\nExcel Data:\n{excel_str}"
            
            # Use the OpenAI model to analyze the data
            model_name = kwargs.get('model') if kwargs.get('model') else OPENAI_DEFAULT_MODEL
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a data analyst. Analyze the provided Excel data and provide insights."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Return both the analysis and the structured data
            return {
                "analysis": response.choices[0].message.content,
                "data": df.to_dict(orient='records')
            }

        except Exception as e:
            raise Exception(f"Error processing Excel with OpenAI: {str(e)}")
            
    async def process_excel(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process Excel file.
        
        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments including model and prompt
            
        Returns:
            Dictionary containing processed data and analysis
        """
        try:
            # Process the Excel file
            result = ExcelProcessor.process_excel(file_path)
            
            # Get the model from kwargs or use default
            model = kwargs.get('model') if kwargs.get('model') else OPENAI_DEFAULT_MODEL
            
            # Construct the prompt for analysis
            prompt = kwargs.get('prompt', '')
            if not prompt:
                prompt = f"""Analyze the following data and provide insights:
                Columns: {', '.join(result['columns'])}
                Number of rows: {len(result['data'])}
                """
            
            # Get analysis from OpenAI
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            
            # Add analysis to result
            result['analysis'] = response.choices[0].message.content
            
            return result
            
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
            
    async def process_excel_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Process Excel file from URL.
        
        Args:
            url: URL of the Excel file
            **kwargs: Additional arguments including model and prompt
            
        Returns:
            Dictionary containing processed data and analysis
        """
        try:
            # Download the Excel file
            file_path, content_type = ExcelProcessor.download_excel(url)
            
            # Process the Excel file
            result = ExcelProcessor.process_excel(file_path, content_type)
            
            # Get the model from kwargs or use default
            model = kwargs.get('model') if kwargs.get('model') else OPENAI_DEFAULT_MODEL
            
            # Construct the prompt for analysis
            prompt = kwargs.get('prompt', '')
            if not prompt:
                prompt = f"""Analyze the following data and provide insights:
                Columns: {', '.join(result['columns'])}
                Number of rows: {len(result['data'])}
                """
            
            # Get analysis from OpenAI
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            
            # Add analysis to result
            result['analysis'] = response.choices[0].message.content
            
            return result
            
        except Exception as e:
            raise Exception(f"Error processing Excel URL: {str(e)}")
            
    async def process_facebook_post(self, post_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Facebook post using OpenAI"""
        try:
            # Format the prompt to include both the instruction and the URL
            full_prompt = f"{prompt or 'Analyze this Facebook post and provide insights.'}\n\nFacebook Post URL: {post_url}"
            
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]

            completion = self.client.chat.completions.create(
                model=kwargs.get('model', OPENAI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}

        except Exception as e:
            raise Exception(f"Error processing Facebook post with OpenAI: {str(e)}")
            
    async def process_facebook_page(self, page_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Facebook page using OpenAI"""
        try:
            # Format the prompt to include both the instruction and the URL
            full_prompt = f"{prompt or 'Analyze this Facebook page and provide insights.'}\n\nFacebook Page URL: {page_url}"
            
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]

            completion = self.client.chat.completions.create(
                model=kwargs.get('model', OPENAI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}

        except Exception as e:
            raise Exception(f"Error processing Facebook page with OpenAI: {str(e)}")
