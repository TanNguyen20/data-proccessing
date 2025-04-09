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

import google.generativeai as genai
from PIL import Image

from .base_provider import BaseAIProvider
from ..config import GEMINI_API_KEY, GEMINI_DEFAULT_MODEL, GEMINI_VISION_MODEL


class GeminiProvider(BaseAIProvider):
    def __init__(self):
        self.client = None
        self.available_models = []

    def initialize(self) -> None:
        """Initialize Gemini client"""
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key not found in environment variables")
        genai.configure(api_key=GEMINI_API_KEY)
        self.available_models = self.get_available_models()

    def process_text(self, text: str, **kwargs) -> str:
        """Process text using Gemini"""
        model_name = kwargs.get('model', GEMINI_DEFAULT_MODEL)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            text,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_output_tokens=kwargs.get('max_tokens', 1000)
            )
        )
        return response.text

    def process_image(self, image_path: str, **kwargs) -> str:
        """Process image using Gemini's vision capabilities"""
        model_name = kwargs.get('model', GEMINI_VISION_MODEL)
        model = genai.GenerativeModel(model_name)
        
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        response = model.generate_content(
            [kwargs.get('prompt', "What's in this image?"), image_data],
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_output_tokens=kwargs.get('max_tokens', 1000)
            )
        )
        return response.text

    def process_table(self, table_data: Any, **kwargs) -> Dict:
        """Process table data using Gemini"""
        # Convert table data to string representation
        table_str = str(table_data)
        prompt = kwargs.get('prompt', f"Analyze this table data and return a structured JSON response: {table_str}")
        
        model_name = kwargs.get('model', GEMINI_DEFAULT_MODEL)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0.7),
                max_output_tokens=kwargs.get('max_tokens', 1000)
            )
        )
        return {"analysis": response.text}

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using Gemini's embedding model"""
        model = genai.GenerativeModel('embedding-001')
        response = model.embed_content(text)
        return response.embedding

    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models"""
        return ['gemini-pro', 'gemini-pro-vision']
        
    async def process_excel(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Process Excel file using Gemini"""
        try:
            # Read Excel file content
            with open(file_path, "rb") as excel_file:
                base64_excel = base64.b64encode(excel_file.read()).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": kwargs.get('prompt', "Analyze this Excel file and provide insights.")},
                        {
                            "type": "file_url",
                            "file_url": {
                                "url": f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64_excel}"
                            }
                        }
                    ]
                }
            ]

            completion = self.client.chat.completions.create(
                model=kwargs.get('model', GEMINI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}
        except Exception as e:
            raise Exception(f"Error processing Excel with Gemini: {str(e)}")
            
    async def process_excel_url(self, url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using Gemini"""
        try:
            # Check if the URL is a Google Sheets URL
            if "docs.google.com/spreadsheets" in url:
                # Extract the spreadsheet ID from the URL
                patterns = [
                    r'/d/([a-zA-Z0-9-_]+)',  # Standard format
                    r'id=([a-zA-Z0-9-_]+)',  # Sharing format
                    r'spreadsheets/d/([a-zA-Z0-9-_]+)',  # Alternative format
                    r'/([a-zA-Z0-9-_]{25,})'  # Fallback for long IDs
                ]
                
                spreadsheet_id = None
                for pattern in patterns:
                    match = re.search(pattern, url)
                    if match:
                        spreadsheet_id = match.group(1)
                        break
                
                if not spreadsheet_id:
                    raise Exception("Could not extract spreadsheet ID from Google Sheets URL")
                
                # Try different export formats and methods
                export_formats = ['csv', 'xlsx']
                excel_data = None
                last_error = None
                
                for format in export_formats:
                    try:
                        # Try direct export URL
                        export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format={format}"
                        response = requests.get(export_url)
                        
                        if response.status_code == 200:
                            excel_data = response.content
                            break
                        else:
                            # Try alternative export URL format
                            alt_export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:{format}"
                            alt_response = requests.get(alt_export_url)
                            
                            if alt_response.status_code == 200:
                                excel_data = alt_response.content
                                break
                    except Exception as e:
                        last_error = str(e)
                
                if not excel_data:
                    raise Exception(f"Error downloading Excel data: {last_error}")
                
                # Process the Excel data
                df = pd.read_excel(BytesIO(excel_data))
                
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
                
                # Use the Gemini model to analyze the data
                model_name = kwargs.get('model') if kwargs.get('model') else GEMINI_DEFAULT_MODEL
                model = self.client.get_model(model_name)
                response = model.generate_content(full_prompt)
                
                # Return both the analysis and the structured data
                return {
                    "analysis": response.text,
                    "data": df.to_dict(orient='records')
                }

        except Exception as e:
            raise Exception(f"Error processing Excel URL with Gemini: {str(e)}")
            
    async def process_facebook_post(self, post_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Facebook post using Gemini"""
        try:
            # Format the prompt to include both the instruction and the URL
            full_prompt = f"{prompt or 'Analyze this Facebook post and provide insights.'}\n\nFacebook Post URL: {post_url}"
            
            model_name = kwargs.get('model', GEMINI_DEFAULT_MODEL)
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    max_output_tokens=kwargs.get('max_tokens', 1000)
                )
            )
            
            return {"analysis": response.text}
            
        except Exception as e:
            raise Exception(f"Error processing Facebook post with Gemini: {str(e)}")

    async def process_facebook_page(self, page_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Facebook page using Gemini"""
        try:
            # Format the prompt to include both the instruction and the URL
            full_prompt = f"{prompt or 'Analyze this Facebook page and provide insights.'}\n\nFacebook Page URL: {page_url}"
            
            model_name = kwargs.get('model', GEMINI_DEFAULT_MODEL)
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    max_output_tokens=kwargs.get('max_tokens', 1000)
                )
            )
            
            return {"analysis": response.text}
            
        except Exception as e:
            raise Exception(f"Error processing Facebook page with Gemini: {str(e)}")
