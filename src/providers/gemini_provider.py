from typing import Any, Dict, List
import base64
import json
import pandas as pd
import requests
from io import BytesIO
import tempfile
import os
import numpy as np

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
        
    async def process_excel(self, file_path: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file using Gemini"""
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
            
            # Use the standard text completion API
            model = self.model
            response = model.generate_content(full_prompt)
            
            # Return both the analysis and the structured data
            return {
                "analysis": response.text,
                "data": df.to_dict(orient='records')
            }

        except Exception as e:
            raise Exception(f"Error processing Excel with Gemini: {str(e)}")
            
    async def process_excel_url(self, url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using Gemini"""
        try:
            # Check if the URL is a Google Sheets URL
            if "docs.google.com/spreadsheets" in url:
                # Extract the spreadsheet ID from the URL
                import re
                # Handle various Google Sheets URL formats
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
                        export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format={format}"
                        response = requests.get(export_url)
                        if response.status_code == 200:
                            excel_data = response.content
                            break
                    except Exception as e:
                        last_error = e
                        continue
                
                if not excel_data:
                    raise Exception(f"Failed to download Google Sheet data: {last_error}")
                
                # Save the downloaded data to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_formats[0]}") as temp_file:
                    temp_file.write(excel_data)
                    temp_file_path = temp_file.name
                
                try:
                    # Try to read the file as CSV first
                    try:
                        df = pd.read_csv(temp_file_path)
                    except Exception as csv_error:
                        # If CSV reading fails, try Excel engines
                        engines = ['openpyxl', 'xlrd']
                        df = None
                        last_error = None
                        
                        for engine in engines:
                            try:
                                df = pd.read_excel(temp_file_path, engine=engine)
                                break
                            except Exception as e:
                                last_error = e
                                continue
                        
                        if df is None:
                            raise Exception(f"Failed to read file with any method: {last_error}")
                    
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
                    model = self.client.get_model(kwargs.get('model', GEMINI_DEFAULT_MODEL))
                    response = model.generate_content(full_prompt)
                    
                    # Return both the analysis and the structured data
                    return {
                        "analysis": response.text,
                        "data": df.to_dict(orient='records')
                    }
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
            else:
                # For regular Excel URLs, download the file
                response = requests.get(url)
                if response.status_code != 200:
                    raise Exception(f"Failed to download Excel file: HTTP {response.status_code}")
                
                # Determine file type from content or URL
                content_type = response.headers.get('Content-Type', '')
                file_extension = url.split('.')[-1].lower() if '.' in url else ''
                
                # Save the downloaded data to a temporary file
                suffix = ".csv" if (content_type == 'text/csv' or file_extension == 'csv') else ".xlsx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                try:
                    # Try to read the file as CSV first
                    try:
                        df = pd.read_csv(temp_file_path)
                    except Exception as csv_error:
                        # If CSV reading fails, try Excel engines
                        engines = ['openpyxl', 'xlrd']
                        df = None
                        last_error = None
                        
                        for engine in engines:
                            try:
                                df = pd.read_excel(temp_file_path, engine=engine)
                                break
                            except Exception as e:
                                last_error = e
                                continue
                        
                        if df is None:
                            raise Exception(f"Failed to read file with any method: {last_error}")
                    
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
                    model = self.client.get_model(kwargs.get('model', GEMINI_DEFAULT_MODEL))
                    response = model.generate_content(full_prompt)
                    
                    # Return both the analysis and the structured data
                    return {
                        "analysis": response.text,
                        "data": df.to_dict(orient='records')
                    }
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)

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
