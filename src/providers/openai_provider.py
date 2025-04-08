from typing import Any, Dict, List
import base64
import json
import pandas as pd
import requests
from io import BytesIO
import tempfile
import os
import numpy as np

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
            
            # Use the standard text completion API
            completion = self.client.chat.completions.create(
                model=kwargs.get('model', OPENAI_DEFAULT_MODEL),
                messages=[{"role": "user", "content": full_prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            
            # Return both the analysis and the structured data
            return {
                "analysis": completion.choices[0].message.content,
                "data": df.to_dict(orient='records')
            }

        except Exception as e:
            raise Exception(f"Error processing Excel with OpenAI: {str(e)}")
            
    async def process_excel_url(self, url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using OpenAI"""
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
                    
                    # Use the standard text completion API
                    response = self.client.chat.completions.create(
                        model=kwargs.get('model', OPENAI_DEFAULT_MODEL),
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 1000)
                    )
                    
                    # Return both the analysis and the structured data
                    return {
                        "analysis": response.choices[0].message.content,
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
                    
                    # Use the standard text completion API
                    response = self.client.chat.completions.create(
                        model=kwargs.get('model', OPENAI_DEFAULT_MODEL),
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 1000)
                    )
                    
                    # Return both the analysis and the structured data
                    return {
                        "analysis": response.choices[0].message.content,
                        "data": df.to_dict(orient='records')
                    }
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)

        except Exception as e:
            raise Exception(f"Error processing Excel URL with OpenAI: {str(e)}")
            
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
