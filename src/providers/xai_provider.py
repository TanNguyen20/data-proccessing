import base64
import json
import pandas as pd
import requests
from io import BytesIO
from typing import Any, Dict, List
import tempfile
import os
import numpy as np
import re

from openai import OpenAI

from .base_provider import BaseAIProvider
from ..config import XAI_API_KEY, XAI_BASE_URL, XAI_DEFAULT_MODEL, XAI_VISION_MODEL, XAI_EMBEDDING_MODEL


class XAIProvider(BaseAIProvider):
    """xAI Provider implementation using OpenAI client"""

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            api_key=XAI_API_KEY,
            base_url=XAI_BASE_URL
        )
        
    def initialize(self) -> None:
        """Initialize xAI client"""
        # No additional initialization needed as it's done in __init__
        pass
        
    def process_text(self, text: str, **kwargs) -> str:
        """Process text using xAI's text model"""
        try:
            messages = []
            if kwargs.get('system_prompt'):
                messages.append({"role": "system", "content": kwargs.get('system_prompt')})
            messages.append({"role": "user", "content": text})

            completion = self.client.chat.completions.create(
                model=kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return completion.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error processing text with xAI: {str(e)}")

    def process_image(self, image_path: str, **kwargs) -> str:
        """Process image using xAI's vision model"""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": kwargs.get('prompt', "What's in this image?")},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            completion = self.client.chat.completions.create(
                model=kwargs.get('model', XAI_VISION_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return completion.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error processing image with xAI: {str(e)}")
            
    def process_table(self, table_data: Any, **kwargs) -> Dict:
        """Process table data using xAI"""
        try:
            # Convert table data to string representation
            table_str = str(table_data)
            prompt = kwargs.get('prompt', f"Analyze this table data and return a structured JSON response: {table_str}")

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            completion = self.client.chat.completions.create(
                model=kwargs.get('model', XAI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}

        except Exception as e:
            raise Exception(f"Error processing table with xAI: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using xAI's embedding model"""
        try:
            response = self.client.embeddings.create(
                model=XAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"Error getting embeddings with xAI: {str(e)}")
            
    def get_available_models(self) -> List[str]:
        """Get list of available xAI models"""
        # Based on the documentation, these are the available models
        return [
            "grok-1",
            "grok-1-vision",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using xAI's embedding model"""
        try:
            response = self.client.embeddings.create(
                model=XAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            raise Exception(f"Error getting embeddings with xAI: {str(e)}")

    async def process_excel(self, file_path: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file using xAI"""
        try:
            # Read Excel file content
            with open(file_path, "rb") as excel_file:
                base64_excel = base64.b64encode(excel_file.read()).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Analyze this Excel file and provide insights."},
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
                model=kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}

        except Exception as e:
            raise Exception(f"Error processing Excel with xAI: {str(e)}")

    async def process_excel_url(self, url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Excel file from URL using xAI"""
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
                            else:
                                last_error = f"HTTP {response.status_code} for {format} format"
                    except Exception as e:
                        last_error = str(e)
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
                    # First, replace infinity values with None
                    df = df.replace([np.inf, -np.inf], None)
                    
                    # Then, replace NaN values with None for all columns
                    df = df.where(pd.notnull(df), None)
                    
                    # Convert all numeric columns to strings to handle any remaining out-of-range values
                    for col in df.select_dtypes(include=['float64', 'int64']).columns:
                        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else None)
                    
                    # Detect header row dynamically
                    header_row = self._detect_header_row(df)
                    
                    if header_row is not None:
                        # Use this row as the header
                        df.columns = df.iloc[header_row]
                        # Remove the header row and any rows before it
                        df = df.iloc[header_row+1:].reset_index(drop=True)
                    
                    # Handle merged columns in the header
                    df = self._handle_merged_columns(df)
                    
                    # Clean column names
                    df = self._clean_column_names(df)
                    
                    # Convert DataFrame to a string representation for the prompt
                    excel_str = df.to_string()
                    
                    # Create a prompt that includes the Excel data
                    analysis_prompt = prompt or "Analyze this Excel data and provide insights."
                    full_prompt = f"{analysis_prompt}\n\nExcel Data:\n{excel_str}"
                    
                    # Use the xAI model to analyze the data
                    model_name = kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Analyze the provided Excel data and provide insights."},
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                    
                    # Convert DataFrame to dict with proper handling of special values
                    data_dict = self._convert_df_to_json_safe_dict(df)
                    
                    # Return both the analysis and the structured data
                    return {
                        "analysis": response.choices[0].message.content,
                        "data": data_dict
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
                    # First, replace infinity values with None
                    df = df.replace([np.inf, -np.inf], None)
                    
                    # Then, replace NaN values with None for all columns
                    df = df.where(pd.notnull(df), None)
                    
                    # Convert all numeric columns to strings to handle any remaining out-of-range values
                    for col in df.select_dtypes(include=['float64', 'int64']).columns:
                        df[col] = df[col].apply(lambda x: str(x) if pd.notnull(x) else None)
                    
                    # Detect header row dynamically
                    header_row = self._detect_header_row(df)
                    
                    if header_row is not None:
                        # Use this row as the header
                        df.columns = df.iloc[header_row]
                        # Remove the header row and any rows before it
                        df = df.iloc[header_row+1:].reset_index(drop=True)
                    
                    # Handle merged columns in the header
                    df = self._handle_merged_columns(df)
                    
                    # Clean column names
                    df = self._clean_column_names(df)
                    
                    # Convert DataFrame to a string representation for the prompt
                    excel_str = df.to_string()
                    
                    # Create a prompt that includes the Excel data
                    analysis_prompt = prompt or "Analyze this Excel data and provide insights."
                    full_prompt = f"{analysis_prompt}\n\nExcel Data:\n{excel_str}"
                    
                    # Use the xAI model to analyze the data
                    model_name = kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Analyze the provided Excel data and provide insights."},
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                    
                    # Convert DataFrame to dict with proper handling of special values
                    data_dict = self._convert_df_to_json_safe_dict(df)
                    
                    # Return both the analysis and the structured data
                    return {
                        "analysis": response.choices[0].message.content,
                        "data": data_dict
                    }
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)

        except Exception as e:
            raise Exception(f"Error processing Excel URL with xAI: {str(e)}")
            
    def _detect_header_row(self, df: pd.DataFrame, max_rows_to_check: int = 10) -> int:
        """
        Dynamically detect the header row in a DataFrame.
        
        Args:
            df: The DataFrame to analyze
            max_rows_to_check: Maximum number of rows to check for header
            
        Returns:
            The index of the header row, or None if no header row is found
        """
        # First, check if the first row contains valid string headers
        first_row = df.iloc[0]
        if all(isinstance(val, str) and len(val.strip()) > 0 for val in first_row if pd.notnull(val)):
            return 0
            
        # Check subsequent rows for valid headers
        for i in range(1, min(max_rows_to_check, len(df))):
            row_values = df.iloc[i].values
            # Check if this row contains mostly string values and has meaningful content
            string_count = sum(1 for val in row_values if isinstance(val, str) and len(val.strip()) > 0)
            if string_count >= len(row_values) * 0.7:  # At least 70% of values are strings
                # Additional check for meaningful header content
                meaningful_content = any(
                    isinstance(val, str) and len(val.strip()) > 0
                    for val in row_values if pd.notnull(val)
                )
                if meaningful_content:
                    return i
                
        # If no clear header row is found, return None
        return None
        
    def _handle_merged_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle merged cells in the header by splitting long column names.
        
        Args:
            df: The DataFrame with potentially merged column headers
            
        Returns:
            DataFrame with properly split column names
        """
        # Check for columns with very long names that might be merged
        for col in df.columns:
            if isinstance(col, str) and len(col) > 30:
                # Try to split the column name based on common delimiters
                parts = re.split(r'[\s,;|]+', col)
                if len(parts) > 1:
                    # Create new columns for each part
                    for i, part in enumerate(parts):
                        if i == 0:
                            # Replace the original column name
                            df = df.rename(columns={col: part})
                        else:
                            # Add a new column with the part as the name
                            df[part] = None
        
        return df
        
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing whitespace and handling duplicates.
        
        Args:
            df: The DataFrame with column names to clean
            
        Returns:
            DataFrame with cleaned column names
        """
        # Strip whitespace from column names
        df.columns = [str(col).strip() if isinstance(col, str) else col for col in df.columns]
        
        # Handle duplicate column names
        seen = {}
        new_columns = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df.columns = new_columns
        return df
        
    def _convert_df_to_json_safe_dict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to a JSON-safe dictionary with proper handling of special values.
        
        Args:
            df: The DataFrame to convert
            
        Returns:
            List of dictionaries representing the DataFrame rows
        """
        # Create a list to store the rows
        data_dict = []
        
        # Get column names, ensuring they are strings and properly formatted
        columns = [str(col).strip() if isinstance(col, str) else f"Column_{i}" 
                  for i, col in enumerate(df.columns)]
        
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            # Create a dictionary for the current row
            row_dict = {}
            
            # Iterate through each column in the row
            for col_name, val in zip(columns, row):
                # Clean the column name
                clean_col_name = col_name.strip()
                
                # Handle special values
                if pd.isna(val) or val is None:
                    row_dict[clean_col_name] = None
                elif isinstance(val, (int, float)):
                    if np.isinf(val) or np.isnan(val):
                        row_dict[clean_col_name] = None
                    else:
                        # Convert numeric values to strings to ensure JSON compatibility
                        row_dict[clean_col_name] = str(val)
                else:
                    # For string values, ensure they are properly formatted
                    row_dict[clean_col_name] = str(val).strip() if isinstance(val, str) else str(val)
            
            # Add the row dictionary to the list
            data_dict.append(row_dict)
        
        return data_dict

    async def process_facebook_post(self, post_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Facebook post using xAI"""
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
                model=kwargs.get('model', XAI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}

        except Exception as e:
            raise Exception(f"Error processing Facebook post with xAI: {str(e)}")

    async def process_facebook_page(self, page_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Process Facebook page using xAI"""
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
                model=kwargs.get('model', XAI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return {"analysis": completion.choices[0].message.content}

        except Exception as e:
            raise Exception(f"Error processing Facebook page with xAI: {str(e)}")
