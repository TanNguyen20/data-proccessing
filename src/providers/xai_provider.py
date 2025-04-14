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
                model=kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL,
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
            # Try to read the file as CSV first
            try:
                df = pd.read_csv(file_path)
            except Exception as csv_error:
                # If CSV reading fails, try Excel engines
                engines = ['openpyxl', 'xlrd']
                df = None
                last_error = None
                
                for engine in engines:
                    try:
                        df = pd.read_excel(file_path, engine=engine)
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
        Detect the header row by analyzing data patterns, without making assumptions about specific column names.
        
        Args:
            df: The DataFrame to analyze
            max_rows_to_check: Maximum number of rows to check for header
            
        Returns:
            The index of the likely header row
        """
        if df.empty:
            return 0

        # Check the first few rows
        for i in range(min(max_rows_to_check, len(df))):
            row = df.iloc[i]
            
            # Skip completely empty rows
            if row.isna().all():
                continue
                
            # Calculate metrics for this row
            non_numeric_ratio = sum(1 for val in row if pd.notnull(val) and not str(val).replace('.', '').isdigit()) / len(row)
            unique_values_ratio = len(set(str(val).strip() for val in row if pd.notnull(val))) / len(row)
            avg_value_length = sum(len(str(val).strip()) for val in row if pd.notnull(val)) / max(1, sum(pd.notnull(val) for val in row))
            
            # A good header row typically has:
            # 1. High ratio of non-numeric values (column names are usually text)
            # 2. High ratio of unique values (column names should be unique)
            # 3. Reasonable average length (not too short, not too long)
            if (non_numeric_ratio > 0.7 and 
                unique_values_ratio > 0.8 and 
                3 <= avg_value_length <= 30):
                return i
        
        # If no clear header is found, use the first non-empty row
        for i in range(min(max_rows_to_check, len(df))):
            if not df.iloc[i].isna().all():
                return i
        
        return 0

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
        Minimally clean column names while preserving original names.
        Only handles empty columns and ensures uniqueness.
        """
        # Drop empty columns
        df = df.dropna(how='all', axis=1)

        # Clean basic formatting while preserving original names
        cleaned_columns = []
        for col in df.columns:
            clean_name = str(col).strip()
            if not clean_name:  # Only generate generic names for empty columns
                clean_name = f"column_{len(cleaned_columns)}"
            cleaned_columns.append(clean_name)

        # Handle duplicate column names if any exist
        if len(cleaned_columns) != len(set(cleaned_columns)):
            seen = {}
            unique_columns = []
            for col in cleaned_columns:
                if col in seen:
                    seen[col] += 1
                    unique_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    unique_columns.append(col)
            cleaned_columns = unique_columns

        df.columns = cleaned_columns
        return df

    def _detect_column_type(self, column_values) -> str:
        """
        Detect column type based on data analysis.
        Returns a type identifier based on data patterns.
        """
        values = [str(x).strip() for x in column_values if pd.notnull(x)]
        if not values:
            return None

        # Calculate ratios of different patterns in the data
        patterns = {
            'numeric': sum(str(v).replace('.', '').isdigit() for v in values) / len(values),
            'multi_word': sum(len(str(v).split()) > 1 for v in values) / len(values),
            'class_code': sum(str(v).upper().startswith(('DH', 'TH')) for v in values) / len(values)
        }

        # Check for sequential numbers
        if patterns['numeric'] > 0.9:
            try:
                nums = [float(v) for v in values]
                if len(nums) >= 2:
                    is_sequential = all(abs(nums[i] - nums[i-1]) <= 5 for i in range(1, len(nums)))
                    if is_sequential:
                        return 'sequence'
            except ValueError:
                pass

        # Check for student ID pattern 
        if patterns['numeric'] > 0.9 and all(7 <= len(str(v).replace('.', '')) <= 9 for v in values):
            return 'student_id'

        # Check for class codes
        if patterns['class_code'] > 0.8:
            return 'class'

        # Check for phone numbers
        phone_pattern = sum(
            (str(v).startswith(('0', '+84', '*')) and len(str(v).replace('+84', '0').replace('*', '').replace(' ', '')) in (9, 10, 11))
            or (str(v).isdigit() and len(str(v)) in (9, 10))
            for v in values
        ) / len(values)
        if phone_pattern > 0.8:
            return 'contact'

        # Check for names
        if patterns['multi_word'] > 0.8:
            return 'name'

        return None

    def _convert_df_to_json_safe_dict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to JSON format preserving original column names.
        """
        if df.empty:
            return []

        data = []
        for _, row in df.iterrows():
            row_dict = {}
            for col_name, value in row.items():
                # Clean and format value
                clean_value = str(value).strip() if pd.notnull(value) else None
                # Use original column name
                row_dict[str(col_name)] = clean_value
            data.append(row_dict)
            
        return data
