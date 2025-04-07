import base64
import json
import pandas as pd
import requests
from io import BytesIO
from typing import Any, Dict, List

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
                model=kwargs.get('model', XAI_DEFAULT_MODEL),
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
                model=kwargs.get('model', XAI_DEFAULT_MODEL),
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
                last_status_code = None
                last_response = None
                
                # First try the standard export URL
                for format in export_formats:
                    try:
                        # Construct the export URL
                        export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format={format}"
                        
                        # Download the file
                        response = requests.get(export_url)
                        last_status_code = response.status_code
                        last_response = response
                        
                        if response.status_code == 403:
                            raise Exception("Access denied. The Google Sheet may be private or require authentication.")
                        elif response.status_code == 404:
                            raise Exception("Sheet not found. Please check if the URL is correct and the sheet exists.")
                        elif response.status_code != 200:
                            continue
                            
                        # Read the data based on format
                        if format == 'csv':
                            excel_data = pd.read_csv(BytesIO(response.content))
                        else:  # xlsx
                            excel_data = pd.read_excel(BytesIO(response.content), engine='openpyxl')
                            
                        if excel_data is not None and not excel_data.empty:
                            break
                    except Exception as e:
                        last_error = e
                        continue
                
                # If standard export failed, try alternative methods
                if excel_data is None:
                    try:
                        # Try the gid parameter if present in the URL
                        gid_match = re.search(r'gid=(\d+)', url)
                        gid = gid_match.group(1) if gid_match else '0'
                        
                        # Try the alternative export URL format
                        alt_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
                        response = requests.get(alt_url)
                        last_status_code = response.status_code
                        last_response = response
                        
                        if response.status_code == 200:
                            excel_data = pd.read_csv(BytesIO(response.content))
                    except Exception as e:
                        last_error = e
                
                if excel_data is None:
                    error_msg = "Failed to download Google Sheets data. "
                    if last_status_code:
                        error_msg += f"Last HTTP status code: {last_status_code}. "
                    if last_response and last_response.text:
                        error_msg += f"Response content: {last_response.text[:200]}... "
                    if last_error:
                        error_msg += f"Last error: {str(last_error)}"
                    else:
                        error_msg += "No data could be read from the sheet."
                    raise Exception(error_msg)
            else:
                # For regular Excel files, download and read as before
                response = requests.get(url)
                if response.status_code != 200:
                    raise Exception(f"Failed to download Excel file: HTTP {response.status_code}")
                
                # Try different engines in order of preference
                engines = ['openpyxl', 'xlrd']
                excel_data = None
                last_error = None
                
                for engine in engines:
                    try:
                        excel_data = pd.read_excel(BytesIO(response.content), engine=engine)
                        break
                    except Exception as e:
                        last_error = e
                        continue
                
                if excel_data is None:
                    raise Exception(f"Failed to read Excel file with any engine. Last error: {str(last_error)}")
            
            # Improve table detection
            # First, try to find the header row by looking for common header patterns
            header_row_index = None
            
            # Look for rows that contain all the expected headers
            expected_headers = ["STT", "Họ và Tên", "MSSV", "Lớp", "SDT"]
            for i in range(min(10, len(excel_data))):  # Check first 10 rows
                row_values = [str(val).strip() if pd.notna(val) else "" for val in excel_data.iloc[i]]
                # Check if this row contains all expected headers
                if all(header in row_values for header in expected_headers):
                    header_row_index = i
                    break
            
            # If no header row found with expected headers, try common patterns
            if header_row_index is None:
                for i in range(min(10, len(excel_data))):  # Check first 10 rows
                    row = excel_data.iloc[i]
                    # Check if this row has values that look like headers (non-numeric, not empty)
                    if all(isinstance(val, str) and val.strip() and not val.replace('.', '').isdigit() 
                          for val in row if pd.notna(val)):
                        header_row_index = i
                        break
            
            # If no header row found, use the first non-empty row
            if header_row_index is None:
                for i in range(len(excel_data)):
                    row = excel_data.iloc[i]
                    if any(pd.notna(val) for val in row):
                        header_row_index = i
                        break
            
            # If still no header row found, use the first row
            if header_row_index is None:
                header_row_index = 0
            
            # Extract headers from the identified header row
            headers = []
            for col_idx in range(len(excel_data.columns)):
                val = excel_data.iloc[header_row_index, col_idx]
                if pd.isna(val) or val == '':
                    # If header is empty, use a generic name
                    headers.append(f"Column_{col_idx+1}")
                else:
                    headers.append(str(val).strip())
            
            # Create a new DataFrame with the correct headers
            data_start_row = header_row_index + 1
            clean_data = excel_data.iloc[data_start_row:].copy()
            clean_data.columns = headers
            
            # Remove rows where all values are NaN
            clean_data = clean_data.dropna(how='all')
            
            # Remove rows that look like headers or titles
            clean_data = clean_data[~clean_data.apply(lambda row: 
                all(isinstance(val, str) and val.strip() and not val.replace('.', '').isdigit() 
                    for val in row if pd.notna(val)), axis=1)]
            
            # Extract table structure
            table_info = {
                "columns": headers,
                "row_count": len(clean_data),
                "column_count": len(headers),
                "data": []
            }
            
            # Add data rows (limit to first 100 rows for large tables)
            max_rows_to_include = min(100, len(clean_data))
            for i in range(max_rows_to_include):
                row_data = {}
                for col_idx, col_name in enumerate(clean_data.columns):
                    val = clean_data.iloc[i, col_idx]
                    # Convert NaN to null for JSON
                    row_data[col_name] = None if pd.isna(val) else val
                table_info["data"].append(row_data)
            
            # If a prompt is provided, use the AI to analyze the data
            if prompt:
                # Create a summary of the table structure
                table_summary = f"Table with {table_info['row_count']} rows and {table_info['column_count']} columns.\n"
                table_summary += f"Columns: {', '.join(table_info['columns'])}\n\n"
                
                # Add sample data (first 5 rows)
                sample_data = table_info["data"][:5]
                sample_data_str = json.dumps(sample_data, indent=2)
                
                full_prompt = f"{prompt}\n\n{table_summary}Sample data (first 5 rows):\n{sample_data_str}"
                
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
                
                return {
                    "table": table_info,
                    "analysis": completion.choices[0].message.content
                }
            
            # If no prompt, just return the table structure
            return {"table": table_info}

        except Exception as e:
            raise Exception(f"Error processing Excel URL with xAI: {str(e)}")

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
