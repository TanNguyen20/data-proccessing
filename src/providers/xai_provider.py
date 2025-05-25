import base64
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pdfplumber
import requests
from fastapi import UploadFile
from openai import OpenAI
from urllib.parse import urlparse

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
            # Check if required Excel engines are installed
            try:
                import openpyxl
                import xlrd
            except ImportError as e:
                raise Exception(f"Required Excel engine not installed: {str(e)}. Please install openpyxl and xlrd packages.")

            # Try different methods to read the file
            df = None
            last_error = None

            # First try to read as CSV with different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except Exception as e:
                    last_error = e
                    continue

            # If CSV reading fails, try Excel engines
            if df is None:
                engines = ['openpyxl', 'xlrd']
                for engine in engines:
                    try:
                        # For xlrd engine, we need to specify the engine explicitly
                        if engine == 'xlrd':
                            df = pd.read_excel(file_path, engine=engine)
                        else:
                            # For openpyxl, try with default settings first
                            try:
                                df = pd.read_excel(file_path, engine=engine)
                            except Exception:
                                # If that fails, try with additional parameters
                                df = pd.read_excel(
                                    file_path,
                                    engine=engine,
                                    sheet_name=0,  # Read first sheet
                                    header=None  # Don't assume header row
                                )
                        break
                    except Exception as e:
                        last_error = e
                        continue

            if df is None:
                # Provide more detailed error information
                error_msg = f"Failed to read file with any method. Last error: {str(last_error)}"
                if "Excel xlsx file; not supported" in str(last_error):
                    error_msg += "\nPlease ensure the file is a valid Excel file and not corrupted."
                raise Exception(error_msg)

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
                df = df.iloc[header_row + 1:].reset_index(drop=True)

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
                    {"role": "system",
                     "content": "You are a data analyst. Analyze the provided Excel data and provide insights."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
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
            # Check if required Excel engines are installed
            try:
                import openpyxl
                import xlrd
            except ImportError as e:
                raise Exception(f"Required Excel engine not installed: {str(e)}. Please install openpyxl and xlrd packages.")

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
                # Try different methods to read the file
                df = None
                last_error = None

                # First try to read as CSV with different encodings
                encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(temp_file_path, encoding=encoding)
                        break
                    except Exception as e:
                        last_error = e
                        continue

                # If CSV reading fails, try Excel engines
                if df is None:
                    engines = ['openpyxl', 'xlrd']
                    for engine in engines:
                        try:
                            # For xlrd engine, we need to specify the engine explicitly
                            if engine == 'xlrd':
                                df = pd.read_excel(temp_file_path, engine=engine)
                            else:
                                # For openpyxl, try with default settings first
                                try:
                                    df = pd.read_excel(temp_file_path, engine=engine)
                                except Exception:
                                    # If that fails, try with additional parameters
                                    df = pd.read_excel(
                                        temp_file_path,
                                        engine=engine,
                                        sheet_name=0,  # Read first sheet
                                        header=None  # Don't assume header row
                                    )
                            break
                        except Exception as e:
                            last_error = e
                            continue

                if df is None:
                    # Provide more detailed error information
                    error_msg = f"Failed to read file with any method. Last error: {str(last_error)}"
                    if "Excel xlsx file; not supported" in str(last_error):
                        error_msg += "\nPlease ensure the file is a valid Excel file and not corrupted."
                    raise Exception(error_msg)

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
                    df = df.iloc[header_row + 1:].reset_index(drop=True)

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
                        {"role": "system",
                         "content": "You are a data analyst. Analyze the provided Excel data and provide insights."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 1000)
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
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            raise Exception(f"Error processing Excel URL with xAI: {str(e)}")

    async def process_pdf(self, pdf_file: UploadFile, **kwargs) -> Dict[str, Any]:
        """Process PDF file using pdfplumber for table extraction and AI for formatting"""
        try:
            import pdfplumber
            import io

            # Read the PDF file content
            pdf_content = await pdf_file.read()
            pdf_file_obj = io.BytesIO(pdf_content)

            all_table_data = []
            full_text = ""
            page_headers = {}  # Store headers by page number
            first_table_headers = None  # Store headers from first table found

            # Open PDF with pdfplumber
            with pdfplumber.open(pdf_file_obj) as pdf:
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract page text
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"

                    try:
                        # Extract tables from the page
                        tables = page.extract_tables()

                        if tables:
                            for table_num, table in enumerate(tables, 1):
                                # Clean and validate table data
                                cleaned_table = await self._clean_table_data(table)

                                if cleaned_table:
                                    # Get appropriate headers for this table
                                    headers, is_header_row = await self._detect_and_validate_headers(
                                        cleaned_table,
                                        first_table_headers  # Pass headers from first table as fallback
                                    )

                                    # Store headers from first table found
                                    if first_table_headers is None and headers:
                                        first_table_headers = headers.copy()

                                    # Store headers for this page/table combination
                                    page_headers[f"page_{page_num}_table_{table_num}"] = headers

                                    # Skip header row if it was detected as a header
                                    table_data = cleaned_table[1:] if is_header_row else cleaned_table

                                    if table_data:  # Process only if we have data rows
                                        # Process table data with AI
                                        formatted_data = await self._format_table_data(
                                            headers,
                                            table_data,
                                            model=kwargs.get('model'),
                                            temperature=kwargs.get('temperature', 0.2),
                                            max_tokens=kwargs.get('max_tokens', 2000)
                                        )

                                        if formatted_data:
                                            # Add page and table information to each row
                                            for row in formatted_data:
                                                row['_page_number'] = page_num
                                                row['_table_number'] = table_num
                                            all_table_data.extend(formatted_data)

                    except Exception as table_error:
                        # Log the error but continue processing other tables
                        print(f"Error processing table on page {page_num}: {str(table_error)}")
                        continue

            # Get overall analysis
            analysis_response = self.client.chat.completions.create(
                model=kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a PDF document analyzer. Analyze this document and provide:
                        1. A summary of the key information
                        2. Any patterns or trends identified
                        3. Important data points
                        4. Document structure analysis
                        Make the analysis clear and concise."""
                    },
                    {"role": "user", "content": full_text}
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )

            return {
                "table_data": all_table_data,
                "analysis": analysis_response.choices[0].message.content,
                "page_count": len(pdf.pages),
                "row_count": len(all_table_data),
                "headers_by_page": page_headers  # Include header information for reference
            }

        except Exception as e:
            raise Exception(f"Error processing PDF with xAI: {str(e)}")

    async def process_pdf_from_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Process PDF file from URL using xAI
        
        Args:
            url: URL of the PDF file to process
            **kwargs: Additional arguments like model, temperature, prompt etc.
                - verify_ssl: Whether to verify SSL certificates (default: True)
            
        Returns:
            Dict containing:
                - table_data: List of extracted and formatted table data
                - analysis: Overall document analysis
                - page_count: Number of pages in the PDF
                - row_count: Total number of rows extracted
                - headers_by_page: Dictionary of headers found on each page
                - source_url: Original URL of the PDF file
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL provided")

            # Get verify_ssl parameter with default True
            verify_ssl = kwargs.get('verify_ssl', True)

            # Download PDF with streaming
            response = requests.get(url, stream=True, verify=verify_ssl)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' not in content_type:
                raise ValueError(f"URL does not point to a PDF file. Content-Type: {content_type}")

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                # Stream the content to the file
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_file.flush()

                # Read the PDF
                with pdfplumber.open(temp_file.name) as pdf:
                    # Extract text and tables
                    text = ""
                    tables = []
                    headers_by_page = {}
                    page_count = len(pdf.pages)
                    row_count = 0

                    for page_num, page in enumerate(pdf.pages, 1):
                        # Extract text
                        page_text = page.extract_text() or ""
                        text += f"\n--- Page {page_num} ---\n{page_text}"

                        # Extract tables
                        page_tables = page.extract_tables()
                        if page_tables:
                            for table in page_tables:
                                if table and len(table) > 1:  # Ensure table has data
                                    # Clean and format table data
                                    cleaned_table = []
                                    headers = []
                                    for row in table:
                                        # Clean each cell
                                        cleaned_row = []
                                        for cell in row:
                                            if cell is not None:
                                                # Convert to string and clean
                                                cell_str = str(cell).strip()
                                                # Remove special characters but keep Vietnamese
                                                cell_str = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', cell_str)
                                                cleaned_row.append(cell_str)
                                            else:
                                                cleaned_row.append("")
                                        
                                        if cleaned_row and any(cleaned_row):  # Skip empty rows
                                            if not headers:  # First non-empty row is headers
                                                headers = cleaned_row
                                                headers_by_page[page_num] = headers
                                            else:
                                                # Create row dict with headers
                                                row_dict = dict(zip(headers, cleaned_row))
                                                cleaned_table.append(row_dict)
                                                row_count += 1
                                    
                                    if cleaned_table:
                                        tables.extend(cleaned_table)

                    # Generate analysis using OpenAI
                    analysis_prompt = f"""Analyze this PDF document and provide a comprehensive summary:

Document Content:
{text}

Please provide:
1. A brief overview of the document
2. Key points and important information
3. Any notable patterns or trends in the data
4. Recommendations or conclusions

Format the response in a clear, structured way."""

                    analysis = self.client.chat.completions.create(
                        model=kwargs.get('model', 'gpt-3.5-turbo'),
                        messages=[{"role": "user", "content": analysis_prompt}],
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 1000)
                    ).choices[0].message.content

                    return {
                        "table_data": tables,
                        "analysis": analysis,
                        "page_count": page_count,
                        "row_count": row_count,
                        "headers_by_page": headers_by_page,
                        "source_url": url
                    }

        except requests.exceptions.SSLError as e:
            raise Exception(f"SSL certificate verification failed. Try setting verify_ssl=False: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading PDF: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        finally:
            # Clean up temporary file
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    async def _clean_table_data(self, table: List[List[Any]]) -> List[List[str]]:
        """Clean and validate table data before processing
        
        Args:
            table: Raw table data from pdfplumber
            
        Returns:
            Cleaned table data with properly formatted strings
        """
        cleaned_table = []
        for row in table:
            # Skip rows that are completely empty or None
            if not any(cell for cell in row if cell is not None):
                continue

            cleaned_row = []
            for cell in row:
                # Convert cell to string and clean it
                if cell is None:
                    cleaned_cell = ""
                else:
                    # Remove any problematic characters that might cause JSON issues
                    cleaned_cell = str(cell).strip()
                    # Replace any characters that might cause JSON parsing issues
                    cleaned_cell = (cleaned_cell
                                    .replace('"', "'")  # Replace double quotes with single quotes
                                    .replace('\n', ' ')  # Replace newlines with spaces
                                    .replace('\r', '')  # Remove carriage returns
                                    .replace('\t', ' ')  # Replace tabs with spaces
                                    .replace('\\', '/')  # Replace backslashes with forward slashes
                                    )
                    # Normalize whitespace
                    cleaned_cell = ' '.join(cleaned_cell.split())

                cleaned_row.append(cleaned_cell)

            # Only add rows that have at least one non-empty cell
            if any(cell for cell in cleaned_row):
                cleaned_table.append(cleaned_row)

        return cleaned_table

    async def _detect_and_validate_headers(self, table: List[List[str]], previous_headers: List[str] = None) -> Tuple[
        List[str], bool]:
        """Detect and validate table headers, using previous headers as fallback
        
        Args:
            table: The table data to analyze
            previous_headers: Headers from a previous table (optional)
            
        Returns:
            Tuple of (headers, is_header_row) where headers is the list of column names
            and is_header_row indicates if the first row was detected as a header
        """
        if not table or not table[0]:
            return previous_headers or [], False

        first_row = table[0]

        # Check if first row looks like a header
        non_empty_cells = [cell for cell in first_row if cell.strip()]
        if non_empty_cells:
            # Calculate metrics for header detection
            non_numeric_ratio = sum(1 for cell in non_empty_cells if not str(cell).replace('.', '').isdigit()) / len(
                non_empty_cells)
            unique_ratio = len(set(non_empty_cells)) / len(non_empty_cells)
            avg_length = sum(len(str(cell)) for cell in non_empty_cells) / len(non_empty_cells)

            is_header = (non_numeric_ratio > 0.7 and
                         unique_ratio > 0.8 and
                         3 <= avg_length <= 30)

            if is_header:
                # Clean and format headers
                headers = [
                    str(cell).strip() or f"Column_{i + 1}"
                    for i, cell in enumerate(first_row)
                ]

                # Make headers unique
                seen_headers = set()
                unique_headers = []
                for header in headers:
                    base_header = header
                    counter = 1
                    while header in seen_headers:
                        header = f"{base_header}_{counter}"
                        counter += 1
                    seen_headers.add(header)
                    unique_headers.append(header)

                return unique_headers, True

        # If no valid header detected, use previous headers or generate generic ones
        if previous_headers and len(previous_headers) >= len(first_row):
            return previous_headers[:len(first_row)], False
        else:
            # Generate generic headers
            return [f"Column_{i + 1}" for i in range(len(first_row))], False

    async def _format_table_data(self, headers: List[str], rows: List[List[str]], **kwargs) -> List[Dict[str, str]]:
        """Format extracted table data using AI"""
        try:
            # Create a more structured format for the data to minimize JSON parsing issues
            system_prompt = """You are a data formatting specialist. Your task is to:
            1. Format the provided table data as a JSON array
            2. Each object in the array should have the exact column headers as keys
            3. Values should be properly escaped strings
            4. Ensure all quotes and special characters are properly escaped
            5. Return ONLY valid JSON data with no additional text or formatting
            
            IMPORTANT: Your response must be ONLY the JSON array, properly formatted and escaped."""

            # Convert the rows to a more structured format
            formatted_rows = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    if i < len(headers):
                        header = headers[i]
                        # Clean and escape the value
                        cleaned_value = str(value).strip().replace('"', '\\"').replace('\n', ' ').replace('\r', '')
                        row_dict[header] = cleaned_value
                formatted_rows.append(row_dict)

            # Create a simpler prompt with pre-formatted data
            user_prompt = f"""Format this data as a valid JSON array. Use these exact column headers: {headers}
            
            Pre-formatted data (already in dictionary format):
            {json.dumps(formatted_rows, ensure_ascii=False, indent=2)}
            
            Return ONLY the JSON array with proper formatting and escaping."""

            # Use the AI model to format the data
            response = self.client.chat.completions.create(
                model=kwargs.get('model') if kwargs.get('model') else XAI_DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent formatting
                max_tokens=kwargs.get('max_tokens', 4000),
                response_format={"type": "json_object"}
            )

            try:
                # Get and clean the response content
                content = response.choices[0].message.content.strip()

                # Remove any markdown code block markers
                content = re.sub(r'^```json\s*|\s*```$', '', content)

                # Try to fix common JSON formatting issues
                content = content.replace('\n', ' ').replace('\r', '')  # Remove newlines
                content = re.sub(r',\s*]', ']', content)  # Remove trailing commas
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

                # If content doesn't start with '[', try to find the array
                if not content.startswith('['):
                    array_start = content.find('[')
                    array_end = content.rfind(']')
                    if array_start != -1 and array_end != -1:
                        content = content[array_start:array_end + 1]

                # Parse the JSON
                result = json.loads(content)

                # Ensure we have a list
                if not isinstance(result, list):
                    if isinstance(result, dict):
                        # If we got a dict with a data/rows field, try to extract it
                        for key in ['data', 'rows', 'table', 'results']:
                            if key in result and isinstance(result[key], list):
                                result = result[key]
                                break
                        if not isinstance(result, list):
                            result = [result]  # If still not a list, wrap it

                # If parsing succeeded but we got an empty result, return the pre-formatted data
                if not result and formatted_rows:
                    return formatted_rows

                # Validate and standardize the structure
                standardized_result = []
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    # Create a new dict with all headers
                    row_dict = {}
                    for header in headers:
                        # Get the value, using an empty string as default
                        value = str(item.get(header, "")).strip()
                        row_dict[header] = value
                    standardized_result.append(row_dict)

                # If we lost data in standardization, return the pre-formatted data
                if len(standardized_result) < len(formatted_rows):
                    return formatted_rows

                return standardized_result

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract valid JSON using regex
                try:
                    # Look for array pattern
                    array_pattern = r'\[\s*{[^}]*}\s*(,\s*{[^}]*}\s*)*\]'
                    matches = re.findall(array_pattern, content)
                    if matches:
                        content = matches[0]
                        result = json.loads(content)
                        if isinstance(result, list):
                            return result
                except:
                    pass

                # If all parsing attempts fail, return the pre-formatted data
                return formatted_rows

        except Exception as e:
            # Instead of raising an exception, return the pre-formatted data
            return formatted_rows

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
            non_numeric_ratio = sum(
                1 for val in row if pd.notnull(val) and not str(val).replace('.', '').isdigit()) / len(row)
            unique_values_ratio = len(set(str(val).strip() for val in row if pd.notnull(val))) / len(row)
            avg_value_length = sum(len(str(val).strip()) for val in row if pd.notnull(val)) / max(1, sum(
                pd.notnull(val) for val in row))

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
                    is_sequential = all(abs(nums[i] - nums[i - 1]) <= 5 for i in range(1, len(nums)))
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
            (str(v).startswith(('0', '+84', '*')) and len(
                str(v).replace('+84', '0').replace('*', '').replace(' ', '')) in (9, 10, 11))
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

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on the given prompt using xAI
        
        Args:
            prompt (str): The prompt to generate text from
            **kwargs: Additional arguments like temperature, max_tokens, etc.
            
        Returns:
            str: The generated text
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            completion = self.client.chat.completions.create(
                model=kwargs.get('model', XAI_DEFAULT_MODEL),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error generating text with xAI: {str(e)}")
