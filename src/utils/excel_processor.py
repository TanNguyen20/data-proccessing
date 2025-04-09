import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from urllib.parse import urlparse
import requests
import tempfile

class ExcelProcessor:
    """Utility class for processing Excel files with dynamic table structure detection."""
    
    @staticmethod
    def download_excel(url: str) -> Tuple[str, str]:
        """
        Download Excel file from URL and return temporary file path and content type.
        
        Args:
            url: URL of the Excel file
            
        Returns:
            Tuple of (temporary file path, content type)
        """
        response = requests.get(url, stream=True)
        content_type = response.headers.get('content-type', '')
        
        # Create a temporary file with appropriate extension
        ext = '.xlsx'  # default
        if 'csv' in content_type or url.endswith('.csv'):
            ext = '.csv'
        elif 'excel' in content_type or url.endswith('.xls'):
            ext = '.xls'
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        return temp_file_path, content_type
    
    @staticmethod
    def read_excel_file(file_path: str, content_type: str = None) -> pd.DataFrame:
        """
        Read Excel file and return DataFrame.
        
        Args:
            file_path: Path to the Excel file
            content_type: Content type of the file
            
        Returns:
            DataFrame containing the Excel data
        """
        if content_type and 'csv' in content_type or file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            return pd.read_excel(file_path)
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by handling special values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Replace infinity values with None
        df = df.replace([np.inf, -np.inf], None)
        
        # Replace NaN values with None
        df = df.where(pd.notnull(df), None)
        
        return df
    
    @staticmethod
    def detect_header_row(df: pd.DataFrame, max_rows_to_check: int = 10) -> int:
        """
        Dynamically detect the header row in the DataFrame.
        
        Args:
            df: Input DataFrame
            max_rows_to_check: Maximum number of rows to check for header
            
        Returns:
            Index of the header row
        """
        potential_header_rows = []
        
        # Check the first few rows
        for i in range(min(max_rows_to_check, len(df))):
            row = df.iloc[i]
            non_empty_cells = sum(1 for x in row if pd.notnull(x) and str(x).strip())
            total_cells = len(row)
            non_empty_ratio = non_empty_cells / total_cells if total_cells > 0 else 0
            
            # If this row has a high ratio of non-empty cells and they're all strings,
            # it might be a header row
            if non_empty_ratio > 0.5 and all(isinstance(x, str) for x in row if pd.notnull(x)):
                potential_header_rows.append(i)
        
        # If we found potential header rows, use the one with the most non-empty cells
        if potential_header_rows:
            return max(potential_header_rows, 
                      key=lambda i: sum(1 for x in df.iloc[i] if pd.notnull(x) and str(x).strip()))
        
        return 0  # Default to first row if no clear header found
    
    @staticmethod
    def handle_merged_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle merged columns in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled merged columns
        """
        # Check for long column names that might be merged
        for col in df.columns:
            col_str = str(col).strip()
            if len(col_str) > 50:  # Arbitrary threshold for "long" column names
                # Split by common delimiters
                parts = re.split(r'[,;\n]', col_str)
                if len(parts) > 1:
                    # Create new columns for each part
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part:
                            new_col = f"{part}_{i+1}" if i > 0 else part
                            df[new_col] = df[col]
                    
                    # Drop the original merged column
                    df = df.drop(columns=[col])
        
        return df
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up column names in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        # Strip whitespace and convert to string
        df.columns = [str(col).strip() for col in df.columns]
        
        # Handle duplicate column names
        if len(df.columns) != len(set(df.columns)):
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
    
    @staticmethod
    def process_excel(file_path: str, content_type: str = None, prompt: str = None) -> Dict[str, Any]:
        """
        Process Excel file with dynamic table structure detection.
        
        Args:
            file_path: Path to the Excel file
            content_type: Content type of the file
            prompt: Optional prompt for analysis
            
        Returns:
            Dictionary containing processed data and analysis
        """
        try:
            # Read the Excel file
            df = ExcelProcessor.read_excel_file(file_path, content_type)
            
            # Clean the data
            df = ExcelProcessor.clean_data(df)
            
            # Detect header row
            header_row = ExcelProcessor.detect_header_row(df)
            
            # Use the detected header row
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
            
            # Handle merged columns
            df = ExcelProcessor.handle_merged_columns(df)
            
            # Clean column names
            df = ExcelProcessor.clean_column_names(df)
            
            # Convert DataFrame to dict
            data_dict = df.to_dict(orient='records')
            
            return {
                "data": data_dict,
                "columns": list(df.columns)
            }
            
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
        finally:
            # Clean up the temporary file if it exists
            if os.path.exists(file_path):
                os.unlink(file_path) 