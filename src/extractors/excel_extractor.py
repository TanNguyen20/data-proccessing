from io import BytesIO
from typing import Union, Dict, List

import pandas as pd
import requests

from ..config import SUPPORTED_EXCEL_FORMATS


class ExcelExtractor:
    """Extractor for Excel files (local and online)"""

    @staticmethod
    def is_valid_excel_file(file_path: str) -> bool:
        """Check if the file is a supported Excel format"""
        return any(file_path.lower().endswith(ext) for ext in SUPPORTED_EXCEL_FORMATS)

    @staticmethod
    def read_local_excel(file_path: str, **kwargs) -> pd.DataFrame:
        """Read local Excel file"""
        if not ExcelExtractor.is_valid_excel_file(file_path):
            raise ValueError(f"Unsupported file format. Supported formats: {SUPPORTED_EXCEL_FORMATS}")

        return pd.read_excel(file_path, **kwargs)

    @staticmethod
    def read_online_excel(url: str, **kwargs) -> pd.DataFrame:
        """Read Excel file from URL"""
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download file from {url}")

        # Check if URL points to an Excel file
        if not any(url.lower().endswith(ext) for ext in SUPPORTED_EXCEL_FORMATS):
            raise ValueError(f"URL must point to a supported Excel format: {SUPPORTED_EXCEL_FORMATS}")

        return pd.read_excel(BytesIO(response.content), **kwargs)

    @staticmethod
    def extract_tables(df: pd.DataFrame) -> List[Dict]:
        """Extract tables from DataFrame"""
        tables = []

        # Get all unique column combinations that might form tables
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns) + 1):
                table_data = df.iloc[:, i:j].to_dict('records')
                if table_data:
                    tables.append({
                        'columns': df.columns[i:j].tolist(),
                        'data': table_data
                    })

        return tables

    @staticmethod
    def process_excel(file_path_or_url: str, **kwargs) -> Union[pd.DataFrame, List[Dict]]:
        """Process Excel file from local path or URL"""
        try:
            # Try to read as local file first
            df = ExcelExtractor.read_local_excel(file_path_or_url, **kwargs)
        except FileNotFoundError:
            # If not found locally, try as URL
            df = ExcelExtractor.read_online_excel(file_path_or_url, **kwargs)

        # Extract tables if requested
        if kwargs.get('extract_tables', False):
            return ExcelExtractor.extract_tables(df)

        return df
