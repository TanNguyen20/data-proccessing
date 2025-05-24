from pymongo import MongoClient
from typing import Dict, Any, Optional, List, Union
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote
from fastapi import UploadFile
from .providers.base_provider import BaseProvider
from .providers.openai_provider import OpenAIProvider
from .providers.gemini_provider import GeminiProvider
from .providers.xai_provider import XAIProvider

# Load environment variables
load_dotenv()

# MongoDB connection string from environment variable
MONGODB_URI = os.getenv('MONGODB_URI')

def get_mongodb_client() -> MongoClient:
    """
    Create and return a MongoDB client instance
    """
    if not MONGODB_URI:
        raise ValueError("MongoDB URI not found in environment variables")
    return MongoClient(MONGODB_URI)

def get_ai_provider(provider_name: str = None) -> BaseProvider:
    """
    Get the appropriate AI provider instance
    
    Args:
        provider_name (str, optional): Name of the provider to use. If None, uses default from env.
        
    Returns:
        BaseProvider: Instance of the selected AI provider
    """
    provider_name = provider_name or os.getenv('DEFAULT_PROVIDER', 'xai').lower()
    
    providers = {
        'openai': OpenAIProvider,
        'gemini': GeminiProvider,
        'xai': XAIProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}")
        
    return providers[provider_name]()

class DatabaseNameGenerator:
    """Class to handle database table/collection name generation"""
    
    @staticmethod
    def clean_name(name: str, max_length: int = 30) -> str:
        """
        Clean and validate a database name according to common conventions
        
        Args:
            name (str): Name to clean
            max_length (int): Maximum length of the name
            
        Returns:
            str: Cleaned name
        """
        # Convert to lowercase
        name = name.strip().lower()
        
        # Replace spaces and special characters with underscores
        name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        
        # Remove consecutive underscores
        name = '_'.join(filter(None, name.split('_')))
        
        # Ensure the name is not too long
        if len(name) > max_length:
            name = name[:max_length].rstrip('_')
            
        return name
    
    @staticmethod
    def extract_url_components(url: str) -> Dict[str, str]:
        """
        Extract meaningful components from a URL
        
        Args:
            url (str): URL to parse
            
        Returns:
            Dict[str, str]: Dictionary containing URL components
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            # Decode URL-encoded components
            path = unquote(parsed.path)
            query = unquote(parsed.query)
            
            # Split path into segments
            path_segments = [seg for seg in path.split('/') if seg]
            
            # Get the last meaningful segment (usually the most specific)
            last_segment = path_segments[-1] if path_segments else ''
            
            # Remove file extension if present
            last_segment = os.path.splitext(last_segment)[0]
            
            return {
                'domain': parsed.netloc,
                'path': path,
                'last_segment': last_segment,
                'query': query
            }
        except Exception as e:
            raise Exception(f"Error parsing URL: {str(e)}")
    
    @staticmethod
    async def generate_table_name_from_url(
        url: str,
        provider_name: Optional[str] = None,
        db_type: str = 'generic',
        max_length: int = 30,
        use_domain: bool = False
    ) -> str:
        """
        Generate a database table/collection name based on URL
        
        Args:
            url (str): URL to analyze
            provider_name (str, optional): Name of the AI provider to use
            db_type (str): Type of database
            max_length (int): Maximum length of the generated name
            use_domain (bool): Whether to include domain in the name
            
        Returns:
            str: Generated table/collection name
        """
        try:
            # Extract URL components
            url_components = DatabaseNameGenerator.extract_url_components(url)
            
            # Get AI provider
            provider = get_ai_provider(provider_name)
            
            # Create database-specific prompt
            db_specific_rules = {
                'mongodb': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Avoid using reserved MongoDB collection names
                """,
                'mysql': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Avoid using reserved MySQL keywords
                """,
                'postgresql': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Avoid using reserved PostgreSQL keywords
                """,
                'generic': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Be descriptive but concise
                """
            }
            
            db_rules = db_specific_rules.get(db_type.lower(), db_specific_rules['generic'])
            
            # Create prompt for name generation
            prompt = f"""Based on the following URL components, generate a suitable database {db_type} table/collection name.
            The name should be:
            {db_rules}
            - Descriptive of the URL content
            - Be concise (max {max_length} characters)
            - Reflect the main topic or data type from the URL
            
            URL Components:
            Domain: {url_components['domain']}
            Path: {url_components['path']}
            Last Segment: {url_components['last_segment']}
            Query: {url_components['query']}
            
            Return ONLY the name, nothing else."""
            
            # Get AI response
            response = await provider.generate_text(prompt)
            
            # Clean and validate the name
            name = DatabaseNameGenerator.clean_name(response, max_length)
            
            # Add domain prefix if requested
            if use_domain and url_components['domain']:
                domain_prefix = DatabaseNameGenerator.clean_name(url_components['domain'].split('.')[0])
                name = f"{domain_prefix}_{name}"
            
            return name
            
        except Exception as e:
            raise Exception(f"Error generating database name from URL: {str(e)}")
    
    @staticmethod
    async def generate_table_names_from_urls(
        urls: List[str],
        provider_name: Optional[str] = None,
        db_type: str = 'generic',
        max_length: int = 30,
        use_domain: bool = False
    ) -> List[str]:
        """
        Generate multiple database table/collection names based on URLs
        
        Args:
            urls (List[str]): List of URLs to analyze
            provider_name (str, optional): Name of the AI provider to use
            db_type (str): Type of database
            max_length (int): Maximum length of the generated names
            use_domain (bool): Whether to include domain in the names
            
        Returns:
            List[str]: List of generated table/collection names
        """
        names = []
        for url in urls:
            name = await DatabaseNameGenerator.generate_name_from_url(
                url,
                provider_name,
                db_type,
                max_length,
                use_domain
            )
            names.append(name)
        return names

    @staticmethod
    async def generate_table_name_from_file_content(
        file: UploadFile,
        provider_name: Optional[str] = None,
        db_type: str = 'generic',
        max_length: int = 30
    ) -> str:
        """
        Generate a database table/collection name based on file content using AI
        
        Args:
            file (UploadFile): FastAPI UploadFile object containing the file to analyze
            provider_name (str, optional): Name of the AI provider to use
            db_type (str): Type of database (e.g., 'mongodb', 'mysql', 'postgresql', 'generic')
            max_length (int): Maximum length of the generated name
            
        Returns:
            str: Generated table/collection name
            
        Raises:
            Exception: If there's an error reading the file or generating the name
        """
        try:
            # Read file content
            content = await file.read()
            file_content = content.decode('utf-8')
            
            # Get AI provider
            provider = get_ai_provider(provider_name)
            
            # Create database-specific prompt
            db_specific_rules = {
                'mongodb': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Avoid using reserved MongoDB collection names
                """,
                'mysql': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Avoid using reserved MySQL keywords
                """,
                'postgresql': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Avoid using reserved PostgreSQL keywords
                """,
                'generic': """
                - Use lowercase letters
                - Use underscores for spaces
                - No special characters except underscores
                - Be descriptive but concise
                """
            }
            
            db_rules = db_specific_rules.get(db_type.lower(), db_specific_rules['generic'])
            
            # Create prompt for name generation
            prompt = f"""Based on the following file content from {file.filename}, generate a suitable database {db_type} table/collection name.
            The name should be:
            {db_rules}
            - Descriptive of the data content
            - Be concise (max {max_length} characters)
            - Consider the file name and content type
            
            File content:
            {file_content[:1000]}  # Limit content to first 1000 chars to avoid token limits
            
            Return ONLY the name, nothing else."""
            
            # Get AI response
            response = await provider.generate_text(prompt)
            
            # Clean and validate the name
            return DatabaseNameGenerator.clean_name(response, max_length)
            
        except Exception as e:
            raise Exception(f"Error generating database name from file: {str(e)}")
        finally:
            # Reset file pointer
            await file.seek(0)
    
    @staticmethod
    async def generate_table_names_from_file_contents(
        files: List[UploadFile],
        provider_name: Optional[str] = None,
        db_type: str = 'generic',
        max_length: int = 30
    ) -> List[str]:
        """
        Generate multiple database table/collection names based on file contents
        
        Args:
            files (List[UploadFile]): List of FastAPI UploadFile objects to analyze
            provider_name (str, optional): Name of the AI provider to use
            db_type (str): Type of database
            max_length (int): Maximum length of the generated names
            
        Returns:
            List[str]: List of generated table/collection names
            
        Raises:
            Exception: If there's an error processing any of the files
        """
        names = []
        try:
            for file in files:
                name = await DatabaseNameGenerator.generate_table_name_from_file_content(
                    file,
                    provider_name,
                    db_type,
                    max_length
                )
                names.append(name)
            return names
        except Exception as e:
            raise Exception(f"Error generating names from files: {str(e)}")
        finally:
            # Reset file pointers
            for file in files:
                await file.seek(0)

def insert_json_data(collection_name: str, json_data: Dict[str, Any]) -> str:
    """
    Insert JSON data into specified MongoDB collection
    
    Args:
        collection_name (str): Name of the collection to insert data into
        json_data (Dict[str, Any]): JSON data to insert
        
    Returns:
        str: ID of the inserted document
    """
    try:
        # Get MongoDB client
        client = get_mongodb_client()
        
        # Get database and collection
        db = client.get_database()
        collection = db[collection_name]
        
        # Insert the document
        result = collection.insert_one(json_data)
        
        # Close the connection
        client.close()
        
        return str(result.inserted_id)
        
    except Exception as e:
        raise Exception(f"Error inserting data into MongoDB: {str(e)}")

def insert_many_json_data(collection_name: str, json_data_list: list[Dict[str, Any]]) -> list[str]:
    """
    Insert multiple JSON documents into specified MongoDB collection
    
    Args:
        collection_name (str): Name of the collection to insert data into
        json_data_list (list[Dict[str, Any]]): List of JSON documents to insert
    
    Returns:
        list[str]: List of IDs of the inserted documents
    """
    try:
        # Get MongoDB client
        client = get_mongodb_client()
        
        # Get database and collection
        db = client.get_database()
        collection = db[collection_name]
        
        # Insert the documents
        result = collection.insert_many(json_data_list)
        
        # Close the connection
        client.close()
        
        return [str(id) for id in result.inserted_ids]
        
    except Exception as e:
        raise Exception(f"Error inserting data into MongoDB: {str(e)}") 