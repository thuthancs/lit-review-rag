"""
Configuration module for Literature Review RAG system.
Handles environment variables, API keys, and system constants.
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Literature Review RAG system."""
    
    # API Keys and URLs
    WEAVIATE_URL: str = os.getenv('WEAVIATE_URL', '')
    WEAVIATE_API_KEY: str = os.getenv('WEAVIATE_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    
    # Collection Configuration
    COLLECTION_NAME: str = 'research_papers'
    
    # Text Processing Configuration
    CHUNK_SIZE: int = 250  # words per chunk
    CHUNK_OVERLAP: int = 50  # words overlap between chunks
    MAX_CHUNKS_PER_PAPER: int = 100
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5  # Number of relevant chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score
    
    # OpenAI Configuration
    OPENAI_MODEL: str = 'gpt-3.5-turbo'
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.3
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate that all required environment variables are set.
        
        Returns:
            bool: True if all required config is present, False otherwise
        """
        required_vars = [
            'WEAVIATE_URL',
            'WEAVIATE_API_KEY', 
            'OPENAI_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please create a .env file with the following variables:")
            print("WEAVIATE_URL=your_weaviate_url")
            print("WEAVIATE_API_KEY=your_weaviate_api_key")
            print("OPENAI_API_KEY=your_openai_api_key")
            return False
        
        return True
    
    @classmethod
    def get_weaviate_config(cls) -> dict:
        """
        Get Weaviate connection configuration.
        
        Returns:
            dict: Configuration for Weaviate client
        """
        return {
            'url': cls.WEAVIATE_URL,
            'auth_client_secret': cls.WEAVIATE_API_KEY,
            'timeout_config': (5, 15)  # (connection timeout, read timeout)
        }
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """
        Get OpenAI configuration.
        
        Returns:
            dict: Configuration for OpenAI client
        """
        return {
            'api_key': cls.OPENAI_API_KEY,
            'model': cls.OPENAI_MODEL,
            'max_tokens': cls.MAX_TOKENS,
            'temperature': cls.TEMPERATURE
        }

# Global config instance
config = Config()

# Validate configuration on import
if not config.validate_config():
    raise ValueError("Configuration validation failed. Please check your .env file.")
