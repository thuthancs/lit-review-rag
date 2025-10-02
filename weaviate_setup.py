"""
Weaviate setup and collection management for Literature Review RAG system.
Handles connection to Weaviate cloud instance and collection creation.
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from typing import Optional, Dict, Any
import logging

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateManager:
    """Manages Weaviate connection and collection operations."""
    
    def __init__(self):
        """Initialize Weaviate client."""
        self.client: Optional[weaviate.Client] = None
        self.collection_name = config.COLLECTION_NAME
        
    def connect(self) -> bool:
        """
        Connect to Weaviate cloud instance.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            weaviate_config = config.get_weaviate_config()
            
            self.client = weaviate.Client(
                url=weaviate_config['url'],
                auth_client_secret=weaviate_config['auth_client_secret'],
                timeout_config=weaviate_config['timeout_config']
            )
            
            # Test connection
            if self.client.is_ready():
                logger.info("Successfully connected to Weaviate")
                return True
            else:
                logger.error("Failed to connect to Weaviate - service not ready")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}")
            return False
    
    def collection_exists(self) -> bool:
        """
        Check if the research papers collection exists.
        
        Returns:
            bool: True if collection exists, False otherwise
        """
        if not self.client:
            logger.error("Weaviate client not connected")
            return False
            
        try:
            collections = self.client.collections.list_all()
            collection_names = [col.name for col in collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    def create_collection(self) -> bool:
        """
        Create the research papers collection with defined schema.
        
        Returns:
            bool: True if collection created successfully, False otherwise
        """
        if not self.client:
            logger.error("Weaviate client not connected")
            return False
            
        try:
            # Define collection schema
            collection_schema = {
                "class": self.collection_name,
                "description": "Collection for research papers with text chunks for RAG",
                "vectorizer": "text2vec-openai",
                "vectorizerConfig": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                },
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Title of the research paper"
                    },
                    {
                        "name": "authors",
                        "dataType": ["text[]"],
                        "description": "List of authors"
                    },
                    {
                        "name": "abstract",
                        "dataType": ["text"],
                        "description": "Abstract of the paper"
                    },
                    {
                        "name": "content_chunk",
                        "dataType": ["text"],
                        "description": "Text chunk from the paper content"
                    },
                    {
                        "name": "publication_year",
                        "dataType": ["int"],
                        "description": "Year of publication"
                    },
                    {
                        "name": "doi",
                        "dataType": ["text"],
                        "description": "Digital Object Identifier"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Index of this chunk within the paper"
                    },
                    {
                        "name": "paper_id",
                        "dataType": ["text"],
                        "description": "Unique identifier for the paper"
                    }
                ]
            }
            
            # Create collection
            self.client.schema.create_class(collection_schema)
            logger.info(f"Successfully created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False
    
    def get_collection(self):
        """
        Get the research papers collection.
        
        Returns:
            Collection object or None if not available
        """
        if not self.client:
            logger.error("Weaviate client not connected")
            return None
            
        try:
            return self.client.collections.get(self.collection_name)
        except Exception as e:
            logger.error(f"Error getting collection: {str(e)}")
            return None
    
    def setup_collection(self) -> bool:
        """
        Complete setup: connect, check if collection exists, create if needed.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        # Connect to Weaviate
        if not self.connect():
            return False
        
        # Check if collection exists
        if self.collection_exists():
            logger.info(f"Collection {self.collection_name} already exists")
            return True
        
        # Create collection if it doesn't exist
        logger.info(f"Creating collection {self.collection_name}")
        return self.create_collection()
    
    def close(self):
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")

# Global instance
weaviate_manager = WeaviateManager()

def setup_weaviate() -> bool:
    """
    Convenience function to set up Weaviate collection.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    return weaviate_manager.setup_collection()

def get_weaviate_collection():
    """
    Get the Weaviate collection for use in other modules.
    
    Returns:
        Collection object or None if not available
    """
    return weaviate_manager.get_collection()

if __name__ == "__main__":
    # Test the setup
    print("Testing Weaviate setup...")
    if setup_weaviate():
        print("✅ Weaviate setup successful!")
        
        # Test collection access
        collection = get_weaviate_collection()
        if collection:
            print("✅ Collection access successful!")
        else:
            print("❌ Collection access failed!")
    else:
        print("❌ Weaviate setup failed!")
    
    # Clean up
    weaviate_manager.close()
