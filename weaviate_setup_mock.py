"""
Modified Weaviate setup that bypasses OpenAI vectorization.
Uses mock vectors for testing without OpenAI quota issues.
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from typing import Optional, Dict, Any
import logging
import numpy as np

from config import config
from mock_vectorizer import mock_vectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockWeaviateManager:
    """Weaviate manager that uses mock vectorization instead of OpenAI."""
    
    def __init__(self):
        """Initialize Weaviate client."""
        self.client: Optional[weaviate.WeaviateClient] = None
        self.collection_name = config.COLLECTION_NAME + "_mock"
        
    def connect(self) -> bool:
        """
        Connect to Weaviate cloud instance.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            weaviate_config = config.get_weaviate_config()
            
            # Close existing connection if any
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_config['url'],
                auth_credentials=weaviate.AuthApiKey(weaviate_config['auth_client_secret'])
            )
            
            # Test connection
            if self.client.is_ready():
                logger.info("Successfully connected to Weaviate (mock mode)")
                return True
            else:
                logger.error("Failed to connect to Weaviate - service not ready")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if the mock collection exists."""
        if not self.client:
            logger.error("Weaviate client not connected")
            return False
            
        try:
            collections = self.client.collections.list_all()
            collection_names = [col if isinstance(col, str) else col.name for col in collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    def create_collection(self) -> bool:
        """
        Create the research papers collection WITHOUT OpenAI vectorization.
        
        Returns:
            bool: True if collection created successfully, False otherwise
        """
        if not self.client:
            logger.error("Weaviate client not connected")
            return False
            
        try:
            # Create collection WITHOUT vectorizer
            self.client.collections.create(
                name=self.collection_name,
                description="Collection for research papers with mock vectors (no OpenAI)",
                properties=[
                    Property(name="title", data_type=DataType.TEXT, description="Title of the research paper"),
                    Property(name="authors", data_type=DataType.TEXT_ARRAY, description="List of authors"),
                    Property(name="abstract", data_type=DataType.TEXT, description="Abstract of the paper"),
                    Property(name="content_chunk", data_type=DataType.TEXT, description="Text chunk from the paper content"),
                    Property(name="publication_year", data_type=DataType.INT, description="Year of publication"),
                    Property(name="doi", data_type=DataType.TEXT, description="Digital Object Identifier"),
                    Property(name="chunk_index", data_type=DataType.INT, description="Index of this chunk within the paper"),
                    Property(name="paper_id", data_type=DataType.TEXT, description="Unique identifier for the paper")
                ]
            )
            logger.info(f"Successfully created mock collection: {self.collection_name}")
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
            collection = self.client.collections.get(self.collection_name)
            logger.info(f"Successfully retrieved mock collection: {self.collection_name}")
            return collection
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
            logger.info(f"Mock collection {self.collection_name} already exists")
            return True
        
        # Create collection if it doesn't exist
        logger.info(f"Creating mock collection {self.collection_name}")
        return self.create_collection()
    
    def insert_with_mock_vectors(self, data: Dict[str, Any]) -> bool:
        """
        Insert data with manually generated mock vectors.
        
        Args:
            data: Data to insert
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.get_collection()
            if not collection:
                return False
            
            # Generate mock vector for the content
            content = data.get("content_chunk", "")
            mock_vector = mock_vectorizer.vectorize(content)
            
            # Insert with custom vector - use the new API format
            collection.data.insert(
                properties=data,
                vector=mock_vector
            )
            
            logger.info("Successfully inserted data with mock vector")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting with mock vector: {str(e)}")
            return False
    
    def close(self):
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")

# Global instance
mock_weaviate_manager = MockWeaviateManager()

def setup_mock_weaviate() -> bool:
    """
    Convenience function to set up Weaviate with mock vectorization.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    return mock_weaviate_manager.setup_collection()

def get_mock_weaviate_collection():
    """
    Get the Weaviate collection for use in other modules.
    
    Returns:
        Collection object or None if not available
    """
    return mock_weaviate_manager.get_collection()

if __name__ == "__main__":
    # Test the mock setup
    print("Testing mock Weaviate setup...")
    if setup_mock_weaviate():
        print("✅ Mock Weaviate setup successful!")
        
        # Test collection access
        collection = get_mock_weaviate_collection()
        if collection:
            print("✅ Mock collection access successful!")
        else:
            print("❌ Mock collection access failed!")
        
        # Clean up
        mock_weaviate_manager.close()
    else:
        print("❌ Mock Weaviate setup failed!")
