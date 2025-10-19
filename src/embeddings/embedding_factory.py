"""
Embedding Factory for creating embedding model instances.
Uses free and open-source sentence-transformers models.
"""

from typing import Optional
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
from src.utils.logger import app_logger


class EmbeddingFactory:
    """Factory class for creating embedding model instances."""
    
    _instance: Optional[Embeddings] = None
    
    @classmethod
    def create_embeddings(cls, force_new: bool = False) -> Embeddings:
        """
        Create or return cached embedding model instance.
        
        Args:
            force_new: Force creation of new instance
        
        Returns:
            Configured embeddings instance
        """
        if cls._instance is not None and not force_new:
            app_logger.debug("Returning cached embedding model")
            return cls._instance
        
        app_logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL}")
        
        try:
            
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': settings.EMBEDDING_DEVICE},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                }
            )
            
            # Test the embeddings
            test_text = "This is a test sentence."
            test_embedding = embeddings.embed_query(test_text)
            
            app_logger.info(
                f"Embedding model initialized successfully. "
                f"Dimension: {len(test_embedding)}, Device: {settings.EMBEDDING_DEVICE}"
            )
            
            cls._instance = embeddings
            return embeddings
        
        except Exception as e:
            app_logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    @classmethod
    def get_embedding_dimension(cls) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        embeddings = cls.create_embeddings()
        test_embedding = embeddings.embed_query("test")
        return len(test_embedding)


def get_embeddings() -> Embeddings:
    """
    Convenience function to get configured embeddings instance.
    
    Returns:
        Configured embeddings instance
    """
    return EmbeddingFactory.create_embeddings()
