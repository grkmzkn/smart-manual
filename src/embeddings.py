from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Import configuration
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class EmbeddingModel:
    """Handles text to vector embedding operations."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert single text into vector embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array representing text vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Convert multiple texts into vector embeddings.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of text vectors (shape: [num_texts, embedding_dim])
        """
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=show_progress
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get dimensionality of embeddings.
        
        Returns:
            Integer dimension of embedding vectors
        """
        return self.dimension
