"""
Vector database module for Smart Manual application.
FAISS-based vector storage and similarity search operations.
"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss

# Import configuration
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import EMBEDDING_DIMENSION


class VectorStore:
    """FAISS-based vector database for document storage and retrieval."""
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        """
        Initialize vector store.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, any]]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            embeddings: Numpy array of document embeddings
            documents: List of document dictionaries with 'content' and 'metadata'
        """
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        for doc in documents:
            self.documents.append(doc['content'])
            self.metadata.append(doc['metadata'])
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Search for most similar documents to query.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of tuples (document_content, distance_score, metadata)
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure proper shape and type
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(distances[0][i]),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, directory_path: str) -> None:
        """
        Save vector database to disk.
        
        Args:
            directory_path: Directory to save database files
        """
        os.makedirs(directory_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory_path, "index.faiss"))
        
        # Save documents and metadata
        with open(os.path.join(directory_path, "documents.pkl"), "wb") as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
    
    def load(self, directory_path: str) -> None:
        """
        Load vector database from disk.
        
        Args:
            directory_path: Directory containing database files
        """
        # Load FAISS index
        index_path = os.path.join(directory_path, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        doc_path = os.path.join(directory_path, "documents.pkl")
        if os.path.exists(doc_path):
            with open(doc_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
    
    def clear(self) -> None:
        """Clear all data from vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
    
    def get_document_count(self) -> int:
        """
        Get total number of documents in store.
        
        Returns:
            Number of documents
        """
        return self.index.ntotal
