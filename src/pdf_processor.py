"""
PDF processing module for Smart Manual application.
Handles PDF text extraction and text chunking operations.
"""

import os
from typing import List, Dict
import PyPDF2
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpful_functions import clean_text


class PDFProcessor:
    """Handles PDF file processing operations."""
    
    def __init__(self):
        """Initialize PDF processor."""
        self.documents = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            Exception: If PDF reading fails
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text
            
            # Clean extracted text
            text = clean_text(text)
            return text
            
        except Exception as e:
            raise Exception(f"PDF reading error: {str(e)}")
    
    def split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int = CHUNK_SIZE, 
        overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """
        Split text into smaller chunks with overlap.
        
        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Process PDF file: extract text and split into chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        chunks = self.split_text_into_chunks(text)
        
        # Create documents with metadata
        documents = []
        filename = os.path.basename(pdf_path)
        
        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'metadata': {
                    'source': filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })
        
        self.documents = documents
        return documents
