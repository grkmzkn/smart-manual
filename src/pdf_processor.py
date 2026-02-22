"""
PDF processing module for Smart Manual application.
Handles PDF text extraction, table extraction, and text chunking operations.
Uses pdfplumber for better text and table extraction.
"""

import os
from typing import List, Dict
import pdfplumber
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, EXTRACT_TABLES, PROCESS_IMAGES
from utils.helpful_functions import clean_text
from src.image_processor import ImageProcessor


class PDFProcessor:
    """Handles PDF file processing operations."""
    
    def __init__(self, process_images: bool = PROCESS_IMAGES):
        """
        Initialize PDF processor.
        
        Args:
            process_images: Whether to process images in PDFs
        """
        self.documents = []
        self.process_images = process_images
        
        if process_images:
            self.image_processor = ImageProcessor()
        else:
            self.image_processor = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            Exception: If PDF reading fails
        """
        try:
            text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    
                    # Extract tables if enabled
                    if EXTRACT_TABLES:
                        tables = page.extract_tables()
                        for table in tables:
                            # Convert table to text format
                            table_text = self._table_to_text(table)
                            text += f"\n[TABLE]\n{table_text}\n[/TABLE]\n\n"
            
            # Clean extracted text
            text = clean_text(text)
            return text
            
        except Exception as e:
            raise Exception(f"PDF reading error: {str(e)}")
    
    def _table_to_text(self, table: List[List]) -> str:
        """
        Convert table to readable text format.
        
        Args:
            table: 2D list representing table data
            
        Returns:
            Formatted table as string
        """
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            # Filter out None values and join cells
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            text_lines.append(" | ".join(cleaned_row))
        
        return "\n".join(text_lines)
    
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
        Process PDF file: extract text, tables, images and split into chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
        """
        filename = os.path.basename(pdf_path)
        all_documents = []
        
        # Extract and process text
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.split_text_into_chunks(text)
        
        # Create text documents with metadata
        for i, chunk in enumerate(chunks):
            all_documents.append({
                'content': chunk,
                'type': 'text',
                'metadata': {
                    'source': filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'content_type': 'text'
                }
            })
        
        # Process images if enabled
        if self.process_images and self.image_processor:
            print(f"\n🖼️  Processing images from {filename}...")
            image_documents = self.image_processor.process_pdf_images(pdf_path)
            
            # Add image documents with adjusted metadata
            for i, img_doc in enumerate(image_documents):
                img_doc['metadata']['source'] = filename
                img_doc['metadata']['chunk_id'] = len(chunks) + i
                img_doc['metadata']['total_chunks'] = len(chunks) + len(image_documents)
                img_doc['metadata']['content_type'] = 'image'
                all_documents.append(img_doc)
        
        self.documents = all_documents
        return all_documents
