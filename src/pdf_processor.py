import os
from typing import List, Dict
import pdfplumber

# Import configuration
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EXTRACT_TABLES, PROCESS_IMAGES
from src.helpful_functions import clean_text
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
    
    def extract_text_by_pages(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text content from PDF file page by page.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with page text and page numbers
            
        Raises:
            Exception: If PDF reading fails
        """
        try:
            pages_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = ""
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        page_text += text + "\n\n"
                    
                    # Extract tables if enabled
                    if EXTRACT_TABLES:
                        tables = page.extract_tables()
                        for table in tables:
                            table_text = self._table_to_text(table)
                            page_text += f"\n[TABLE]\n{table_text}\n[/TABLE]\n\n"
                    
                    if page_text.strip():
                        pages_data.append({
                            'text': clean_text(page_text),
                            'page_num': page_num
                        })
            
            return pages_data
            
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
        
        # Extract text page by page
        pages_data = self.extract_text_by_pages(pdf_path)
        
        # Process each page and create chunks
        chunk_id = 0
        for page_data in pages_data:
            page_chunks = self.split_text_into_chunks(page_data['text'])
            
            for chunk in page_chunks:
                all_documents.append({
                    'content': chunk,
                    'type': 'text',
                    'metadata': {
                        'source': filename,
                        'page_number': page_data['page_num'],
                        'chunk_id': chunk_id,
                        'content_type': 'text'
                    }
                })
                chunk_id += 1
        
        # Process images if enabled
        if self.process_images and self.image_processor:
            print(f"\n🖼️  Processing images from {filename}...")
            image_documents = self.image_processor.process_pdf_images(pdf_path)
            
            # Create page content map for context enrichment
            page_content_map = {page['page_num']: page['text'] for page in pages_data}
            
            # Add image documents with context from their page
            for i, img_doc in enumerate(image_documents):
                page_num = img_doc['metadata'].get('page_num', 0)
                
                # Get page context (first 300 chars as header/title context)
                page_context = ""
                if page_num in page_content_map:
                    page_text = page_content_map[page_num]
                    # Extract first part as context (likely contains title/heading)
                    page_context = page_text[:300].strip()
                    if len(page_text) > 300:
                        page_context += "..."
                
                # Enrich image content with page context
                if page_context:
                    original_content = img_doc['content']
                    enriched_content = f"""[SAYFA BAĞLAMI - Sayfa {page_num}]
{page_context}

{original_content}"""
                    img_doc['content'] = enriched_content
                
                # Update metadata
                img_doc['metadata']['source'] = filename
                img_doc['metadata']['page_number'] = page_num
                img_doc['metadata']['chunk_id'] = chunk_id
                img_doc['metadata']['content_type'] = 'image'
                all_documents.append(img_doc)
                chunk_id += 1
        
        self.documents = all_documents
        return all_documents
