"""
Image processing module for Smart Manual application.
Handles image extraction from PDFs and text extraction from images.
Supports both Gemini Vision API and OCR (Tesseract).
"""

import io
import base64
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Third-party imports
from PIL import Image as PILImage
import fitz  # PyMuPDF for image extraction
import google.generativeai as genai

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import (
    GEMINI_API_KEY, 
    GEMINI_VISION_MODEL,
    IMAGE_PROCESSING_MODE,
    TESSERACT_PATH,
    TESSERACT_LANG
)


class ImageProcessor:
    """Handles image extraction and text extraction from images."""
    
    def __init__(self, processing_mode: str = IMAGE_PROCESSING_MODE):
        """
        Initialize image processor.
        
        Args:
            processing_mode: "gemini", "ocr", or "none"
        """
        self.processing_mode = processing_mode.lower()
        
        # Initialize Gemini Vision if needed
        if self.processing_mode == "gemini" and GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.vision_model = genai.GenerativeModel(GEMINI_VISION_MODEL)
        else:
            self.vision_model = None
        
        # Check if OCR is available
        if self.processing_mode == "ocr":
            try:
                import pytesseract
                
                # Set Tesseract executable path from config
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
                
                self.ocr_available = True
            except ImportError:
                self.ocr_available = False
                print("⚠️ Warning: pytesseract not installed. OCR mode unavailable.")
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract all images from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with image data and metadata
        """
        images = []
        
        try:
            # Open PDF with PyMuPDF (fitz)
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        image = PILImage.open(io.BytesIO(image_bytes))
                        
                        images.append({
                            'image': image,
                            'page_num': page_num + 1,
                            'image_index': img_index,
                            'format': base_image["ext"],
                            'width': image.width,
                            'height': image.height
                        })
                    except Exception as e:
                        print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
        
        return images
    
    def process_image_with_gemini(self, image: PILImage.Image) -> str:
        """
        Process image using Gemini Vision API.
        
        Args:
            image: PIL Image object
            
        Returns:
            Description and text content from image
        """
        if not self.vision_model:
            return "Gemini Vision API not available."
        
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create prompt for Gemini
            prompt = """Bu görseli analiz et ve şunları yap:
1. Görselde ne olduğunu detaylıca açıkla
2. Görseldeki tüm yazıları ve metinleri oku
3. Tablolar varsa içeriğini çıkar
4. Diyagram veya şema varsa açıkla

Türkçe olarak yanıt ver."""
            
            # Generate response - pass PIL Image directly
            response = self.vision_model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            return f"Gemini Vision error: {str(e)}"
    
    def process_image_with_ocr(self, image: PILImage.Image) -> str:
        """
        Process image using Tesseract OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text from image
        """
        if not self.ocr_available:
            return "OCR not available. Please install pytesseract."
        
        try:
            import pytesseract
            
            # Ensure Tesseract path is set from config
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            
            # Perform OCR with configured languages
            text = pytesseract.image_to_string(image, lang=TESSERACT_LANG)
            
            if text.strip():
                return f"[Görsel içeriği - OCR ile çıkarıldı]\n{text.strip()}"
            else:
                return "[Görselde metin bulunamadı]"
                
        except Exception as e:
            return f"OCR error: {str(e)}"
    
    def process_image(self, image: PILImage.Image, metadata: Dict) -> Dict:
        """
        Process a single image based on configured mode.
        
        Args:
            image: PIL Image object
            metadata: Image metadata (page number, index, etc.)
            
        Returns:
            Dictionary with processed content and metadata
        """
        content = ""
        
        if self.processing_mode == "gemini":
            content = self.process_image_with_gemini(image)
        elif self.processing_mode == "ocr":
            content = self.process_image_with_ocr(image)
        else:
            content = "[Image processing disabled]"
        
        return {
            'content': content,
            'type': 'image',
            'metadata': metadata
        }
    
    def process_pdf_images(self, pdf_path: str) -> List[Dict]:
        """
        Extract and process all images from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed image documents
        """
        if self.processing_mode == "none":
            return []
        
        print(f"🖼️  Extracting images from PDF (mode: {self.processing_mode})...")
        
        # Extract images
        images = self.extract_images_from_pdf(pdf_path)
        
        if not images:
            print("No images found in PDF.")
            return []
        
        print(f"Found {len(images)} images. Processing...")
        
        # Process each image
        processed_documents = []
        for idx, img_data in enumerate(images):
            print(f"  Processing image {idx + 1}/{len(images)}...", end=" ")
            
            result = self.process_image(
                img_data['image'],
                {
                    'page_num': img_data['page_num'],
                    'image_index': img_data['image_index'],
                    'format': img_data['format'],
                    'dimensions': f"{img_data['width']}x{img_data['height']}"
                }
            )
            
            processed_documents.append(result)
            print("✓")
        
        return processed_documents
