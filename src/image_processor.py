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
    TESSERACT_LANG,
    TESSERACT_CONFIG
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
    
    def preprocess_image_for_ocr(self, image: PILImage.Image) -> PILImage.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        from PIL import ImageEnhance, ImageFilter, ImageOps
        import numpy as np
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too small (upscale more aggressively for better OCR)
            min_dimension = 1200  # Increased from 800
            if min(image.width, image.height) < min_dimension:
                scale = min_dimension / min(image.width, image.height)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, PILImage.Resampling.LANCZOS)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast more aggressively
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increased from 1.5
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)  # Increased from 1.3
            
            # Apply slight denoising
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Auto-contrast for better text visibility
            image = ImageOps.autocontrast(image)
            
            return image
            
        except Exception as e:
            print(f"Warning: Image preprocessing failed: {e}")
            return image
    
    def clean_ocr_output(self, text: str) -> str:
        """
        Clean OCR output by removing noise, fixing line breaks, and merging fragmented lines.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned and properly formatted text
        """
        import re
        
        # Split into lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are mostly special characters/symbols
            # Keep lines with at least 50% alphanumeric characters
            alnum_count = sum(c.isalnum() or c.isspace() for c in line)
            if len(line) > 0 and (alnum_count / len(line)) < 0.5:
                continue
            
            # Remove common diagram artifacts
            line = re.sub(r'[>|<\-_=\[\]{}()]+', ' ', line)
            
            # Remove extra spaces
            line = re.sub(r'\s+', ' ', line).strip()
            
            # Skip very short lines (likely noise)
            if len(line) < 3:
                continue
            
            cleaned_lines.append(line)
        
        # Merge fragmented lines (satır kaymalarını düzelt)
        merged_lines = []
        i = 0
        while i < len(cleaned_lines):
            current_line = cleaned_lines[i]
            
            # Liste öğesi mi kontrol et (numaralı veya madde işaretli)
            is_list_item = re.match(r'^\s*[\d\u2022\u25CF\u25E6\-\*]\s*[\.)]?\s+', current_line)
            
            # Başlık mı kontrol et (kısa ve büyük harfle başlayan)
            is_heading = (len(current_line) < 50 and 
                         current_line[0].isupper() and 
                         not current_line.endswith(('.', ':', ';', ',', ')', '!')))
            
            # Cümle sonu noktalama işaretleri
            ends_with_punctuation = current_line.endswith(('.', '!', '?', ':', ';', ')'))
            
            # Bir sonraki satırı birleştirmeyi dene
            while i + 1 < len(cleaned_lines):
                next_line = cleaned_lines[i + 1]
                
                # Sonraki satır liste öğesi mi?
                next_is_list_item = re.match(r'^\s*[\d\u2022\u25CF\u25E6\-\*]\s*[\.)]?\s+', next_line)
                
                # Mevcut satır noktalama ile bitiyorsa veya liste öğesi ise birleştirme
                if ends_with_punctuation or is_list_item or next_is_list_item or is_heading:
                    break
                
                # Küçük harfle başlıyorsa veya cümle devamıysa birleştir
                if (next_line and 
                    (next_line[0].islower() or 
                     current_line[-1] in (',', 've', 'veya', 'ile', 'için'))):
                    current_line += ' ' + next_line
                    i += 1
                    ends_with_punctuation = current_line.endswith(('.', '!', '?', ':', ';', ')'))
                else:
                    break
            
            merged_lines.append(current_line)
            i += 1
        
        return '\n\n'.join(merged_lines)
    
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
            
            # Preprocess image for better OCR
            processed_image = self.preprocess_image_for_ocr(image)
            
            # Perform OCR with configured languages and settings
            text = pytesseract.image_to_string(
                processed_image, 
                lang=TESSERACT_LANG,
                config=TESSERACT_CONFIG
            )
            
            # Clean OCR output
            cleaned_text = self.clean_ocr_output(text)
            
            if cleaned_text.strip():
                # Create a more detailed description
                result = f"""[GÖRSEL İÇERİĞİ - OCR ile çıkarıldı]

Görselde tespit edilen metin:
{cleaned_text.strip()}

[Bu bilgiler görsel üzerindeki yazılardan okunmuştur]"""
                return result
            else:
                return "[Görselde metin bulunamadı veya okunamadı]"
                
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
            page_num = img_data['page_num']
            print(f"  [{idx + 1}/{len(images)}] Page {page_num} - {img_data['width']}x{img_data['height']} - ", end="")
            
            result = self.process_image(
                img_data['image'],
                {
                    'page_num': img_data['page_num'],
                    'image_index': img_data['image_index'],
                    'format': img_data['format'],
                    'dimensions': f"{img_data['width']}x{img_data['height']}"
                }
            )
            
            # Show preview of extracted text
            content_preview = result['content'][:100].replace('\n', ' ')
            print(f"✓ ({len(result['content'])} chars: {content_preview}...)")
            
            processed_documents.append(result)
        
        return processed_documents
