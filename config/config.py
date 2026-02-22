"""
Configuration settings for Smart Manual application.
All project-wide settings, paths, and constants are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploaded"
VECTORDB_DIR = DATA_DIR / "vectordb"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model settings
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Supports Turkish
EMBEDDING_DIMENSION = 384

# PDF processing settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Vector search settings
TOP_K_RESULTS = 3  # Number of relevant chunks to retrieve

# Google Gemini settings (required for Q&A)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"  # Latest Gemini 2.0 model - fast and efficient
GEMINI_VISION_MODEL = "gemini-2.5-flash"  # Gemini 2.0 Vision for images

# Image processing settings
# Options: "gemini" (Gemini Vision API - more accurate but uses API credits)
#          "ocr" (Tesseract OCR - free but requires installation)
#          "none" (skip images - fastest, text-only)
IMAGE_PROCESSING_MODE = "ocr"  # Change this to "gemini" or "none" as needed
PROCESS_IMAGES = IMAGE_PROCESSING_MODE != "none"  # Whether to process images at all
EXTRACT_TABLES = True  # Extract tables from PDFs using pdfplumber

# Tesseract OCR settings
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Tesseract executable path
TESSERACT_LANG = "tur+eng"  # OCR languages (Turkish + English)

# Streamlit UI settings
PAGE_TITLE = "Smart Manual - Akıllı Kullanım Talimatı Asistanı"
PAGE_ICON = "📚"
