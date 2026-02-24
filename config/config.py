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
MAX_CONTEXT_LENGTH = 8000  # Maximum characters to send to LLM (~2000 tokens)

# LLM Configuration
# Options: "gemini" (Google Gemini API - cloud-based, requires API key)
#          "local" (Local LLM via Ollama - privacy-focused, no API costs)
LLM_TYPE = "local"  # Change to "local" for offline LLM

# Google Gemini settings (used when LLM_TYPE="gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Only secret info in .env
GEMINI_MODEL = "gemini-2.5-flash"  # Latest Gemini 2.0 model - fast and efficient
GEMINI_VISION_MODEL = "gemini-2.5-flash"  # Gemini 2.0 Vision for images

# Local LLM settings (used when LLM_TYPE="local")
# Requires Ollama installation: https://ollama.ai/
# 
# PERFORMANS ÖNERİLERİ (CPU'da hız için):
#   - "llama3.2:1b"   → En hızlı (1B param, ~1GB, düşük kalite)
#   - "phi3:mini"     → Hızlı + Kaliteli (3.8B param, ~2.3GB) ✅ ÖNERİLEN
#   - "llama3.2:3b"   → Dengeli (3B param, ~2GB)
#   - "qwen2.5:3b"    → İyi kalite (3B param, ~2GB, biraz yavaş)
#   - "mistral:7b"    → Yüksek kalite (7B param, ~4GB, CPU'da çok yavaş)
#
# GPU varsa: Otomatik hızlanma sağlanır (nvidia-smi ile kontrol edin)
# Kurulum: ollama pull phi3:mini
LOCAL_LLM_BASE_URL = "http://localhost:11434"  # Ollama default URL
LOCAL_LLM_MODEL = "qwen2.5:3b"  # Değiştirmek için: ollama pull <model-name>

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
TESSERACT_CONFIG = "--psm 11"  # Page segmentation mode: 11 = sparse text (best for diagrams)

# Streamlit UI settings
PAGE_TITLE = "Smart Manual - Akıllı Kullanım Talimatı Asistanı"
PAGE_ICON = "📚"
