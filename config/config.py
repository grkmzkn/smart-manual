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

# Google Gemini settings (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_LLM = bool(GEMINI_API_KEY)  # Enable LLM if API key is available
GEMINI_MODEL = "gemini-pro"  # Gemini model to use

# Streamlit UI settings
PAGE_TITLE = "Smart Manual - Akıllı Kullanım Talimatı Asistanı"
PAGE_ICON = "📚"
