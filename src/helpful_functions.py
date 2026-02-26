import os
from datetime import datetime
from typing import Optional


def format_file_size(size_bytes: int) -> str:
    """
    Convert file size in bytes to human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB", "500 KB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        Timestamp string in format "YYYY-MM-DD HH:MM:SS"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_directory(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if file is a valid PDF.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        True if file is a valid PDF, False otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    if not file_path.lower().endswith('.pdf'):
        return False
    
    # Check PDF header
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove null characters
    text = text.replace('\x00', '')
    return text.strip()
