"""
File Processing module
Extracts text content from uploaded files (PDF, TXT, CSV, DOCX).
"""
import io
from typing import Tuple

# Maximum file size in bytes (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024


def extract_text_from_file(file_content: bytes, filename: str) -> Tuple[bool, str]:
    """
    Extract text content from uploaded file.

    Args:
        file_content: Raw file bytes
        filename: Original filename (used to determine type)

    Returns:
        Tuple of (success: bool, text_or_error: str)
    """
    try:
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"

        file_type = filename.lower().split('.')[-1]

        if file_type == 'txt':
            return True, file_content.decode('utf-8')

        elif file_type == 'csv':
            try:
                import pandas as pd
                df = pd.read_csv(io.BytesIO(file_content))
                return True, df.to_string()
            except Exception as e:
                return False, f"Error reading CSV: {str(e)}"

        elif file_type == 'pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if not text.strip():
                    return False, "PDF appears to be empty or image-only (no extractable text)"
                return True, text.strip()
            except ImportError:
                return False, "PDF extraction requires PyPDF2 - file content not extracted"
            except Exception as e:
                return False, f"Error reading PDF: {str(e)}"

        elif file_type == 'docx':
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_content))
                text = "\n".join([para.text for para in doc.paragraphs])
                if not text.strip():
                    return False, "DOCX appears to be empty"
                return True, text.strip()
            except ImportError:
                return False, "DOCX extraction requires python-docx - file content not extracted"
            except Exception as e:
                return False, f"Error reading DOCX: {str(e)}"

        else:
            return False, f"Unsupported file type: {file_type}. Supported types: txt, csv, pdf, docx"

    except Exception as e:
        return False, f"Error extracting file content: {str(e)}"


def get_supported_extensions() -> list:
    """Return list of supported file extensions."""
    return ['txt', 'csv', 'pdf', 'docx']


def validate_file(filename: str, file_size: int) -> Tuple[bool, str]:
    """
    Validate file before processing.

    Args:
        filename: Original filename
        file_size: File size in bytes

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"

    if file_size == 0:
        return False, "File is empty"

    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    if extension not in get_supported_extensions():
        return False, f"Unsupported file type: {extension}. Supported types: {', '.join(get_supported_extensions())}"

    return True, ""
