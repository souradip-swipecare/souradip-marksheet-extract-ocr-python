

import io
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from PIL import Image  
import fitz 

from app.core.config import settings
from app.utils.exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    FileProcessingError
)

logger = logging.getLogger(__name__)


class FileProcessor:

    
    ALLOWED_IMAGE_TYPES = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_PDF_TYPES = {"pdf"}
    
    # Minimum characters to consider PDF as text-based
    MIN_TEXT_CHARS_PER_PAGE = 80
    
    def __init__(self):
        self.max_file_size = settings.max_file_size_bytes
        self.allowed_extensions = set(settings.allowed_extensions_list)
    
    def validate_file(self, filename: str, file_size: int, content_type: str) -> None:
 
        # Check file size (error handling for large files)
        if file_size > self.max_file_size:
            raise FileTooLargeError(
                f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({settings.max_file_size_mb} MB)"
            )
        
        # Check file extension (error handling for wrong format)
        extension = self._get_extension(filename)
        if extension not in self.allowed_extensions:
            raise InvalidFileTypeError(
                f"File type '.{extension}' not allowed. "
                f"Allowed types: {', '.join(self.allowed_extensions)}"
            )
    
    def _get_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower().lstrip(".")
    
    def is_pdf(self, filename: str) -> bool:
        return self._get_extension(filename) in self.ALLOWED_PDF_TYPES
    # checking file have images or not
    def is_image(self, filename: str) -> bool:
        return self._get_extension(filename) in self.ALLOWED_IMAGE_TYPES
    
    def check_pdf_type(self, file_content: bytes) -> Dict[str, Any]:

        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            page_count = len(pdf_document)
            
            text_content = []
            total_chars = 0
            
            for page_num in range(page_count):
                page = pdf_document[page_num]
                # Extract text from page
                text = page.get_text("text").strip()
                text_content.append(text)
                total_chars += len(text)
            
            pdf_document.close()
            
            # Consider text-based if average chars per page >= threshold
            avg_chars_per_page = total_chars / page_count if page_count > 0 else 0
            is_text_based = avg_chars_per_page >= self.MIN_TEXT_CHARS_PER_PAGE
            
            logger.info(
                f"PDF analysis: {page_count} pages, {total_chars} total chars, "
                f"avg {avg_chars_per_page:.0f} chars/page, "
                f"type: {'text-based' if is_text_based else 'image-based/scanned'}"
            )
            
            return {
                "is_text_based": is_text_based,
                "text_content": text_content,
                "total_chars": total_chars,
                "page_count": page_count,
                "avg_chars_per_page": avg_chars_per_page
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze PDF type: {e}")
            return {
                "is_text_based": False,
                "text_content": [],
                "total_chars": 0,
                "page_count": 0,
                "avg_chars_per_page": 0
            }
    
    async def process_file(
        self, 
        file_content: bytes, 
        filename: str
    ) -> List[Tuple[bytes, str]]:

        if self.is_pdf(filename):
            return await self._process_pdf(file_content)
        else:
            return await self._process_image(file_content, filename)
    
    async def process_file_smart(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Dict[str, Any]:
       
        if self.is_pdf(filename):
            pdf_info = self.check_pdf_type(file_content)
            
            if pdf_info["is_text_based"]:
                logger.info("PDF is text-based, extracting text directly (no OCR needed)")
                return {
                    "type": "text_pdf",
                    "images": None,
                    "text_content": pdf_info["text_content"],
                    "pdf_info": pdf_info
                }
            else:
                # Image-based/scanned PDF - convert to images for OCR
                logger.info("PDF is image-based/scanned, converting to images for OCR")
                images = await self._process_pdf(file_content)
                return {
                    "type": "image_pdf",
                    "images": images,
                    "text_content": None,
                    "pdf_info": pdf_info
                }
        else:
         
            images = await self._process_image(file_content, filename)
            return {
                "type": "image",
                "images": images,
                "text_content": None,
                "pdf_info": None
            }
    
    async def _process_image(
        self, 
        file_content: bytes, 
        filename: str
    ) -> List[Tuple[bytes, str]]:

        try:
           
            image = Image.open(io.BytesIO(file_content))
            
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            
            max_dimension = 4096
            if max(image.size) > max_dimension:
                image = self._resize_image(image, max_dimension)
            
            # Convert to bytes
            buffer = io.BytesIO()
            image_format = "JPEG" if self._get_extension(filename) in ("jpg", "jpeg") else "PNG"
            image.save(buffer, format=image_format, quality=95)
            
            mime_type = f"image/{image_format.lower()}"
            return [(buffer.getvalue(), mime_type)]
            
        except Exception as e:
            raise FileProcessingError(f"Failed to process image: {str(e)}")
    
    async def _process_pdf(
        self, 
        file_content: bytes
    ) -> List[Tuple[bytes, str]]:
 
        try:
            images = []
            
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Render page to image (higher resolution for better OCR)
                # 300 DPI is good for text extraction
                zoom = 300 / 72  # 72 is default DPI
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix)
                
                # Convert to PIL Image
                image_data = pixmap.tobytes("png")
                
                images.append((image_data, "image/png"))
            
            pdf_document.close()
            
            if not images:
                raise FileProcessingError("PDF contains no pages")
            
            return images
            
        except fitz.FileDataError as e:
            raise FileProcessingError(f"Invalid or corrupted PDF file: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Failed to process PDF: {str(e)}")
    
    def _resize_image(self, image: Image.Image, max_dimension: int) -> Image.Image:
        width, height = image.size
        
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def get_image_dimensions(self, image_bytes: bytes) -> Tuple[int, int]:
        image = Image.open(io.BytesIO(image_bytes))
        return image.size
file_processor = FileProcessor()
