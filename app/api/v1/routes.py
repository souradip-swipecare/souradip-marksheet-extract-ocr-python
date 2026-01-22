
import time
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Query, Request
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import (
    ExtractionResponse,
    BatchExtractionResponse,
    HealthResponse,
    ErrorResponse,
    MarksheetExtraction
)
from app.services.extraction import llm_service
from app.services.ocr_service import ocr_service  # Tesseract OCR - Open Source

from app.utils.file_processor import file_processor
from app.utils.exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    FileProcessingError,
    LLMExtractionError
)
from app.core.config import settings

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and LLM service health status"
)
async def health_check():
    try:
        llm_healthy, provider = await llm_service.health_check()
        return HealthResponse(
            status="healthy" if llm_healthy else "degraded",
            version=settings.app_version,
            llm_provider=provider,
            llm_status="connected" if llm_healthy else "disconnected"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            llm_provider=settings.default_llm_provider,
            llm_status=f"error: {str(e)}"
        )


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Extraction failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Extract Marksheet Data",
   )
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def extract_marksheet(
    request: Request,
    file: UploadFile = File(..., description="Marksheet file (JPG, PNG, or PDF)"),
    apikey: Optional[str] = Query(default=None, description="Your Gemini API key. If not provided, server default key will be used."),
    model: str = Query(default="gemini", description="llm model to use. Currently only 'gemini' is supported.")
):

    start_time = time.time()
    extraction_method = "unknown"
    
    try:
        file_content = await file.read()
        file_size = len(file_content)
        
        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")
        
        # check file type
        file_processor.validate_file(
            filename=file.filename,
            file_size=file_size,
            content_type=file.content_type
        )
        
        processed = await file_processor.process_file_smart(file_content, file.filename)
        file_type = processed["type"]
        
        ocr_results = []
        total_ocr_confidence = 0.0
        avg_ocr_confidence = 0.0
        images = []
        
        if file_type == "text_pdf":
            extraction_method = "text_pdf_llm"
            text_content = processed["text_content"]
            
            # Build OCR-like results from PDF text
            for idx, text in enumerate(text_content):
                ocr_results.append({
                    "page": idx + 1,
                    "text": text,
                    "avg_confidence": 0.99,
                })
                total_ocr_confidence += 0.99
            
            avg_ocr_confidence = total_ocr_confidence / len(ocr_results) if ocr_results else 0.0
            logger.info(f"ocr sCore: {avg_ocr_confidence}")
            extraction = await llm_service.extract_marksheet_ocr(ocr_results, user_api_key=apikey)
            
        else:
            images = processed["images"]
            for idx, (image_bytes, mime_type) in enumerate(images):
                ocr_data = ocr_service.extract_text(image_bytes, save_text=True)
                ocr_results.append({
                    "page": idx + 1,
                    "text": ocr_data["raw_text"],
                    "avg_confidence": ocr_data["avg_confidence"],
                })
                total_ocr_confidence += ocr_data["avg_confidence"]
            
            avg_ocr_confidence = total_ocr_confidence / len(ocr_results) if ocr_results else 0.0
            logger.info(f"ocr sCore: {avg_ocr_confidence}")

            if avg_ocr_confidence >= settings.ocr_confidence_threshold:
                extraction_method = "ocr_llm"
                extraction = await llm_service.extract_marksheet_ocr(ocr_results, user_api_key=apikey)
            else:
                extraction_method = "direct_llm_vision"
                extraction = await llm_service.extract_marksheet(images, user_api_key=apikey)
        
        processing_time = (time.time() - start_time) * 1000
        
        response_data = {
            "success": True,
            "data": extraction,
            "processing_time_ms": round(processing_time, 2),
            "file_name": file.filename,
            "extraction_method": extraction_method,
            "ocr_confidence": round(avg_ocr_confidence, 3),
        }
        
        return response_data
        
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e)
        )
    except InvalidFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except LLMExtractionError as e:
        logger.error(f"LLM extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract data: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error during extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        await file.close()

@router.post(
    "/extract/batch",
    response_model=BatchExtractionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Batch Extract Marksheets",
    description="Upload multiple marksheets and extract data from all of them"
)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def batch_extract_marksheets(
    request: Request,
    files: List[UploadFile] = File(..., description="List of marksheet files"),
    apikey: Optional[str] = Query(default=None, description="Your Gemini API key (optional)"),
    model: str = Query(default="gemini", description="LLM model to use")
):
   
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files allowed per batch request"
        )
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        file_start = time.time()
        
        try:
            file_content = await file.read()
            file_size = len(file_content)
            
            file_processor.validate_file(
                filename=file.filename,
                file_size=file_size,
                content_type=file.content_type
            )
            
            # process
            images = await file_processor.process_file(file_content, file.filename)
            
            # Extract
            extraction = await llm_service.extract_marksheet(images, user_api_key=apikey)
            
            processing_time = (time.time() - file_start) * 1000
            
            results.append(ExtractionResponse(
                success=True,
                data=extraction,
                processing_time_ms=round(processing_time, 2),
                file_name=file.filename,
                file_size_bytes=file_size
            ))
            successful += 1
            
        except Exception as e:
            logger.error(f"Batch extraction failed for {file.filename}: {e}")
            processing_time = (time.time() - file_start) * 1000
            
            results.append(ExtractionResponse(
                success=False,
                data=None,
                error=str(e),
                processing_time_ms=round(processing_time, 2),
                file_name=file.filename,
                file_size_bytes=len(await file.read()) if file else 0
            ))
            failed += 1
            
        finally:
            await file.close()
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchExtractionResponse(
        success=failed == 0,
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results,
        total_processing_time_ms=round(total_time, 2)
    )


@router.get(
    "/schema",
    summary="Get Response Schema",
    description="Get the JSON schema for the extraction response"
)
async def get_schema():
    """Return the JSON schema for marksheet extraction response"""
    return MarksheetExtraction.model_json_schema()
