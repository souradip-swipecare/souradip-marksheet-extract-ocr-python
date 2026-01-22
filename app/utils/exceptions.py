# coustom exception creation

from typing import Optional, Dict, Any

#  main class  extraction when we want to send error

class MarksheetExtractionError(Exception):
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "EXTRACTION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

# file size exceeds
class FileTooLargeError(MarksheetExtractionError):
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="FILE_TOO_LARGE"
        )

# when file type does not meet our defined tpes
class InvalidFileTypeError(MarksheetExtractionError):
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="INVALID_FILE_TYPE"
        )

 # we can not process the file

# errors ocuurs during file processing   
class FileProcessingError(MarksheetExtractionError):
 
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR"
        )

# llm model error when we can not extract data from output
    
class LLMExtractionError(MarksheetExtractionError):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LLM_EXTRACTION_ERROR",
            details=details
        )

# if llm does not connec we will call this exception
    
class LLMConnectionError(MarksheetExtractionError):
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            error_code="LLM_CONNECTION_ERROR"
        )

#   if api key invalid then this exception occurs
    
class InvalidAPIKeyError(MarksheetExtractionError):

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(
            message=message,
            error_code="INVALID_API_KEY"
        )

# rate limit exception handler
    
class RateLimitError(MarksheetExtractionError):

    def __init__(self, message: str = "Rate limit exceeded. Please try again later."):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED"
        )

# this one helps  for only validation error purpuses
    
class ValidationError(MarksheetExtractionError):

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )
