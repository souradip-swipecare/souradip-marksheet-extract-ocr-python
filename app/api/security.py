
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from app.core.config import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query)
) -> Optional[str]:

    if not settings.api_key_enabled:
        return None
    
    provided_key = api_key_header or api_key_query
    
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Provide it via X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if provided_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return provided_key


async def optional_api_key(
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query)
) -> Optional[str]:
    """Optional API key - doesn't raise error if missing"""
    return api_key_header or api_key_query
