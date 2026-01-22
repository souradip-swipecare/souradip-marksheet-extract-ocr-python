from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sys
import os

from app.core.config import settings
from app.utils.exceptions import MarksheetExtractionError
from app.services.extraction import llm_service
from app.api.routes import api_router

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)

# creating logs directory
os.makedirs("logs", exist_ok=True)
logger.add(
    settings.log_file,
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level
)

#  safe initialize and shut down
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    try:
        await llm_service.initialize()
        logger.info("LLM service initialized successfully")
    except Exception as e:
        logger.warning(f"LLM service initialization deferred: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")



app = FastAPI(
    title=settings.app_name,
    description="""
## souradip
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow from any weher in prod we have to add domain here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# global exception hndlr
@app.exception_handler(MarksheetExtractionError)
async def marksheet_extraction_error_handler(
    request: Request, 
    exc: MarksheetExtractionError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR",
            "details": {"message": str(exc)} if settings.debug else {}
        }
    )


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount testdata directory for demo files
testdata_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testdata")
if os.path.exists(testdata_dir):
    app.mount("/testdata", StaticFiles(directory=testdata_dir), name="testdata")

# api routes 
app.include_router(api_router, prefix="/api", tags=["Extraction"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered marksheet data extraction API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "extract": "/api/v1/extract"
    }


# demo page route
@app.get("/demo", tags=["Demo"])
async def demo_page():
    """Redirect to demo page"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/demo.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
