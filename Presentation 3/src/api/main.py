"""
Main FastAPI Application Entry Point
TrueSight - AI-Powered Deepfake Detection System
"""

import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from api.routes import (
    health,
    detection,
    forensic,
    security,
    blockchain,
    streaming
)
from shared.config import settings
from shared.database import init_db
from shared.security import setup_security_middleware
from shared.monitoring_enhanced import setup_monitoring, setup_metrics_endpoint, setup_monitoring_middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting TrueSight API Server...")
    
    # Initialize database connections (with fallback)
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (continuing without DB): {e}")
        # Continue without database for demo purposes
    
    # Setup monitoring
    setup_monitoring()
    logger.info("Monitoring system initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TrueSight API Server...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Multi-Modal Deepfake Detection and Forensic Attribution System",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # In production, specify trusted hosts
)

# Setup security middleware
setup_security_middleware(app)

# Setup monitoring middleware
setup_monitoring_middleware(app)

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(detection.router, prefix="/api/v1/detection", tags=["Detection"])
app.include_router(forensic.router, prefix="/api/v1/forensic", tags=["Forensic"])
app.include_router(security.router, prefix="/api/v1/security", tags=["Security"])
app.include_router(blockchain.router, prefix="/api/v1/blockchain", tags=["Blockchain"])
app.include_router(streaming.router, prefix="/api/v1/stream", tags=["Streaming"])

# Setup monitoring endpoints
setup_metrics_endpoint(app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "description": "AI-Powered Multi-Modal Deepfake Detection System"
    }

if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )