"""
Health Check API Routes
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
from loguru import logger
import psutil
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "TrueSight API"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application specific checks could be added here
        # e.g., database connectivity, model loading status, etc.
        
        health_status = "healthy" if cpu_percent < 90 and memory.percent < 90 else "degraded"
        
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "uptime": psutil.boot_time()
            },
            "application": {
                "version": "1.0.0",
                "started_at": datetime.utcnow().isoformat()  # Would be stored in app state
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/ready")
async def readiness_probe():
    """Readiness probe for Kubernetes"""
    # Check if all required services are ready
    # This would check database connections, model loading, etc.
    return {"status": "ready"}

@router.get("/live")
async def liveness_probe():
    """Liveness probe for Kubernetes"""
    # Simple check to see if the application is responding
    return {"status": "alive"}