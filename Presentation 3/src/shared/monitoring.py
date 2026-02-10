"""
Monitoring and Observability
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import FastAPI
from fastapi.responses import Response
import time
from loguru import logger

# Check if metrics already exist to avoid duplication
def get_or_create_counter(name, description, labels):
    try:
        return Counter(name, description, labels)
    except ValueError:
        # Metric already exists, return existing one
        return REGISTRY._names_to_collectors[name]

def get_or_create_histogram(name, description, labels):
    try:
        return Histogram(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors[name]

def get_or_create_gauge(name, description):
    try:
        return Gauge(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors[name]

# Prometheus metrics
REQUEST_COUNT = get_or_create_counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = get_or_create_histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = get_or_create_gauge(
    'active_connections',
    'Number of active connections'
)

DETECTION_PROCESSING_TIME = get_or_create_histogram(
    'detection_processing_time_seconds',
    'Time spent processing detections',
    ['detection_type']
)

MODEL_INFERENCE_TIME = get_or_create_histogram(
    'model_inference_time_seconds',
    'Time spent in model inference',
    ['model_type']
)

def setup_monitoring():
    """Setup monitoring and metrics collection"""
    logger.info("Monitoring system initialized")
    # Additional monitoring setup can be added here

def record_request_metrics(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics"""
    try:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    except Exception as e:
        logger.warning(f"Failed to record metrics: {e}")

def record_detection_metrics(detection_type: str, processing_time: float):
    """Record detection processing metrics"""
    try:
        DETECTION_PROCESSING_TIME.labels(detection_type=detection_type).observe(processing_time)
    except Exception as e:
        logger.warning(f"Failed to record detection metrics: {e}")

def record_model_inference(model_type: str, inference_time: float):
    """Record model inference metrics"""
    try:
        MODEL_INFERENCE_TIME.labels(model_type=model_type).observe(inference_time)
    except Exception as e:
        logger.warning(f"Failed to record model metrics: {e}")

def setup_metrics_endpoint(app: FastAPI):
    """Setup Prometheus metrics endpoint"""
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type="text/plain")

# Export commonly used items
__all__ = [
    "setup_monitoring",
    "record_request_metrics", 
    "record_detection_metrics",
    "record_model_inference",
    "setup_metrics_endpoint"
]