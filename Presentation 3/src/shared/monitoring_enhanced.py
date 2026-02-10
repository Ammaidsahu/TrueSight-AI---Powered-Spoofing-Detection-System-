"""
Enhanced Monitoring and Observability System
Real-time performance tracking and system health monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, Summary
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import time
import psutil
import asyncio
from typing import Dict, Any
from loguru import logger
import json
from collections import defaultdict, deque
import threading

class EnhancedMonitoringSystem:
    """Advanced monitoring system with real-time performance tracking"""
    
    def __init__(self):
        self.system_metrics_enabled = True
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'response_time': 2.0,  # seconds
            'error_rate': 0.05,    # 5%
            'throughput': 100      # requests per minute
        }
        
        # Enhanced Prometheus metrics
        self._setup_enhanced_metrics()
        self._start_background_monitoring()
        
        logger.info("Enhanced monitoring system initialized")
    
    def _setup_enhanced_metrics(self):
        """Setup comprehensive monitoring metrics"""
        # HTTP Request Metrics
        self.REQUEST_COUNT = Counter(
            'http_requests_enhanced_total',
            'Total HTTP requests (enhanced)',
            ['method', 'endpoint', 'status', 'user_agent']
        )
        
        self.REQUEST_DURATION = Histogram(
            'http_request_duration_enhanced_seconds',
            'HTTP request duration with detailed breakdown (enhanced)',
            ['method', 'endpoint', 'status'],
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.REQUEST_SIZE = Summary(
            'http_request_size_enhanced_bytes',
            'HTTP request size in bytes (enhanced)',
            ['method', 'endpoint']
        )
        
        self.RESPONSE_SIZE = Summary(
            'http_response_size_enhanced_bytes',
            'HTTP response size in bytes (enhanced)',
            ['method', 'endpoint', 'status']
        )
        
        # System Performance Metrics
        self.SYSTEM_CPU_USAGE = Gauge(
            'system_cpu_usage_enhanced_percent',
            'Current CPU usage percentage (enhanced)'
        )
        
        self.SYSTEM_MEMORY_USAGE = Gauge(
            'system_memory_usage_enhanced_percent',
            'Current memory usage percentage (enhanced)'
        )
        
        self.SYSTEM_DISK_USAGE = Gauge(
            'system_disk_usage_enhanced_percent',
            'Current disk usage percentage (enhanced)'
        )
        
        # Application Performance Metrics
        self.ACTIVE_CONNECTIONS = Gauge(
            'active_connections_enhanced',
            'Number of currently active connections (enhanced)'
        )
        
        self.DETECTION_PROCESSING_TIME = Histogram(
            'detection_processing_time_enhanced_seconds',
            'Time spent processing detections by type (enhanced)',
            ['detection_type', 'media_type'],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.MODEL_INFERENCE_TIME = Histogram(
            'model_inference_time_enhanced_seconds',
            'Time spent in model inference (enhanced)',
            ['model_type', 'operation'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.FILE_PROCESSING_SIZE = Histogram(
            'file_processing_size_enhanced_bytes',
            'Size of files being processed (enhanced)',
            ['file_type'],
            buckets=(1024, 10240, 102400, 1048576, 10485760, 104857600)  # 1KB to 100MB
        )
        
        # Error and Alert Metrics
        self.ERROR_COUNT = Counter(
            'errors_enhanced_total',
            'Total number of errors (enhanced)',
            ['error_type', 'endpoint', 'severity']
        )
        
        self.ALERT_FIRED = Counter(
            'alerts_fired_enhanced_total',
            'Total number of alerts fired (enhanced)',
            ['alert_type', 'severity']
        )
    
    def _start_background_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while True:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    # Update gauges
                    self.SYSTEM_CPU_USAGE.set(cpu_percent)
                    self.SYSTEM_MEMORY_USAGE.set(memory.percent)
                    self.SYSTEM_DISK_USAGE.set((disk.used / disk.total) * 100)
                    
                    # Check for alerts
                    self._check_system_alerts(cpu_percent, memory.percent)
                    
                    # Store in history
                    self.performance_history['cpu'].append(cpu_percent)
                    self.performance_history['memory'].append(memory.percent)
                    
                except Exception as e:
                    logger.warning(f"Background monitoring error: {e}")
                
                time.sleep(10)  # Check every 10 seconds
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def _check_system_alerts(self, cpu_percent: float, memory_percent: float):
        """Check system metrics against alert thresholds"""
        if cpu_percent > self.alert_thresholds['cpu_usage']:
            self.ALERT_FIRED.labels(alert_type='high_cpu', severity='warning').inc()
            logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
        
        if memory_percent > self.alert_thresholds['memory_usage']:
            self.ALERT_FIRED.labels(alert_type='high_memory', severity='warning').inc()
            logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
    
    def record_http_request(self, method: str, endpoint: str, status: int, 
                          duration: float, request_size: int = 0, 
                          response_size: int = 0, user_agent: str = "unknown"):
        """Record comprehensive HTTP request metrics"""
        # Basic metrics
        self.REQUEST_COUNT.labels(
            method=method, 
            endpoint=endpoint, 
            status=status,
            user_agent=user_agent[:50]  # Limit user agent length
        ).inc()
        
        self.REQUEST_DURATION.labels(
            method=method, 
            endpoint=endpoint, 
            status=status
        ).observe(duration)
        
        # Size metrics
        if request_size > 0:
            self.REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(request_size)
        
        if response_size > 0:
            self.RESPONSE_SIZE.labels(
                method=method, 
                endpoint=endpoint, 
                status=status
            ).observe(response_size)
        
        # Performance history
        self.performance_history[f'{method}_{endpoint}_duration'].append(duration)
        self.performance_history[f'{method}_{endpoint}_status_{status}'].append(1)
        
        # Check for slow requests
        if duration > self.alert_thresholds['response_time']:
            self.ALERT_FIRED.labels(alert_type='slow_request', severity='warning').inc()
    
    def record_detection_processing(self, detection_type: str, media_type: str, 
                                  processing_time: float, file_size: int = 0):
        """Record detection processing metrics"""
        self.DETECTION_PROCESSING_TIME.labels(
            detection_type=detection_type,
            media_type=media_type
        ).observe(processing_time)
        
        if file_size > 0:
            self.FILE_PROCESSING_SIZE.labels(file_type=media_type).observe(file_size)
        
        # Performance tracking
        key = f'detection_{detection_type}_{media_type}'
        self.performance_history[key].append(processing_time)
    
    def record_model_inference(self, model_type: str, operation: str, 
                              inference_time: float):
        """Record model inference metrics"""
        self.MODEL_INFERENCE_TIME.labels(
            model_type=model_type,
            operation=operation
        ).observe(inference_time)
    
    def record_error(self, error_type: str, endpoint: str = "unknown", 
                    severity: str = "error", message: str = ""):
        """Record error metrics"""
        self.ERROR_COUNT.labels(
            error_type=error_type,
            endpoint=endpoint,
            severity=severity
        ).inc()
        
        # Log detailed error
        logger.opt(exception=True).error(
            f"Error recorded: {error_type} at {endpoint} - {message}"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': time.time(),
            'http_metrics': self._get_http_performance(),
            'system_metrics': self._get_system_performance(),
            'detection_metrics': self._get_detection_performance(),
            'error_metrics': self._get_error_performance(),
            'predictions': self._get_performance_predictions()
        }
        return summary
    
    def _get_http_performance(self) -> Dict[str, Any]:
        """Get HTTP performance metrics"""
        # Calculate recent averages
        recent_durations = list(self.performance_history.get('GET_/api/v1/health_duration', []))
        if recent_durations:
            avg_response_time = sum(recent_durations) / len(recent_durations)
            p95_response_time = sorted(recent_durations)[int(0.95 * len(recent_durations))]
        else:
            avg_response_time = 0
            p95_response_time = 0
        
        return {
            'avg_response_time': round(avg_response_time, 3),
            'p95_response_time': round(p95_response_time, 3),
            'total_requests': sum(len(v) for k, v in self.performance_history.items() if 'duration' in k),
            'error_rate': self._calculate_error_rate()
        }
    
    def _get_system_performance(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        cpu_history = list(self.performance_history['cpu'])
        memory_history = list(self.performance_history['memory'])
        
        return {
            'cpu_usage': round(psutil.cpu_percent(), 1),
            'memory_usage': round(psutil.virtual_memory().percent, 1),
            'disk_usage': round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 1),
            'avg_cpu_last_minute': round(sum(cpu_history[-6:]) / len(cpu_history[-6:]), 1) if cpu_history else 0,
            'avg_memory_last_minute': round(sum(memory_history[-6:]) / len(memory_history[-6:]), 1) if memory_history else 0
        }
    
    def _get_detection_performance(self) -> Dict[str, Any]:
        """Get detection performance metrics"""
        # Get recent detection times
        video_times = list(self.performance_history.get('detection_video_video', []))
        audio_times = list(self.performance_history.get('detection_audio_audio', []))
        
        return {
            'video_avg_processing_time': round(sum(video_times)/len(video_times), 3) if video_times else 0,
            'audio_avg_processing_time': round(sum(audio_times)/len(audio_times), 3) if audio_times else 0,
            'total_detections': len(video_times) + len(audio_times),
            'video_throughput': len(video_times) / max(sum(video_times), 1) if video_times else 0,
            'audio_throughput': len(audio_times) / max(sum(audio_times), 1) if audio_times else 0
        }
    
    def _get_error_performance(self) -> Dict[str, Any]:
        """Get error performance metrics"""
        return {
            'total_errors': sum(len(v) for k, v in self.performance_history.items() if k.startswith('error_')),
            'recent_error_rate': self._calculate_recent_error_rate(),
            'alert_count': sum(1 for metric in self.ALERT_FIRED.collect() for sample in metric.samples)
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        total_requests = sum(len(v) for k, v in self.performance_history.items() if 'duration' in k)
        total_errors = sum(len(v) for k, v in self.performance_history.items() if k.startswith('error_'))
        return round((total_errors / max(total_requests, 1)) * 100, 2)
    
    def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate"""
        recent_requests = sum(len(self.performance_history[key][-10:]) 
                            for key in self.performance_history.keys() 
                            if 'duration' in key)
        recent_errors = sum(len(self.performance_history[key][-10:]) 
                          for key in self.performance_history.keys() 
                          if key.startswith('error_'))
        
        return round((recent_errors / max(recent_requests, 1)) * 100, 2)
    
    def _get_performance_predictions(self) -> Dict[str, Any]:
        """Get performance predictions based on trends"""
        # Simple trend analysis
        predictions = {}
        
        # Response time trend
        durations = list(self.performance_history.get('GET_/api/v1/health_duration', []))
        if len(durations) >= 10:
            recent_avg = sum(durations[-10:]) / 10
            older_avg = sum(durations[-20:-10]) / 10 if len(durations) >= 20 else recent_avg
            
            trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            predictions['response_time_trend'] = 'increasing' if trend > 0.1 else 'decreasing' if trend < -0.1 else 'stable'
            predictions['predicted_response_time'] = round(recent_avg * (1 + trend), 3)
        
        return predictions
    
    def get_prometheus_metrics(self) -> str:
        """Get all Prometheus metrics as text"""
        return generate_latest().decode('utf-8')

# Global monitoring instance
monitoring_system = EnhancedMonitoringSystem()

# Convenience functions
def record_request_metrics(method: str, endpoint: str, status: int, duration: float,
                          request_size: int = 0, response_size: int = 0, user_agent: str = "unknown"):
    """Record HTTP request metrics"""
    monitoring_system.record_http_request(
        method, endpoint, status, duration, request_size, response_size, user_agent
    )

def record_detection_metrics(detection_type: str, media_type: str, processing_time: float, file_size: int = 0):
    """Record detection processing metrics"""
    monitoring_system.record_detection_processing(detection_type, media_type, processing_time, file_size)

def record_model_inference(model_type: str, operation: str, inference_time: float):
    """Record model inference metrics"""
    monitoring_system.record_model_inference(model_type, operation, inference_time)

def record_error(error_type: str, endpoint: str = "unknown", severity: str = "error", message: str = ""):
    """Record error metrics"""
    monitoring_system.record_error(error_type, endpoint, severity, message)

def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary"""
    return monitoring_system.get_performance_summary()

def setup_monitoring():
    """Setup monitoring system"""
    logger.info("Enhanced monitoring system initialized")
    return monitoring_system

def setup_metrics_endpoint(app: FastAPI):
    """Setup Prometheus metrics endpoint"""
    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(monitoring_system.get_prometheus_metrics(), media_type="text/plain")
    
    @app.get("/performance")
    async def performance_summary():
        """Get human-readable performance summary"""
        return monitoring_system.get_performance_summary()

# FastAPI middleware for automatic metrics collection
def setup_monitoring_middleware(app: FastAPI):
    """Setup automatic monitoring middleware"""
    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        start_time = time.time()
        
        # Track active connections
        monitoring_system.ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            record_error("middleware_exception", str(request.url), "error", str(e))
            raise
        finally:
            monitoring_system.ACTIVE_CONNECTIONS.dec()
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Get request/response sizes (if available)
        request_size = int(request.headers.get('content-length', 0))
        response_size = int(response.headers.get('content-length', 0))
        
        # Record metrics
        record_request_metrics(
            method=request.method,
            endpoint=str(request.url.path),
            status=status_code,
            duration=duration,
            request_size=request_size,
            response_size=response_size,
            user_agent=request.headers.get('user-agent', 'unknown')
        )
        
        return response

# Export commonly used items
__all__ = [
    "setup_monitoring",
    "setup_metrics_endpoint",
    "setup_monitoring_middleware",
    "record_request_metrics",
    "record_detection_metrics", 
    "record_model_inference",
    "record_error",
    "get_performance_summary",
    "monitoring_system"
]