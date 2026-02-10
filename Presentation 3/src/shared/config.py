"""
Configuration Management
Handles environment variables and application settings
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "TrueSight"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/truesight"
    REDIS_URL: str = "redis://localhost:6379/0"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "your-jwt-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Model Paths
    MODEL_DIR: str = "/app/models"
    VIDEO_MODEL_PATH: str = "/app/models/video_detector.onnx"
    AUDIO_MODEL_PATH: str = "/app/models/audio_detector.onnx"
    MULTILINGUAL_MODEL_PATH: str = "/app/models/multilingual_detector"
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_VIDEO_STREAMS: str = "video-streams"
    KAFKA_TOPIC_AUDIO_STREAMS: str = "audio-streams"
    KAFKA_TOPIC_DETECTION_RESULTS: str = "detection-results"
    
    # Blockchain Configuration
    BLOCKCHAIN_NETWORK: str = "fabric-testnet"
    BLOCKCHAIN_CONTRACT_ID: str = "evidence-contract"
    BLOCKCHAIN_PEER_URL: str = "grpc://localhost:7051"
    BLOCKCHAIN_ORDERER_URL: str = "grpc://localhost:7050"
    
    # Storage Configuration
    STORAGE_BUCKET_NAME: str = "truesight-media"
    STORAGE_REGION: str = "us-east-1"
    STORAGE_ENDPOINT: str = "https://s3.amazonaws.com"
    
    # GPU Configuration
    GPU_ENABLED: bool = True
    GPU_DEVICE_ID: int = 0
    MAX_BATCH_SIZE: int = 32
    
    # Performance Settings
    MAX_CONCURRENT_STREAMS: int = 1000
    PROCESSING_TIMEOUT_SECONDS: int = 30
    CACHE_TTL_SECONDS: int = 3600
    
    # Compliance Settings
    GDPR_ENABLED: bool = True
    HIPAA_ENABLED: bool = True
    SOC2_ENABLED: bool = True
    EVIDENCE_RETENTION_DAYS: int = 365
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Export for use in other modules
__all__ = ["settings"]