"""
Database Connection and ORM Models
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import asyncio
from loguru import logger

from src.shared.config import settings

# Create base class for declarative models
Base = declarative_base()

# Database Models
class User(Base):
    """User model for authentication and access control"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    detections = relationship("DetectionResult", back_populates="user")
    forensic_analyses = relationship("ForensicAnalysis", back_populates="user")

class MediaFile(Base):
    """Media file metadata storage"""
    __tablename__ = "media_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)  # in bytes
    mime_type = Column(String)
    duration = Column(Integer)  # in seconds for audio/video
    checksum = Column(String)  # SHA256 hash
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    detections = relationship("DetectionResult", back_populates="media_file")
    forensic_analyses = relationship("ForensicAnalysis", back_populates="media_file")

class DetectionResult(Base):
    """Detection results storage"""
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    media_file_id = Column(Integer, ForeignKey("media_files.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    detection_type = Column(String, nullable=False)  # "video", "audio", "multimodal"
    confidence_score = Column(Integer)  # 0-100
    is_deepfake = Column(Boolean)
    processing_time_ms = Column(Integer)
    detected_artifacts = Column(Text)  # JSON string of detected anomalies
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="detections")
    media_file = relationship("MediaFile", back_populates="detections")
    forensic_analysis = relationship("ForensicAnalysis", back_populates="detection_result")

class ForensicAnalysis(Base):
    """Forensic analysis results"""
    __tablename__ = "forensic_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    media_file_id = Column(Integer, ForeignKey("media_files.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    detection_result_id = Column(Integer, ForeignKey("detection_results.id"))
    prnu_analysis = Column(Text)  # JSON of PRNU results
    metadata_extraction = Column(Text)  # JSON of extracted metadata
    compression_analysis = Column(Text)  # JSON of compression artifacts
    device_fingerprint = Column(String)  # Identified device signature
    source_attribution = Column(String)  # Probable source device/model
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="forensic_analyses")
    media_file = relationship("MediaFile", back_populates="forensic_analyses")
    detection_result = relationship("DetectionResult", back_populates="forensic_analysis")

class EvidenceLog(Base):
    """Immutable evidence logging for blockchain"""
    __tablename__ = "evidence_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_result_id = Column(Integer, ForeignKey("detection_results.id"))
    forensic_analysis_id = Column(Integer, ForeignKey("forensic_analyses.id"))
    blockchain_tx_hash = Column(String, unique=True)
    timestamp_anchor = Column(String)  # Blockchain timestamp
    evidence_hash = Column(String)  # Hash of evidence data
    custody_chain = Column(Text)  # JSON of custody transfers
    logged_at = Column(DateTime, default=datetime.utcnow)

# Database setup functions
async def init_db():
    """Initialize database connections"""
    try:
        # Create async engine
        engine = create_async_engine(
            settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
            echo=settings.DEBUG,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Create session factory
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Store engine and session factory globally
        global async_engine, SessionLocal
        async_engine = engine
        SessionLocal = async_session
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database initialized successfully")
        return engine, async_session
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def get_db():
    """Dependency for getting database session"""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()

# Export commonly used items
__all__ = [
    "Base",
    "User",
    "MediaFile", 
    "DetectionResult",
    "ForensicAnalysis",
    "EvidenceLog",
    "init_db",
    "get_db"
]