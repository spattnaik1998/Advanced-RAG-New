from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.core.config import settings
import uuid

Base = declarative_base()


class DocumentChunk(Base):
    """SQLite model for storing document chunk metadata"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True, nullable=False)
    document_name = Column(String, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    vector_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ComparisonRun(Base):
    """SQLite model for storing pipeline comparison runs"""
    __tablename__ = "comparison_runs"

    id = Column(Integer, primary_key=True, index=True)
    comparison_id = Column(String, unique=True, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    params = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    errors = Column(JSON, nullable=True)
    total_execution_time_ms = Column(Float, nullable=True)
    pipelines_run = Column(JSON, nullable=True)
    pipelines_failed = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Create engine and session
engine = create_engine(
    f"sqlite:///{settings.METADATA_DB}",
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
