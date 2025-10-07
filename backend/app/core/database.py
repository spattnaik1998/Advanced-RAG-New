from sqlalchemy.orm import Session
from app.models.database import DocumentChunk
from typing import List


def save_chunks_metadata(db: Session, chunks_data: List[dict]) -> int:
    """
    Save chunk metadata to database

    Args:
        db: Database session
        chunks_data: List of dictionaries containing chunk metadata

    Returns:
        Number of chunks saved
    """
    db_chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
    db.bulk_save_objects(db_chunks)
    db.commit()
    return len(db_chunks)


def get_total_chunks(db: Session) -> int:
    """Get total number of chunks in database"""
    return db.query(DocumentChunk).count()


def get_chunks_by_document(db: Session, document_name: str) -> List[DocumentChunk]:
    """Get all chunks for a specific document"""
    return db.query(DocumentChunk).filter(
        DocumentChunk.document_name == document_name
    ).all()
