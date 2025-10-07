from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str
    method: str  # 'simple', 'hybrid', 'fusion'


class DocumentChunk(BaseModel):
    """Model for document chunks"""
    content: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    query: str
    method: str
    answer: str
    retrieved_chunks: List[DocumentChunk]
    execution_time: float
