from pydantic import BaseModel, Field
from typing import List, Optional


class NaiveRAGRequest(BaseModel):
    """Request model for Naive RAG queries"""
    query: str
    k: int = Field(default=5, description="Number of chunks to retrieve")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")


class RetrievedDoc(BaseModel):
    """Model for retrieved document chunks"""
    chunk_id: str
    snippet: str
    score: float


class NaiveRAGResponse(BaseModel):
    """Response model for Naive RAG queries"""
    pipeline: str = "naive_rag"
    query: str
    k: int
    model: str
    retrieved_docs: List[RetrievedDoc]
    answer: str
    tokens_used: Optional[int] = None
    llm_raw_response: str
    latency_ms: float


class RRFFusionRequest(BaseModel):
    """Request model for RRF Fusion queries"""
    query: str
    num_query_variants: int = Field(default=4, description="Number of query variants to generate")
    k_per_query: int = Field(default=5, description="Number of docs to retrieve per query variant")
    top_m: int = Field(default=5, description="Number of top fused docs to use for context")
    rrf_k: int = Field(default=60, description="RRF constant for fusion formula")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")


class FusionMetadata(BaseModel):
    """Metadata for fusion process"""
    queries_generated: List[str]
    num_variants: int
    k_per_query: int
    fusion_scores: List[dict]


class RRFFusionResponse(BaseModel):
    """Response model for RRF Fusion queries"""
    pipeline: str = "rrf_fusion"
    query: str
    k: int
    model: str
    retrieved_docs: List[RetrievedDoc]
    answer: str
    tokens_used: Optional[int] = None
    llm_raw_response: str
    latency_ms: float
    fusion_metadata: FusionMetadata


# Legacy schemas (keeping for compatibility)
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
