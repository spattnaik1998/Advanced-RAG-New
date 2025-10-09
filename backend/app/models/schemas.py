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


class EnsembleRequest(BaseModel):
    """Request model for Ensemble (Hybrid) queries"""
    query: str
    k: int = Field(default=5, description="Number of chunks to retrieve")
    vector_weight: float = Field(default=0.6, description="Weight for vector retriever")
    bm25_weight: float = Field(default=0.4, description="Weight for BM25 retriever")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")


class EnsembleMetadata(BaseModel):
    """Metadata for ensemble retrieval"""
    vector_weight: float
    bm25_weight: float
    build_time_ms: float
    retrieval_time_ms: float


class EnsembleResponse(BaseModel):
    """Response model for Ensemble queries"""
    pipeline: str = "ensemble"
    query: str
    k: int
    model: str
    retrieved_docs: List[RetrievedDoc]
    answer: str
    tokens_used: Optional[int] = None
    llm_raw_response: str
    latency_ms: float
    ensemble_metadata: EnsembleMetadata


class CohereCompressionRequest(BaseModel):
    """Request model for Cohere Compression queries"""
    query: str
    k: int = Field(default=5, description="Initial number of chunks to retrieve")
    top_n: int = Field(default=5, description="Top N chunks after Cohere reranking")
    use_ensemble: bool = Field(default=True, description="Use ensemble (FAISS+BM25) as base retriever")
    vector_weight: float = Field(default=0.5, description="Weight for vector retriever (if ensemble)")
    bm25_weight: float = Field(default=0.5, description="Weight for BM25 retriever (if ensemble)")
    rerank_model: str = Field(default="rerank-english-v3.0", description="Cohere rerank model")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")


class CompressionMetadata(BaseModel):
    """Metadata for compression retrieval"""
    initial_k: int
    reranked_top_n: int
    use_ensemble: bool
    vector_weight: float
    bm25_weight: float
    rerank_model: str
    build_time_ms: float
    retrieval_time_ms: float
    docs_returned: int


class CohereCompressionResponse(BaseModel):
    """Response model for Cohere Compression queries"""
    pipeline: str = "cohere_compression"
    query: str
    k: int
    model: str
    retrieved_docs: List[RetrievedDoc]
    answer: str
    tokens_used: Optional[int] = None
    llm_raw_response: str
    latency_ms: float
    compression_metadata: CompressionMetadata


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


# Comparison/Evaluation schemas
class CompareRequest(BaseModel):
    """Request model for pipeline comparison"""
    query: str
    k: int = Field(default=5, description="Number of chunks to retrieve")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    skip_pipelines: List[str] = Field(default=[], description="Pipelines to skip")
    rrf_num_variants: int = Field(default=4, description="Number of query variants for RRF")
    ensemble_vector_weight: float = Field(default=0.6, description="Vector weight for ensemble")
    cohere_top_n: int = Field(default=5, description="Top N for Cohere reranking")
    cohere_use_ensemble: bool = Field(default=True, description="Use ensemble for Cohere")


class CompareResponse(BaseModel):
    """Response model for pipeline comparison"""
    comparison_id: Optional[str] = None
    query: str
    params: dict
    results: dict
    metrics: dict
    errors: dict
    total_execution_time_ms: float
    pipelines_run: List[str]
    pipelines_failed: List[str]
