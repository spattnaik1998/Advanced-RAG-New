"""
Query endpoint for RAG operations
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    NaiveRAGRequest,
    NaiveRAGResponse,
    RRFFusionRequest,
    RRFFusionResponse
)
from app.services.naive_rag_service import naive_rag_service
from app.services.rrf_fusion_service import rrf_fusion_service
from app.core.query_logger import log_query
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=NaiveRAGResponse)
async def naive_rag_query(request: NaiveRAGRequest):
    """
    Execute Naive RAG query

    Args:
        request: Query request with query text, k value, and model name

    Returns:
        NaiveRAGResponse with answer and retrieval metadata
    """
    try:
        # Check if FAISS index exists
        if not naive_rag_service.faiss_service.index_exists():
            raise HTTPException(
                status_code=400,
                detail="No documents have been indexed. Please upload documents first."
            )

        # Execute RAG pipeline
        result = naive_rag_service.query(
            query=request.query,
            k=request.k,
            model=request.model
        )

        # Log the query and response
        log_query(
            pipeline=result["pipeline"],
            query=request.query,
            k=request.k,
            model=request.model,
            answer=result["answer"],
            retrieved_docs_count=len(result["retrieved_docs"]),
            latency_ms=result["latency_ms"],
            tokens_used=result["tokens_used"]
        )

        return NaiveRAGResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in naive_rag_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/query/rrf_fusion", response_model=RRFFusionResponse)
async def rrf_fusion_query(request: RRFFusionRequest):
    """
    Execute RRF Fusion RAG query

    Args:
        request: Query request with query variants, fusion parameters, and model name

    Returns:
        RRFFusionResponse with answer, retrieval metadata, and fusion details
    """
    try:
        # Check if FAISS index exists
        if not rrf_fusion_service.faiss_service.index_exists():
            raise HTTPException(
                status_code=400,
                detail="No documents have been indexed. Please upload documents first."
            )

        # Execute RRF Fusion pipeline
        result = rrf_fusion_service.query(
            query=request.query,
            num_query_variants=request.num_query_variants,
            k_per_query=request.k_per_query,
            top_m=request.top_m,
            rrf_k=request.rrf_k,
            model=request.model
        )

        # Log the query and response
        log_query(
            pipeline=result["pipeline"],
            query=request.query,
            k=request.top_m,
            model=request.model,
            answer=result["answer"],
            retrieved_docs_count=len(result["retrieved_docs"]),
            latency_ms=result["latency_ms"],
            tokens_used=result["tokens_used"]
        )

        return RRFFusionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in rrf_fusion_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
