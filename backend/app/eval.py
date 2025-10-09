"""
Evaluation module for running and comparing all RAG pipelines
"""
import time
from typing import Dict, Optional
from app.services.naive_rag_service import naive_rag_service
from app.services.rrf_fusion_service import rrf_fusion_service
from app.services.ensemble_service import ensemble_service
from app.services.cohere_compression_service import cohere_compression_service
from app.eval.metrics import evaluation_metrics
from app.core.config import settings


def extract_context_from_results(result: dict) -> str:
    """
    Extract context string from pipeline result

    Args:
        result: Pipeline result dictionary

    Returns:
        Context string built from retrieved documents
    """
    docs = result.get("retrieved_docs", [])
    context_parts = []

    for idx, doc in enumerate(docs, 1):
        chunk_id = doc.get("chunk_id", "unknown")
        snippet = doc.get("snippet", "")
        context_parts.append(f"[Doc {idx}] (chunk_id: {chunk_id})\n{snippet}")

    return "\n\n".join(context_parts)


def run_all_pipelines(
    query: str,
    params: Optional[Dict] = None
) -> Dict:
    """
    Run all RAG pipelines and compare results

    Args:
        query: User query
        params: Optional parameters for pipelines
            {
                "k": int (default: 5),
                "model": str (default: "gpt-4o-mini"),
                "skip_pipelines": list (pipelines to skip),
                "rrf_num_variants": int (default: 4),
                "ensemble_vector_weight": float (default: 0.6),
                "cohere_top_n": int (default: 5),
                "cohere_use_ensemble": bool (default: True)
            }

    Returns:
        Dictionary with all results and metrics
    """
    if params is None:
        params = {}

    k = params.get("k", 5)
    model = params.get("model", "gpt-4o-mini")
    skip_pipelines = params.get("skip_pipelines", [])

    total_start = time.time()

    results = {}
    contexts = {}
    errors = {}

    # 1. Run Naive RAG
    if "naive_rag" not in skip_pipelines:
        try:
            if naive_rag_service.faiss_service.index_exists():
                result = naive_rag_service.query(query=query, k=k, model=model)
                results["naive_rag"] = result
                contexts["naive_rag"] = extract_context_from_results(result)
            else:
                errors["naive_rag"] = "FAISS index not found"
        except Exception as e:
            errors["naive_rag"] = str(e)

    # 2. Run RRF Fusion
    if "rrf_fusion" not in skip_pipelines:
        try:
            if rrf_fusion_service.faiss_service.index_exists():
                num_variants = params.get("rrf_num_variants", 4)
                result = rrf_fusion_service.query(
                    query=query,
                    num_query_variants=num_variants,
                    k_per_query=k,
                    top_m=k,
                    model=model
                )
                results["rrf_fusion"] = result
                contexts["rrf_fusion"] = extract_context_from_results(result)
            else:
                errors["rrf_fusion"] = "FAISS index not found"
        except Exception as e:
            errors["rrf_fusion"] = str(e)

    # 3. Run Ensemble
    if "ensemble" not in skip_pipelines:
        try:
            if ensemble_service.faiss_service.index_exists():
                vector_weight = params.get("ensemble_vector_weight", 0.6)
                bm25_weight = 1.0 - vector_weight

                result = ensemble_service.query(
                    query=query,
                    k=k,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    model=model
                )
                results["ensemble"] = result
                contexts["ensemble"] = extract_context_from_results(result)
            else:
                errors["ensemble"] = "FAISS index not found"
        except Exception as e:
            errors["ensemble"] = str(e)

    # 4. Run Cohere Compression (if API key available)
    if "cohere_compression" not in skip_pipelines:
        try:
            if cohere_compression_service.check_cohere_api_key():
                if cohere_compression_service.faiss_service.index_exists():
                    top_n = params.get("cohere_top_n", k)
                    use_ensemble = params.get("cohere_use_ensemble", True)

                    result = cohere_compression_service.query(
                        query=query,
                        k=k,
                        top_n=top_n,
                        use_ensemble=use_ensemble,
                        model=model
                    )
                    results["cohere_compression"] = result
                    contexts["cohere_compression"] = extract_context_from_results(result)
                else:
                    errors["cohere_compression"] = "FAISS index not found"
            else:
                errors["cohere_compression"] = "COHERE_API_KEY not configured"
        except Exception as e:
            errors["cohere_compression"] = str(e)

    total_time = (time.time() - total_start) * 1000  # Convert to ms

    # Compute metrics if we have results
    metrics = {}
    if results:
        metrics = evaluation_metrics.compute_all_metrics(results, contexts)

    # Build comparison response
    comparison = {
        "query": query,
        "params": params,
        "results": results,
        "metrics": metrics,
        "errors": errors,
        "total_execution_time_ms": round(total_time, 2),
        "pipelines_run": list(results.keys()),
        "pipelines_failed": list(errors.keys())
    }

    return comparison
