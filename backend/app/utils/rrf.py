"""
Reciprocal Rank Fusion (RRF) utilities
Adapted from the user's rrf.py implementation
"""
from langchain.load import dumps, loads
from typing import List


def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List[tuple]:
    """
    Reciprocal Rank Fusion algorithm for combining multiple retrieval results

    Args:
        results: List of lists of documents (each list from a different query/retriever)
        k: Constant for RRF formula (default: 60)

    Returns:
        List of tuples (document, fused_score) sorted by score descending
    """
    fused_scores = {}

    for docs in results:
        # Assumes docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort by score in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results
