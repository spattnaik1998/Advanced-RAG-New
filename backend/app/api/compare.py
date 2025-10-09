"""
Comparison endpoint for evaluating all RAG pipelines
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
import logging

from app.models.schemas import CompareRequest, CompareResponse
from app.models.database import get_db, ComparisonRun, init_db
from app.eval import run_all_pipelines

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize database on module load
init_db()


@router.post("/compare", response_model=CompareResponse)
async def compare_pipelines(
    request: CompareRequest,
    db: Session = Depends(get_db)
):
    """
    Compare all RAG pipelines and return metrics

    Args:
        request: Comparison request with query and parameters
        db: Database session

    Returns:
        CompareResponse with all pipeline results and metrics
    """
    try:
        # Build params dictionary from request
        params = {
            "k": request.k,
            "model": request.model,
            "skip_pipelines": request.skip_pipelines,
            "rrf_num_variants": request.rrf_num_variants,
            "ensemble_vector_weight": request.ensemble_vector_weight,
            "cohere_top_n": request.cohere_top_n,
            "cohere_use_ensemble": request.cohere_use_ensemble
        }

        # Run all pipelines
        comparison = run_all_pipelines(request.query, params)

        # Generate comparison ID
        comparison_id = str(uuid.uuid4())
        comparison["comparison_id"] = comparison_id

        # Save to database
        try:
            db_comparison = ComparisonRun(
                comparison_id=comparison_id,
                query=comparison["query"],
                params=comparison["params"],
                results=comparison["results"],
                metrics=comparison["metrics"],
                errors=comparison["errors"],
                total_execution_time_ms=comparison["total_execution_time_ms"],
                pipelines_run=comparison["pipelines_run"],
                pipelines_failed=comparison["pipelines_failed"]
            )
            db.add(db_comparison)
            db.commit()

            logger.info(f"Comparison saved with ID: {comparison_id}")

        except Exception as db_error:
            logger.error(f"Failed to save comparison to database: {str(db_error)}")
            # Continue even if DB save fails

        return CompareResponse(**comparison)

    except Exception as e:
        logger.error(f"Error in compare_pipelines: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running pipeline comparison: {str(e)}"
        )


@router.get("/compare/{comparison_id}", response_model=CompareResponse)
async def get_comparison(
    comparison_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve a saved comparison by ID

    Args:
        comparison_id: UUID of the comparison
        db: Database session

    Returns:
        CompareResponse with saved comparison data
    """
    try:
        comparison = db.query(ComparisonRun).filter(
            ComparisonRun.comparison_id == comparison_id
        ).first()

        if not comparison:
            raise HTTPException(
                status_code=404,
                detail=f"Comparison with ID {comparison_id} not found"
            )

        return CompareResponse(
            comparison_id=comparison.comparison_id,
            query=comparison.query,
            params=comparison.params or {},
            results=comparison.results or {},
            metrics=comparison.metrics or {},
            errors=comparison.errors or {},
            total_execution_time_ms=comparison.total_execution_time_ms or 0.0,
            pipelines_run=comparison.pipelines_run or [],
            pipelines_failed=comparison.pipelines_failed or []
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving comparison: {str(e)}"
        )
