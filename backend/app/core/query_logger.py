"""
Query logging functionality
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logger
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

QUERIES_LOG_FILE = LOGS_DIR / "queries.jsonl"


def log_query(
    pipeline: str,
    query: str,
    k: int,
    model: str,
    answer: str,
    retrieved_docs_count: int,
    latency_ms: float,
    tokens_used: Optional[int] = None
):
    """
    Log query request and response to queries log file

    Args:
        pipeline: Pipeline name (e.g., 'naive_rag')
        query: User query text
        k: Number of chunks retrieved
        model: Model name used
        answer: Generated answer
        retrieved_docs_count: Number of documents retrieved
        latency_ms: Total latency in milliseconds
        tokens_used: Number of tokens used (optional)
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": pipeline,
        "query": query,
        "k": k,
        "model": model,
        "answer": answer,
        "retrieved_docs_count": retrieved_docs_count,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used
    }

    try:
        # Append to JSONL file
        with open(QUERIES_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"Query logged: {pipeline} - {query[:50]}... (latency: {latency_ms}ms)")

    except Exception as e:
        logger.error(f"Failed to log query: {str(e)}")


def get_query_logs(limit: int = 100) -> list:
    """
    Retrieve recent query logs

    Args:
        limit: Maximum number of logs to retrieve

    Returns:
        List of log entries
    """
    if not QUERIES_LOG_FILE.exists():
        return []

    try:
        with open(QUERIES_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Get last N lines
        recent_lines = lines[-limit:] if len(lines) > limit else lines

        # Parse JSON lines
        logs = [json.loads(line) for line in recent_lines]

        return logs

    except Exception as e:
        logger.error(f"Failed to read query logs: {str(e)}")
        return []
