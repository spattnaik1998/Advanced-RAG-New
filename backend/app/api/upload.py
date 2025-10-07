"""
Upload endpoint for document ingestion
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict
import time

from app.models.database import get_db, init_db
from app.core.database import save_chunks_metadata
from app.services.faiss_service import faiss_service
from app.ingest import (
    load_file,
    extract_text,
    chunk_texts,
    embed_and_index,
    prepare_chunks_metadata
)

router = APIRouter()

# Initialize database on module load
init_db()


@router.post("/upload", response_model=Dict)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and ingest a document (PDF or TXT)

    Args:
        file: Uploaded file (PDF or TXT)
        db: Database session

    Returns:
        Ingestion summary with number of docs, chunks, and index path
    """
    start_time = time.time()

    try:
        # Step 1: Load file
        filename, content = await load_file(file)

        # Step 2: Extract text
        try:
            text = extract_text(filename, content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")

        # Step 3: Chunk texts
        chunks = chunk_texts(text, filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks were created from the document")

        # Step 4: Get FAISS service and embeddings
        embeddings = faiss_service.get_embeddings()
        index_path = faiss_service.index_path

        # Get current index size before adding new chunks
        initial_index_size = faiss_service.get_index_size()

        # Step 5: Embed and index
        vectorstore, starting_vector_id = embed_and_index(
            chunks,
            index_path,
            embeddings
        )

        # Step 6: Prepare and save metadata
        chunks_metadata = prepare_chunks_metadata(chunks, filename, starting_vector_id)
        num_saved = save_chunks_metadata(db, chunks_metadata)

        # Get final index size
        final_index_size = vectorstore.index.ntotal

        execution_time = time.time() - start_time

        return {
            "status": "success",
            "message": f"Successfully ingested {filename}",
            "number_of_docs": 1,
            "number_of_chunks": len(chunks),
            "chunks_saved_to_db": num_saved,
            "index_path": index_path,
            "initial_index_size": initial_index_size,
            "final_index_size": final_index_size,
            "execution_time_seconds": round(execution_time, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
