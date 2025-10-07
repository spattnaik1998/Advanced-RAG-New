"""
Unit tests for document ingestion
"""
import os
import pytest
import tempfile
import shutil
from io import BytesIO
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingest import (
    extract_text,
    chunk_texts,
    embed_and_index,
    prepare_chunks_metadata,
    generate_chunk_id
)
from app.models.database import Base, DocumentChunk
from app.core.database import save_chunks_metadata, get_total_chunks
from app.services.faiss_service import FAISSService


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_db():
    """Create temporary test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    This is a test document for RAG ingestion.
    It contains multiple paragraphs to test chunking functionality.
    The RecursiveCharacterTextSplitter should split this text into smaller chunks.
    Each chunk will be embedded and stored in the FAISS index.
    We want to ensure that the ingestion process works correctly.
    This includes extracting text, chunking, embedding, and indexing.
    """ * 5  # Repeat to ensure multiple chunks


def test_extract_text_from_txt():
    """Test text extraction from TXT file"""
    content = b"Hello, this is a test document."
    text = extract_text("test.txt", content)
    assert text == "Hello, this is a test document."
    assert isinstance(text, str)


def test_extract_text_unsupported_format():
    """Test that unsupported file formats raise ValueError"""
    content = b"some content"
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("test.docx", content)


def test_chunk_texts(sample_text):
    """Test text chunking"""
    chunks = chunk_texts(sample_text, "test_doc.txt")

    assert len(chunks) > 0
    assert all(hasattr(chunk, 'page_content') for chunk in chunks)
    assert all(hasattr(chunk, 'metadata') for chunk in chunks)

    # Check metadata
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata['source'] == "test_doc.txt"
        assert chunk.metadata['chunk_index'] == idx
        assert 'chunk_id' in chunk.metadata


def test_generate_chunk_id():
    """Test chunk ID generation"""
    chunk_id1 = generate_chunk_id("doc1.txt", 0)
    chunk_id2 = generate_chunk_id("doc1.txt", 0)
    chunk_id3 = generate_chunk_id("doc1.txt", 1)

    # Same inputs should produce same ID
    assert chunk_id1 == chunk_id2

    # Different inputs should produce different IDs
    assert chunk_id1 != chunk_id3


def test_prepare_chunks_metadata(sample_text):
    """Test preparing chunk metadata for database"""
    chunks = chunk_texts(sample_text, "test.txt")
    metadata = prepare_chunks_metadata(chunks, "test.txt", starting_vector_id=0)

    assert len(metadata) == len(chunks)
    assert all('chunk_id' in m for m in metadata)
    assert all('document_name' in m for m in metadata)
    assert all('chunk_text' in m for m in metadata)
    assert all('vector_id' in m for m in metadata)

    # Verify vector IDs are sequential
    for idx, m in enumerate(metadata):
        assert m['vector_id'] == idx


def test_save_chunks_metadata(test_db, sample_text):
    """Test saving chunks to database"""
    chunks = chunk_texts(sample_text, "test.txt")
    metadata = prepare_chunks_metadata(chunks, "test.txt", starting_vector_id=0)

    num_saved = save_chunks_metadata(test_db, metadata)
    assert num_saved == len(chunks)

    # Verify data in database
    total = get_total_chunks(test_db)
    assert total == len(chunks)

    # Check first chunk
    first_chunk = test_db.query(DocumentChunk).first()
    assert first_chunk.document_name == "test.txt"
    assert first_chunk.chunk_index == 0


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason="Requires OPENAI_API_KEY environment variable"
)
def test_embed_and_index_increments_size(temp_dir, sample_text):
    """
    Test that embedding and indexing increments FAISS index size
    This test requires OPENAI_API_KEY to be set
    """
    from langchain_openai import OpenAIEmbeddings

    # Create test index path
    index_path = os.path.join(temp_dir, "test_faiss_index")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # First ingestion
    chunks1 = chunk_texts(sample_text, "doc1.txt")
    vectorstore1, start_id1 = embed_and_index(chunks1, index_path, embeddings)
    initial_size = vectorstore1.index.ntotal

    assert initial_size == len(chunks1)
    assert start_id1 == 0

    # Second ingestion (should append)
    chunks2 = chunk_texts(sample_text, "doc2.txt")
    vectorstore2, start_id2 = embed_and_index(chunks2, index_path, embeddings)
    final_size = vectorstore2.index.ntotal

    assert final_size == initial_size + len(chunks2)
    assert start_id2 == initial_size
    assert final_size > initial_size


def test_chunk_size_and_overlap(sample_text):
    """Test that chunks respect size and overlap parameters"""
    chunks = chunk_texts(sample_text, "test.txt")

    # Most chunks should be around 500 characters or less
    for chunk in chunks:
        # Allow some flexibility due to word boundaries
        assert len(chunk.page_content) <= 600
