"""
Document ingestion utilities for processing and indexing documents
"""
import os
from typing import List, Tuple
from fastapi import UploadFile
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import hashlib


async def load_file(file: UploadFile) -> Tuple[str, str]:
    """
    Load file and return filename and content

    Args:
        file: Uploaded file object

    Returns:
        Tuple of (filename, file_content)
    """
    filename = file.filename
    content = await file.read()
    return filename, content


def extract_text(filename: str, content: bytes) -> str:
    """
    Extract text from file based on file type

    Args:
        filename: Name of the file
        content: File content as bytes

    Returns:
        Extracted text content

    Raises:
        ValueError: If file type is not supported
    """
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == '.txt':
        return content.decode('utf-8')

    elif file_extension == '.pdf':
        return extract_text_from_pdf(content)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .txt and .pdf are supported.")


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from PDF content

    Args:
        content: PDF file content as bytes

    Returns:
        Extracted text from all pages
    """
    import io
    pdf_file = io.BytesIO(content)
    pdf_reader = pypdf.PdfReader(pdf_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    return text


def chunk_texts(text: str, document_name: str) -> List[Document]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter

    Args:
        text: Text to split
        document_name: Name of the source document

    Returns:
        List of Document objects with chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)

    # Create Document objects with metadata
    documents = []
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": document_name,
                "chunk_index": idx,
                "chunk_id": generate_chunk_id(document_name, idx)
            }
        )
        documents.append(doc)

    return documents


def generate_chunk_id(document_name: str, chunk_index: int) -> str:
    """
    Generate unique chunk ID

    Args:
        document_name: Name of the document
        chunk_index: Index of the chunk

    Returns:
        Unique chunk ID
    """
    unique_string = f"{document_name}_{chunk_index}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def embed_and_index(
    chunks: List[Document],
    faiss_index_path: str,
    embeddings: OpenAIEmbeddings
) -> Tuple[FAISS, int]:
    """
    Embed chunks and add to FAISS index

    Args:
        chunks: List of Document objects to embed
        faiss_index_path: Path to persist FAISS index
        embeddings: OpenAI embeddings instance

    Returns:
        Tuple of (FAISS vectorstore, starting_vector_id)
    """
    # Check if index exists
    if os.path.exists(faiss_index_path) and os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
        # Load existing index
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        starting_vector_id = vectorstore.index.ntotal

        # Add new documents
        vectorstore.add_documents(chunks)
    else:
        # Create new index
        starting_vector_id = 0
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist the index
    os.makedirs(faiss_index_path, exist_ok=True)
    vectorstore.save_local(faiss_index_path)

    return vectorstore, starting_vector_id


def prepare_chunks_metadata(
    chunks: List[Document],
    document_name: str,
    starting_vector_id: int
) -> List[dict]:
    """
    Prepare chunk metadata for database storage

    Args:
        chunks: List of Document objects
        document_name: Name of the source document
        starting_vector_id: Starting ID in FAISS index

    Returns:
        List of dictionaries with chunk metadata
    """
    chunks_data = []
    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": chunk.metadata["chunk_id"],
            "document_name": document_name,
            "chunk_text": chunk.page_content,
            "chunk_index": chunk.metadata["chunk_index"],
            "vector_id": starting_vector_id + idx
        }
        chunks_data.append(chunk_data)

    return chunks_data
