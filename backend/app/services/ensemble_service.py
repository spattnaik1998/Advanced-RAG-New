"""
Ensemble RAG Service - Hybrid retrieval combining FAISS vector search + BM25 keyword search
"""
import time
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from app.services.faiss_service import faiss_service
from app.models.schemas import RetrievedDoc
from app.core.database import get_total_chunks
from app.models.database import SessionLocal, DocumentChunk


class EnsembleService:
    """Service for Ensemble (Hybrid) RAG implementation"""

    def __init__(self):
        self.faiss_service = faiss_service

    def get_all_chunks_as_documents(self):
        """
        Retrieve all document chunks from database and convert to LangChain Document format

        Returns:
            List of Document objects for BM25 indexing
        """
        from langchain.docstore.document import Document

        db = SessionLocal()
        try:
            chunks = db.query(DocumentChunk).all()

            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.chunk_text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.document_name,
                        "chunk_index": chunk.chunk_index,
                        "vector_id": chunk.vector_id
                    }
                )
                documents.append(doc)

            return documents
        finally:
            db.close()

    def build_ensemble_retriever(
        self,
        k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> Tuple[EnsembleRetriever, float]:
        """
        Build ensemble retriever combining FAISS vector + BM25 keyword search

        Args:
            k: Number of documents to retrieve
            vector_weight: Weight for vector retriever (default: 0.6)
            bm25_weight: Weight for BM25 retriever (default: 0.4)

        Returns:
            Tuple of (ensemble_retriever, build_time_ms)
        """
        start_time = time.time()

        # Build FAISS vector retriever
        vectorstore = self.faiss_service.load_index()
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Build BM25 retriever from all chunks
        documents = self.get_all_chunks_as_documents()
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # Combine into ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )

        build_time = (time.time() - start_time) * 1000  # Convert to ms

        return ensemble_retriever, build_time

    def retrieve_with_ensemble(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> Tuple[List, float, float]:
        """
        Retrieve documents using ensemble retriever

        Args:
            query: User query
            k: Number of documents to retrieve
            vector_weight: Weight for vector retriever
            bm25_weight: Weight for BM25 retriever

        Returns:
            Tuple of (documents, build_time_ms, retrieval_time_ms)
        """
        # Build ensemble retriever
        ensemble_retriever, build_time = self.build_ensemble_retriever(
            k=k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )

        # Retrieve documents
        start_time = time.time()
        docs = ensemble_retriever.get_relevant_documents(query)
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        return docs, build_time, retrieval_time

    def build_context(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> Tuple[str, List[RetrievedDoc], dict, float]:
        """
        Build context from ensemble retrieval

        Args:
            query: User query
            k: Number of documents to retrieve
            vector_weight: Weight for vector retriever
            bm25_weight: Weight for BM25 retriever

        Returns:
            Tuple of (context_string, retrieved_docs, ensemble_metadata, total_time_ms)
        """
        # Retrieve with ensemble
        docs, build_time, retrieval_time = self.retrieve_with_ensemble(
            query, k, vector_weight, bm25_weight
        )

        # Build context with provenance
        context_parts = []
        retrieved_docs = []

        for idx, doc in enumerate(docs, 1):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            source = doc.metadata.get("source", "unknown")
            chunk_text = doc.page_content

            # Add to context with provenance
            context_parts.append(
                f"[Document {idx}] (source: {source}, chunk_id: {chunk_id})\n{chunk_text}"
            )

            # Add to retrieved docs (ensemble doesn't provide scores directly)
            retrieved_doc = RetrievedDoc(
                chunk_id=chunk_id,
                snippet=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                score=1.0 / (idx + 1)  # Inverse rank as pseudo-score
            )
            retrieved_docs.append(retrieved_doc)

        context_string = "\n\n".join(context_parts)

        # Build ensemble metadata
        ensemble_metadata = {
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "build_time_ms": round(build_time, 2),
            "retrieval_time_ms": round(retrieval_time, 2)
        }

        total_time = build_time + retrieval_time

        return context_string, retrieved_docs, ensemble_metadata, total_time

    def generate_answer(
        self,
        query: str,
        context: str,
        model: str = "gpt-4o-mini"
    ) -> Tuple[str, str, int, float]:
        """
        Generate answer using OpenAI LLM

        Args:
            query: User query
            context: Context string from retrieved chunks
            model: OpenAI model name

        Returns:
            Tuple of (answer, raw_response, tokens_used, generation_time_ms)
        """
        start_time = time.time()

        # Create prompt template (same as naive_rag)
        template = """Answer the question using only the context below. If answer can't be found, reply 'I don't know based on the provided context.'

Context:
{context}

Question: {query}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Initialize LLM
        llm = ChatOpenAI(model=model, temperature=0)

        # Create chain
        chain = prompt | llm | StrOutputParser()

        # Generate response
        answer = chain.invoke({"context": context, "query": query})

        generation_time = (time.time() - start_time) * 1000  # Convert to ms

        # Estimate tokens
        tokens_used = len(answer.split()) * 2  # Rough estimate

        return answer, answer, tokens_used, generation_time

    def query(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        model: str = "gpt-4o-mini"
    ) -> dict:
        """
        Execute full Ensemble RAG pipeline

        Args:
            query: User query
            k: Number of chunks to retrieve
            vector_weight: Weight for vector retriever
            bm25_weight: Weight for BM25 retriever
            model: OpenAI model name

        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()

        # Step 1: Build context with ensemble retrieval
        context, retrieved_docs, ensemble_metadata, retrieval_total_time = self.build_context(
            query, k, vector_weight, bm25_weight
        )

        # Step 2: Generate answer
        answer, raw_response, tokens_used, generation_time = self.generate_answer(
            query, context, model
        )

        total_latency = (time.time() - pipeline_start) * 1000  # Convert to ms

        return {
            "pipeline": "ensemble",
            "query": query,
            "k": k,
            "model": model,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
            "tokens_used": tokens_used,
            "llm_raw_response": raw_response,
            "latency_ms": round(total_latency, 2),
            "ensemble_metadata": ensemble_metadata
        }


# Singleton instance
ensemble_service = EnsembleService()
