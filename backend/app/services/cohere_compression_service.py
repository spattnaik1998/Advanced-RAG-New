"""
Cohere Compression Service - Contextual compression with Cohere reranking
"""
import time
import os
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from app.services.faiss_service import faiss_service
from app.models.schemas import RetrievedDoc
from app.core.config import settings
from app.models.database import SessionLocal, DocumentChunk


class CohereCompressionService:
    """Service for Cohere Compression RAG with reranking"""

    def __init__(self):
        self.faiss_service = faiss_service

    def check_cohere_api_key(self) -> bool:
        """
        Check if Cohere API key is configured

        Returns:
            True if key is set, False otherwise
        """
        return settings.COHERE_API_KEY is not None and settings.COHERE_API_KEY.strip() != ""

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

    def build_base_retriever(
        self,
        k: int = 5,
        use_ensemble: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        """
        Build base retriever (ensemble or vector only)

        Args:
            k: Number of documents to retrieve
            use_ensemble: If True, use ensemble (FAISS + BM25), else vector only
            vector_weight: Weight for vector retriever in ensemble
            bm25_weight: Weight for BM25 retriever in ensemble

        Returns:
            Base retriever for compression
        """
        # Build FAISS vector retriever
        vectorstore = self.faiss_service.load_index()
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        if not use_ensemble:
            return vector_retriever

        # Build BM25 retriever for ensemble
        documents = self.get_all_chunks_as_documents()
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # Combine into ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )

        return ensemble_retriever

    def build_compression_retriever(
        self,
        k: int = 5,
        top_n: int = 5,
        use_ensemble: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank_model: str = "rerank-english-v3.0"
    ) -> Tuple[ContextualCompressionRetriever, float]:
        """
        Build contextual compression retriever with Cohere reranking

        Args:
            k: Number of documents to retrieve initially
            top_n: Number of top documents after reranking
            use_ensemble: If True, use ensemble base retriever
            vector_weight: Weight for vector retriever
            bm25_weight: Weight for BM25 retriever
            rerank_model: Cohere rerank model name

        Returns:
            Tuple of (compression_retriever, build_time_ms)
        """
        start_time = time.time()

        # Build base retriever
        base_retriever = self.build_base_retriever(
            k=k,
            use_ensemble=use_ensemble,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )

        # Build Cohere reranker compressor
        compressor = CohereRerank(
            model=rerank_model,
            top_n=top_n,
            cohere_api_key=settings.COHERE_API_KEY
        )

        # Build compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        build_time = (time.time() - start_time) * 1000  # Convert to ms

        return compression_retriever, build_time

    def retrieve_compressed(
        self,
        query: str,
        k: int = 5,
        top_n: int = 5,
        use_ensemble: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank_model: str = "rerank-english-v3.0"
    ) -> Tuple[List, float, float]:
        """
        Retrieve compressed and reranked documents

        Args:
            query: User query
            k: Initial retrieval count
            top_n: Top N after reranking
            use_ensemble: Use ensemble retriever
            vector_weight: Vector weight
            bm25_weight: BM25 weight
            rerank_model: Cohere rerank model

        Returns:
            Tuple of (compressed_docs, build_time_ms, retrieval_time_ms)
        """
        # Build compression retriever
        compression_retriever, build_time = self.build_compression_retriever(
            k=k,
            top_n=top_n,
            use_ensemble=use_ensemble,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            rerank_model=rerank_model
        )

        # Retrieve compressed documents
        start_time = time.time()
        compressed_docs = compression_retriever.get_relevant_documents(query)
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        return compressed_docs, build_time, retrieval_time

    def build_context(
        self,
        query: str,
        k: int = 5,
        top_n: int = 5,
        use_ensemble: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank_model: str = "rerank-english-v3.0"
    ) -> Tuple[str, List[RetrievedDoc], dict, float]:
        """
        Build context from compressed and reranked documents

        Args:
            query: User query
            k: Initial retrieval count
            top_n: Top N after reranking
            use_ensemble: Use ensemble retriever
            vector_weight: Vector weight
            bm25_weight: BM25 weight
            rerank_model: Cohere rerank model

        Returns:
            Tuple of (context_string, retrieved_docs, compression_metadata, total_time_ms)
        """
        # Retrieve compressed docs
        compressed_docs, build_time, retrieval_time = self.retrieve_compressed(
            query, k, top_n, use_ensemble, vector_weight, bm25_weight, rerank_model
        )

        # Build context with provenance
        context_parts = []
        retrieved_docs = []

        for idx, doc in enumerate(compressed_docs, 1):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            source = doc.metadata.get("source", "unknown")
            chunk_text = doc.page_content

            # Check if document has relevance_score from Cohere
            relevance_score = doc.metadata.get("relevance_score", 1.0 / (idx + 1))

            # Add to context with provenance
            context_parts.append(
                f"[Document {idx}] (source: {source}, chunk_id: {chunk_id}, relevance_score: {relevance_score:.4f})\n{chunk_text}"
            )

            # Add to retrieved docs
            retrieved_doc = RetrievedDoc(
                chunk_id=chunk_id,
                snippet=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                score=float(relevance_score)
            )
            retrieved_docs.append(retrieved_doc)

        context_string = "\n\n".join(context_parts)

        # Build compression metadata
        compression_metadata = {
            "initial_k": k,
            "reranked_top_n": top_n,
            "use_ensemble": use_ensemble,
            "vector_weight": vector_weight if use_ensemble else 1.0,
            "bm25_weight": bm25_weight if use_ensemble else 0.0,
            "rerank_model": rerank_model,
            "build_time_ms": round(build_time, 2),
            "retrieval_time_ms": round(retrieval_time, 2),
            "docs_returned": len(compressed_docs)
        }

        total_time = build_time + retrieval_time

        return context_string, retrieved_docs, compression_metadata, total_time

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
            context: Context string from compressed chunks
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
        top_n: int = 5,
        use_ensemble: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rerank_model: str = "rerank-english-v3.0",
        model: str = "gpt-4o-mini"
    ) -> dict:
        """
        Execute full Cohere Compression RAG pipeline

        Args:
            query: User query
            k: Initial retrieval count
            top_n: Top N after reranking
            use_ensemble: Use ensemble retriever
            vector_weight: Vector weight
            bm25_weight: BM25 weight
            rerank_model: Cohere rerank model
            model: OpenAI model name

        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()

        # Step 1: Build context with compression and reranking
        context, retrieved_docs, compression_metadata, retrieval_total_time = self.build_context(
            query, k, top_n, use_ensemble, vector_weight, bm25_weight, rerank_model
        )

        # Step 2: Generate answer
        answer, raw_response, tokens_used, generation_time = self.generate_answer(
            query, context, model
        )

        total_latency = (time.time() - pipeline_start) * 1000  # Convert to ms

        return {
            "pipeline": "cohere_compression",
            "query": query,
            "k": top_n,  # Return top_n as effective k
            "model": model,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
            "tokens_used": tokens_used,
            "llm_raw_response": raw_response,
            "latency_ms": round(total_latency, 2),
            "compression_metadata": compression_metadata
        }


# Singleton instance
cohere_compression_service = CohereCompressionService()
