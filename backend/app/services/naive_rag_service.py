"""
Naive RAG Service - Simple retrieval and generation pipeline
"""
import time
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.faiss_service import faiss_service
from app.models.schemas import RetrievedDoc


class NaiveRAGService:
    """Service for Naive RAG implementation"""

    def __init__(self):
        self.faiss_service = faiss_service

    def retrieve_chunks(self, query: str, k: int = 5) -> Tuple[List[RetrievedDoc], float]:
        """
        Retrieve top-k chunks from FAISS index

        Args:
            query: User query
            k: Number of chunks to retrieve

        Returns:
            Tuple of (retrieved_docs, retrieval_time_ms)
        """
        start_time = time.time()

        # Load FAISS index
        vectorstore = self.faiss_service.load_index()

        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k=k)

        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        # Format results
        retrieved_docs = []
        for doc, score in results:
            retrieved_doc = RetrievedDoc(
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                score=float(score)
            )
            retrieved_docs.append(retrieved_doc)

        return retrieved_docs, retrieval_time

    def build_context(self, query: str, k: int = 5) -> Tuple[str, List[RetrievedDoc], float]:
        """
        Build context string from retrieved chunks

        Args:
            query: User query
            k: Number of chunks to retrieve

        Returns:
            Tuple of (context_string, retrieved_docs, retrieval_time_ms)
        """
        # Load FAISS index
        vectorstore = self.faiss_service.load_index()

        start_time = time.time()

        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k=k)

        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        # Build context with provenance
        context_parts = []
        retrieved_docs = []

        for idx, (doc, score) in enumerate(results, 1):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            source = doc.metadata.get("source", "unknown")
            chunk_text = doc.page_content

            # Add to context with provenance
            context_parts.append(
                f"[Document {idx}] (source: {source}, chunk_id: {chunk_id})\n{chunk_text}"
            )

            # Add to retrieved docs
            retrieved_doc = RetrievedDoc(
                chunk_id=chunk_id,
                snippet=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                score=float(score)
            )
            retrieved_docs.append(retrieved_doc)

        context_string = "\n\n".join(context_parts)

        return context_string, retrieved_docs, retrieval_time

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

        # Create prompt template
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

        # For now, we'll use the answer as raw_response and estimate tokens
        # Note: OpenAI's response doesn't directly provide token count in LangChain's StrOutputParser
        tokens_used = len(answer.split()) * 2  # Rough estimate (1 token ≈ 0.5 words)

        return answer, answer, tokens_used, generation_time

    def query(
        self,
        query: str,
        k: int = 5,
        model: str = "gpt-4o-mini"
    ) -> dict:
        """
        Execute full Naive RAG pipeline

        Args:
            query: User query
            k: Number of chunks to retrieve
            model: OpenAI model name

        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()

        # Step 1: Build context and retrieve docs
        context, retrieved_docs, retrieval_time = self.build_context(query, k)

        # Step 2: Generate answer
        answer, raw_response, tokens_used, generation_time = self.generate_answer(
            query, context, model
        )

        total_latency = (time.time() - pipeline_start) * 1000  # Convert to ms

        return {
            "pipeline": "naive_rag",
            "query": query,
            "k": k,
            "model": model,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
            "tokens_used": tokens_used,
            "llm_raw_response": raw_response,
            "latency_ms": round(total_latency, 2)
        }


# Singleton instance
naive_rag_service = NaiveRAGService()
