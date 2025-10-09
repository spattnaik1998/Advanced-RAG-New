"""
RRF Fusion Service - Query expansion with Reciprocal Rank Fusion
"""
import time
from typing import List, Tuple, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.faiss_service import faiss_service
from app.utils.rrf import reciprocal_rank_fusion
from app.models.schemas import RetrievedDoc


class RRFFusionService:
    """Service for RRF Fusion RAG implementation"""

    def __init__(self):
        self.faiss_service = faiss_service

    def generate_query_variants(
        self,
        original_query: str,
        num_variants: int = 4,
        model: str = "gpt-4o-mini"
    ) -> Tuple[List[str], float]:
        """
        Generate multiple query variants for the original query

        Args:
            original_query: User's original query
            num_variants: Number of query variants to generate
            model: OpenAI model name

        Returns:
            Tuple of (query_variants, generation_time_ms)
        """
        start_time = time.time()

        # Create query generation prompt (adapted from rrf.py)
        query_generation_prompt = ChatPromptTemplate(
            input_variables=['original_query'],
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=[],
                        template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                    )
                ),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=['original_query'],
                        template=f'Generate multiple search queries related to: {{question}} \n OUTPUT ({num_variants} queries):'
                    )
                )
            ]
        )

        # Initialize LLM
        llm = ChatOpenAI(model=model, temperature=0.7)

        # Create query generation chain
        generate_queries_chain = (
            query_generation_prompt
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Generate variants
        generated_queries = generate_queries_chain.invoke({"question": original_query})

        # Filter out empty strings and limit to num_variants
        query_variants = [q.strip() for q in generated_queries if q.strip()][:num_variants]

        # If we didn't get enough variants, add the original query
        if len(query_variants) < num_variants:
            query_variants.append(original_query)

        generation_time = (time.time() - start_time) * 1000  # Convert to ms

        return query_variants, generation_time

    def retrieve_for_queries(
        self,
        query_variants: List[str],
        k_per_query: int = 5
    ) -> Tuple[List[List], float]:
        """
        Retrieve documents for each query variant

        Args:
            query_variants: List of query variants
            k_per_query: Number of documents to retrieve per query

        Returns:
            Tuple of (list_of_retrieval_results, total_retrieval_time_ms)
        """
        start_time = time.time()

        # Load FAISS index
        vectorstore = self.faiss_service.load_index()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_per_query})

        # Retrieve for each query variant
        all_results = []
        for query in query_variants:
            docs = retriever.get_relevant_documents(query)
            all_results.append(docs)

        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        return all_results, retrieval_time

    def apply_rrf_fusion(
        self,
        retrieval_results: List[List],
        top_m: int = 5,
        rrf_k: int = 60
    ) -> Tuple[List, Dict]:
        """
        Apply Reciprocal Rank Fusion to combine results

        Args:
            retrieval_results: List of retrieval results from different queries
            top_m: Number of top documents to select after fusion
            rrf_k: Constant for RRF formula

        Returns:
            Tuple of (fused_docs, fusion_metadata)
        """
        start_time = time.time()

        # Apply RRF
        fused_results = reciprocal_rank_fusion(retrieval_results, k=rrf_k)

        # Select top M
        top_fused = fused_results[:top_m]

        # Build metadata
        fusion_metadata = {
            "total_candidates": len(fused_results),
            "top_m_selected": top_m,
            "rrf_k_constant": rrf_k,
            "fusion_scores": [
                {
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "score": float(score)
                }
                for doc, score in top_fused
            ]
        }

        fusion_time = (time.time() - start_time) * 1000  # Convert to ms
        fusion_metadata["fusion_time_ms"] = round(fusion_time, 2)

        return top_fused, fusion_metadata

    def build_context_from_fused(
        self,
        fused_docs: List[tuple]
    ) -> Tuple[str, List[RetrievedDoc]]:
        """
        Build context string from fused documents

        Args:
            fused_docs: List of (document, score) tuples

        Returns:
            Tuple of (context_string, retrieved_docs)
        """
        context_parts = []
        retrieved_docs = []

        for idx, (doc, score) in enumerate(fused_docs, 1):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            source = doc.metadata.get("source", "unknown")
            chunk_text = doc.page_content

            # Add to context with provenance
            context_parts.append(
                f"[Document {idx}] (source: {source}, chunk_id: {chunk_id}, fusion_score: {score:.4f})\n{chunk_text}"
            )

            # Add to retrieved docs
            retrieved_doc = RetrievedDoc(
                chunk_id=chunk_id,
                snippet=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                score=float(score)
            )
            retrieved_docs.append(retrieved_doc)

        context_string = "\n\n".join(context_parts)

        return context_string, retrieved_docs

    def generate_answer(
        self,
        query: str,
        context: str,
        model: str = "gpt-4o-mini"
    ) -> Tuple[str, str, int, float]:
        """
        Generate answer using OpenAI LLM

        Args:
            query: Original user query
            context: Context string from fused chunks
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
        num_query_variants: int = 4,
        k_per_query: int = 5,
        top_m: int = 5,
        rrf_k: int = 60,
        model: str = "gpt-4o-mini"
    ) -> dict:
        """
        Execute full RRF Fusion pipeline

        Args:
            query: User query
            num_query_variants: Number of query variants to generate
            k_per_query: Number of docs to retrieve per query variant
            top_m: Number of top fused docs to use for context
            rrf_k: RRF constant
            model: OpenAI model name

        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()

        # Step 1: Generate query variants
        query_variants, query_gen_time = self.generate_query_variants(
            query, num_query_variants, model
        )

        # Step 2: Retrieve for each query variant
        retrieval_results, retrieval_time = self.retrieve_for_queries(
            query_variants, k_per_query
        )

        # Step 3: Apply RRF fusion
        fused_docs, fusion_metadata = self.apply_rrf_fusion(
            retrieval_results, top_m, rrf_k
        )

        # Step 4: Build context from fused docs
        context, retrieved_docs = self.build_context_from_fused(fused_docs)

        # Step 5: Generate answer
        answer, raw_response, tokens_used, generation_time = self.generate_answer(
            query, context, model
        )

        total_latency = (time.time() - pipeline_start) * 1000  # Convert to ms

        # Build fusion metadata for response
        fusion_metadata_response = {
            "queries_generated": query_variants,
            "num_variants": len(query_variants),
            "k_per_query": k_per_query,
            "fusion_scores": fusion_metadata["fusion_scores"][:top_m]
        }

        return {
            "pipeline": "rrf_fusion",
            "query": query,
            "k": top_m,  # Using top_m as the effective k
            "model": model,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
            "tokens_used": tokens_used,
            "llm_raw_response": raw_response,
            "latency_ms": round(total_latency, 2),
            "fusion_metadata": fusion_metadata_response
        }


# Singleton instance
rrf_fusion_service = RRFFusionService()
