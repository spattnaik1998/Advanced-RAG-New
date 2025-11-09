import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.load import dumps, loads


class NaiveRAG:
    """
    Naive RAG implementation using simple vector store retrieval.
    This is the baseline approach with semantic search only.
    """

    def __init__(self, openai_api_key: str, chunk_size: int = 500,
                 chunk_overlap: int = 50, top_k: int = 5):
        self.openai_api_key = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

    def load_and_process_document(self, pdf_path: str):
        """Load and process the PDF document"""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        text_splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstore = FAISS.from_documents(text_splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        return len(text_splits)

    def get_context(self, question: str) -> str:
        """Get relevant context for a question"""
        docs = self.retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in docs])

    def answer_question(self, question: str, model_name: str = "gpt-4o-mini") -> str:
        """Answer a question using naive RAG"""
        # Create LLM
        llm = ChatOpenAI(model=model_name, temperature=0.3)

        # Create prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        # Create RAG chain
        chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | output_parser
        )

        # Get answer
        response = chain.invoke(question)
        return response


class CohereRerankingRAG:
    """
    Hybrid search with Cohere reranking implementation.
    Combines BM25 (keyword search) with vector search and reranks using Cohere.
    """

    def __init__(self, openai_api_key: str, cohere_api_key: str,
                 chunk_size: int = 500, chunk_overlap: int = 50, top_k: int = 5):
        self.openai_api_key = openai_api_key
        self.cohere_api_key = cohere_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        os.environ['COHERE_API_KEY'] = cohere_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embeddings = None
        self.vectorstore = None
        self.ensemble_retriever = None
        self.compression_retriever = None
        self.text_splits = None

    def load_and_process_document(self, pdf_path: str):
        """Load and process the PDF document"""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.text_splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstore = FAISS.from_documents(self.text_splits, self.embeddings)

        # Create vector retriever
        retriever_vectordb = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # Create BM25 retriever (keyword-based)
        keyword_retriever = BM25Retriever.from_documents(self.text_splits)
        keyword_retriever.k = self.top_k

        # Create ensemble retriever (combines both)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vectordb, keyword_retriever],
            weights=[0.5, 0.5]
        )

        # Create Cohere reranker
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=self.top_k)

        # Create compression retriever with reranker
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.ensemble_retriever
        )

        return len(self.text_splits)

    def get_context(self, question: str) -> str:
        """Get relevant context for a question"""
        docs = self.compression_retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in docs])

    def answer_question(self, question: str, model_name: str = "gpt-4o-mini") -> str:
        """Answer a question using hybrid search with Cohere reranking"""
        # Create LLM
        llm = ChatOpenAI(model=model_name, temperature=0.3)

        # Create prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        # Create RAG chain with reranking
        def get_context(query):
            docs = self.compression_retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context

        chain = (
            {
                "context": get_context,
                "query": RunnablePassthrough()
            }
            | prompt
            | llm
            | output_parser
        )

        # Get answer
        response = chain.invoke(question)
        return response


class RecipRocalRankFusionRAG:
    """
    RAG Fusion implementation using Reciprocal Rank Fusion.
    Generates multiple queries and fuses results using reciprocal rank scoring.
    """

    def __init__(self, openai_api_key: str, chunk_size: int = 500,
                 chunk_overlap: int = 50, top_k: int = 5):
        self.openai_api_key = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

    def load_and_process_document(self, pdf_path: str):
        """Load and process the PDF document"""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        text_splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstore = FAISS.from_documents(text_splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        return len(text_splits)

    def reciprocal_rank_fusion(self, results: list, k: int = 60):
        """
        Reciprocal Rank Fusion algorithm for combining multiple retrieval results
        """
        fused_scores = {}

        for docs in results:
            # Assumes docs are returned in sorted order of relevance
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort by score in descending order
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results

    def get_context(self, question: str) -> str:
        """Get relevant context for a question using RRF"""
        # Generate multiple queries
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

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
                        template='Generate 4 different search queries related to: {question}\nOUTPUT (4 queries):'
                    )
                )
            ]
        )

        generate_queries = (
            query_generation_prompt
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        # Generate queries
        generated_queries = generate_queries.invoke({"question": question})
        # Filter empty queries
        generated_queries = [q.strip() for q in generated_queries if q.strip()]

        # Retrieve documents for each query
        results = []
        for query in generated_queries[:4]:  # Limit to 4 queries
            try:
                docs = self.retriever.invoke(query)
                results.append(docs)
            except:
                continue

        # Apply RRF
        if results:
            fused_results = self.reciprocal_rank_fusion(results)
            # Get top documents
            top_docs = [doc for doc, score in fused_results[:self.top_k]]
            return "\n\n".join([doc.page_content for doc in top_docs])
        else:
            # Fallback to original query
            docs = self.retriever.invoke(question)
            return "\n\n".join([doc.page_content for doc in docs])

    def answer_question(self, question: str, model_name: str = "gpt-4o-mini") -> str:
        """Answer a question using RAG Fusion with RRF"""
        # Create LLM
        llm = ChatOpenAI(model=model_name, temperature=0.3)

        # Create prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        # Create RAG Fusion chain
        def get_context_for_chain(question):
            return self.get_context(question)

        chain = (
            {
                "context": get_context_for_chain,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | output_parser
        )

        # Get answer
        response = chain.invoke(question)
        return response
