"""
Evaluation metrics for RAG pipeline comparison
"""
import re
from typing import List, Dict, Set
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings


class EvaluationMetrics:
    """Class for computing RAG evaluation metrics"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=settings.OPENAI_API_KEY
        )

    def compute_answer_similarity(self, answer1: str, answer2: str) -> float:
        """
        Compute cosine similarity between two answers using embeddings

        Args:
            answer1: First answer
            answer2: Second answer

        Returns:
            Cosine similarity score (0-1)
        """
        # Get embeddings
        emb1 = self.embeddings.embed_query(answer1)
        emb2 = self.embeddings.embed_query(answer2)

        # Compute cosine similarity
        similarity = cosine_similarity(
            np.array(emb1).reshape(1, -1),
            np.array(emb2).reshape(1, -1)
        )[0][0]

        return float(similarity)

    def compute_pairwise_similarities(self, answers: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise similarities between all answers

        Args:
            answers: Dictionary of {pipeline_name: answer}

        Returns:
            Dictionary of pairwise similarities
        """
        pipelines = list(answers.keys())
        similarities = {}

        for i, pipeline1 in enumerate(pipelines):
            similarities[pipeline1] = {}
            for pipeline2 in pipelines[i:]:
                if pipeline1 == pipeline2:
                    similarities[pipeline1][pipeline2] = 1.0
                else:
                    sim = self.compute_answer_similarity(
                        answers[pipeline1],
                        answers[pipeline2]
                    )
                    similarities[pipeline1][pipeline2] = sim
                    # Mirror the similarity
                    if pipeline2 not in similarities:
                        similarities[pipeline2] = {}
                    similarities[pipeline2][pipeline1] = sim

        return similarities

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Simple sentence tokenization

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def compute_grounding_percentage(self, answer: str, context: str) -> float:
        """
        Compute percentage of answer sentences grounded in context

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Grounding percentage (0-1)
        """
        answer_sentences = self.sentence_tokenize(answer)
        if not answer_sentences:
            return 0.0

        context_lower = context.lower()
        grounded_count = 0

        for sentence in answer_sentences:
            sentence_lower = sentence.lower()
            # Check if sentence or significant parts are in context
            words = sentence_lower.split()

            # Skip very short sentences
            if len(words) < 3:
                continue

            # Check for n-gram overlap (3-grams and higher)
            max_ngram_match = 0
            for n in range(3, min(len(words) + 1, 8)):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    if ngram in context_lower:
                        max_ngram_match = max(max_ngram_match, n)

            # Consider grounded if significant n-gram match
            if max_ngram_match >= 3:
                grounded_count += 1

        return grounded_count / len(answer_sentences) if answer_sentences else 0.0

    def compute_retrieval_overlap(
        self,
        retrieved_docs_by_pipeline: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute retrieval overlap between pipelines

        Args:
            retrieved_docs_by_pipeline: {pipeline_name: [chunk_ids]}

        Returns:
            Dictionary of pairwise Jaccard similarities
        """
        pipelines = list(retrieved_docs_by_pipeline.keys())
        overlaps = {}

        for i, pipeline1 in enumerate(pipelines):
            overlaps[pipeline1] = {}
            set1 = set(retrieved_docs_by_pipeline[pipeline1])

            for pipeline2 in pipelines[i:]:
                set2 = set(retrieved_docs_by_pipeline[pipeline2])

                if pipeline1 == pipeline2:
                    overlaps[pipeline1][pipeline2] = 1.0
                else:
                    # Compute Jaccard similarity
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard = intersection / union if union > 0 else 0.0

                    overlaps[pipeline1][pipeline2] = jaccard

                    # Mirror the overlap
                    if pipeline2 not in overlaps:
                        overlaps[pipeline2] = {}
                    overlaps[pipeline2][pipeline1] = jaccard

        return overlaps

    def compute_all_metrics(
        self,
        results: Dict[str, dict],
        contexts: Dict[str, str]
    ) -> Dict:
        """
        Compute all evaluation metrics

        Args:
            results: Dictionary of {pipeline_name: result_dict}
            contexts: Dictionary of {pipeline_name: context_string}

        Returns:
            Dictionary with all metrics
        """
        # Extract answers
        answers = {name: res["answer"] for name, res in results.items()}

        # Extract retrieved doc IDs
        retrieved_docs = {
            name: [doc["chunk_id"] for doc in res["retrieved_docs"]]
            for name, res in results.items()
        }

        # Extract latencies
        latencies = {name: res["latency_ms"] for name, res in results.items()}

        # Extract token usage
        tokens = {name: res.get("tokens_used", 0) for name, res in results.items()}

        # Compute answer similarities
        answer_similarities = self.compute_pairwise_similarities(answers)

        # Compute grounding percentages
        grounding = {
            name: self.compute_grounding_percentage(answers[name], contexts[name])
            for name in results.keys()
        }

        # Compute retrieval overlaps
        retrieval_overlap = self.compute_retrieval_overlap(retrieved_docs)

        return {
            "answer_similarities": answer_similarities,
            "grounding_percentages": grounding,
            "retrieval_overlaps": retrieval_overlap,
            "latencies_ms": latencies,
            "token_usage": tokens
        }


# Singleton instance
evaluation_metrics = EvaluationMetrics()
