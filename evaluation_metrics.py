import numpy as np
from typing import Dict, List
import os
from openai import OpenAI
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


def calculate_cosine_similarity(answer: str, ground_truth: str, openai_api_key: str) -> float:
    """
    Calculate cosine similarity between answer and ground truth using embeddings.

    Args:
        answer: Generated answer
        ground_truth: Reference answer
        openai_api_key: OpenAI API key for embeddings

    Returns:
        Cosine similarity score (0-1)
    """
    try:
        client = OpenAI(api_key=openai_api_key)

        # Get embeddings
        response1 = client.embeddings.create(
            input=answer,
            model="text-embedding-3-large"
        )
        embedding1 = response1.data[0].embedding

        response2 = client.embeddings.create(
            input=ground_truth,
            model="text-embedding-3-large"
        )
        embedding2 = response2.data[0].embedding

        # Calculate cosine similarity
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return float(similarity)

    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0


def calculate_rouge_scores(answer: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between answer and ground truth.

    Args:
        answer: Generated answer
        ground_truth: Reference answer

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, answer)

        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def calculate_bleu_score(answer: str, ground_truth: str) -> float:
    """
    Calculate BLEU score between answer and ground truth.

    Args:
        answer: Generated answer
        ground_truth: Reference answer

    Returns:
        BLEU score (0-1)
    """
    try:
        # Tokenize
        reference = [ground_truth.lower().split()]
        candidate = answer.lower().split()

        # Use smoothing function to avoid zero scores
        smoothing = SmoothingFunction().method1

        # Calculate BLEU score
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing)

        return float(bleu)

    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0


def calculate_ragas_metrics(question: str, answer: str, ground_truth: str,
                           context: str, openai_api_key: str) -> Dict[str, float]:
    """
    Calculate RAGAS-inspired metrics for RAG evaluation.

    RAGAS metrics include:
    - Faithfulness: Whether the answer is grounded in the context
    - Answer Relevancy: How relevant the answer is to the question
    - Context Precision: Quality of retrieved context
    - Context Recall: Whether all necessary info is in context

    Args:
        question: Input question
        answer: Generated answer
        ground_truth: Reference answer
        context: Retrieved context
        openai_api_key: OpenAI API key

    Returns:
        Dictionary with RAGAS metrics
    """
    try:
        client = OpenAI(api_key=openai_api_key)

        # 1. Faithfulness: Check if answer is grounded in context
        faithfulness_prompt = f"""Given the following context and answer, determine if the answer is faithfully grounded in the context.
Rate the faithfulness on a scale of 0.0 to 1.0, where:
- 1.0 means the answer is completely supported by the context
- 0.0 means the answer has no support in the context

Context: {context}

Answer: {answer}

Provide only a numeric score between 0.0 and 1.0."""

        faithfulness_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": faithfulness_prompt}],
            temperature=0.0
        )
        faithfulness_text = faithfulness_response.choices[0].message.content.strip()
        faithfulness = extract_numeric_score(faithfulness_text)

        # 2. Answer Relevancy: How relevant is the answer to the question
        relevancy_prompt = f"""Given the following question and answer, determine how relevant the answer is to the question.
Rate the relevancy on a scale of 0.0 to 1.0, where:
- 1.0 means the answer is highly relevant and directly addresses the question
- 0.0 means the answer is completely irrelevant

Question: {question}

Answer: {answer}

Provide only a numeric score between 0.0 and 1.0."""

        relevancy_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": relevancy_prompt}],
            temperature=0.0
        )
        relevancy_text = relevancy_response.choices[0].message.content.strip()
        answer_relevancy = extract_numeric_score(relevancy_text)

        # 3. Context Precision: Quality of the retrieved context
        context_precision_prompt = f"""Given the following question and context, determine how precise and relevant the context is for answering the question.
Rate the context precision on a scale of 0.0 to 1.0, where:
- 1.0 means the context is highly relevant with no irrelevant information
- 0.0 means the context is completely irrelevant

Question: {question}

Context: {context}

Provide only a numeric score between 0.0 and 1.0."""

        context_precision_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": context_precision_prompt}],
            temperature=0.0
        )
        precision_text = context_precision_response.choices[0].message.content.strip()
        context_precision = extract_numeric_score(precision_text)

        # 4. Context Recall: Does context contain all necessary information
        context_recall_prompt = f"""Given the following question, ground truth answer, and retrieved context, determine if the context contains all the information needed to generate the ground truth answer.
Rate the context recall on a scale of 0.0 to 1.0, where:
- 1.0 means all information from ground truth is present in context
- 0.0 means none of the ground truth information is in context

Question: {question}

Ground Truth: {ground_truth}

Context: {context}

Provide only a numeric score between 0.0 and 1.0."""

        context_recall_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": context_recall_prompt}],
            temperature=0.0
        )
        recall_text = context_recall_response.choices[0].message.content.strip()
        context_recall = extract_numeric_score(recall_text)

        # 5. Answer Correctness: Compare to ground truth
        correctness_prompt = f"""Given the following question, generated answer, and ground truth answer, determine how correct the generated answer is compared to the ground truth.
Rate the correctness on a scale of 0.0 to 1.0, where:
- 1.0 means the answer is completely correct and matches the ground truth
- 0.0 means the answer is completely incorrect

Question: {question}

Generated Answer: {answer}

Ground Truth: {ground_truth}

Provide only a numeric score between 0.0 and 1.0."""

        correctness_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": correctness_prompt}],
            temperature=0.0
        )
        correctness_text = correctness_response.choices[0].message.content.strip()
        answer_correctness = extract_numeric_score(correctness_text)

        return {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'answer_correctness': answer_correctness
        }

    except Exception as e:
        print(f"Error calculating RAGAS metrics: {e}")
        return {
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'answer_correctness': 0.0
        }


def extract_numeric_score(text: str) -> float:
    """
    Extract a numeric score from text response.

    Args:
        text: Text containing a score

    Returns:
        Extracted score as float (0-1)
    """
    try:
        # Try to find a number between 0 and 1
        import re
        numbers = re.findall(r'0?\.\d+|1\.0+|1|0', text)
        if numbers:
            score = float(numbers[0])
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
        return 0.5  # Default to middle if no number found
    except:
        return 0.5


def calculate_all_metrics(question: str, answer: str, ground_truth: str,
                         context: str, openai_api_key: str) -> Dict[str, any]:
    """
    Calculate all evaluation metrics at once.

    Args:
        question: Input question
        answer: Generated answer
        ground_truth: Reference answer
        context: Retrieved context
        openai_api_key: OpenAI API key

    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'cosine_similarity': calculate_cosine_similarity(answer, ground_truth, openai_api_key),
        'rouge_scores': calculate_rouge_scores(answer, ground_truth),
        'bleu_score': calculate_bleu_score(answer, ground_truth),
        'ragas_metrics': calculate_ragas_metrics(question, answer, ground_truth, context, openai_api_key)
    }

    return metrics


def compare_rag_approaches(results: List[Dict]) -> Dict:
    """
    Compare multiple RAG approaches based on their results.

    Args:
        results: List of result dictionaries from different RAG approaches

    Returns:
        Comparison summary with average scores
    """
    summary = {}

    for approach_name, approach_results in results.items():
        avg_cosine = np.mean([r['metrics']['cosine_similarity'] for r in approach_results])
        avg_rouge_l = np.mean([r['metrics']['rouge_scores']['rougeL'] for r in approach_results])
        avg_bleu = np.mean([r['metrics']['bleu_score'] for r in approach_results])

        if 'ragas_metrics' in approach_results[0]['metrics']:
            avg_faithfulness = np.mean([r['metrics']['ragas_metrics']['faithfulness'] for r in approach_results])
            avg_relevancy = np.mean([r['metrics']['ragas_metrics']['answer_relevancy'] for r in approach_results])
        else:
            avg_faithfulness = 0.0
            avg_relevancy = 0.0

        summary[approach_name] = {
            'avg_cosine_similarity': avg_cosine,
            'avg_rouge_l': avg_rouge_l,
            'avg_bleu': avg_bleu,
            'avg_faithfulness': avg_faithfulness,
            'avg_answer_relevancy': avg_relevancy
        }

    return summary
