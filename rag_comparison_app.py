import streamlit as st
import json
import os
from rag_approaches import (
    NaiveRAG,
    CohereRerankingRAG,
    RecipRocalRankFusionRAG
)
from evaluation_metrics import (
    calculate_cosine_similarity,
    calculate_rouge_scores,
    calculate_bleu_score,
    calculate_ragas_metrics
)
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="RAG Approaches Comparison",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 RAG Approaches Comparison Tool")
st.markdown("""
Compare three different RAG (Retrieval Augmented Generation) approaches on the CogTale paper:
1. **Naive RAG**: Simple vector store retrieval with semantic search
2. **Cohere Reranking**: Hybrid search (BM25 + Vector) with Cohere reranker
3. **Reciprocal Rank Fusion (RRF)**: Query generation with reciprocal rank fusion
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Keys input
st.sidebar.subheader("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password",
                                        value=os.getenv("OPENAI_API_KEY", ""))
cohere_api_key = st.sidebar.text_input("Cohere API Key", type="password",
                                        value=os.getenv("COHERE_API_KEY", ""))

# Model selection
model_name = st.sidebar.selectbox(
    "Select LLM Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
)

# Retrieval parameters
st.sidebar.subheader("Retrieval Parameters")
chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)
top_k = st.sidebar.slider("Top K Documents", 3, 10, 5)

# Initialize session state
if 'rag_systems_initialized' not in st.session_state:
    st.session_state.rag_systems_initialized = False
    st.session_state.results = []

# Function to initialize RAG systems
@st.cache_resource
def initialize_rag_systems(_openai_key, _cohere_key, _chunk_size, _chunk_overlap, _top_k):
    """Initialize all three RAG systems"""
    pdf_path = "paper_new_1.pdf"

    naive_rag = NaiveRAG(
        openai_api_key=_openai_key,
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        top_k=_top_k
    )
    naive_rag.load_and_process_document(pdf_path)

    cohere_rag = CohereRerankingRAG(
        openai_api_key=_openai_key,
        cohere_api_key=_cohere_key,
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        top_k=_top_k
    )
    cohere_rag.load_and_process_document(pdf_path)

    rrf_rag = RecipRocalRankFusionRAG(
        openai_api_key=_openai_key,
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        top_k=_top_k
    )
    rrf_rag.load_and_process_document(pdf_path)

    return naive_rag, cohere_rag, rrf_rag

# Load questions from JSON
@st.cache_data
def load_questions():
    """Load questions from the JSON file"""
    with open("cogtale_qa_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["qa_pairs"]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Question Selection")

    if openai_api_key:
        try:
            qa_pairs = load_questions()

            # Group questions by category
            categories = list(set([qa["category"] for qa in qa_pairs]))
            selected_category = st.selectbox("Select Category", ["All"] + sorted(categories))

            # Filter questions by category
            if selected_category == "All":
                filtered_questions = qa_pairs
            else:
                filtered_questions = [qa for qa in qa_pairs if qa["category"] == selected_category]

            # Select specific questions
            question_options = [f"{i+1}. {qa['question'][:80]}..."
                              for i, qa in enumerate(filtered_questions)]

            # Option to select all questions
            select_all = st.checkbox("Select All Questions", value=False)

            if select_all:
                selected_questions_idx = list(range(len(filtered_questions)))
                st.info(f"✅ All {len(filtered_questions)} questions selected for evaluation")
            else:
                selected_questions_idx = st.multiselect(
                    "Select Questions to Evaluate",
                    range(len(filtered_questions)),
                    format_func=lambda x: question_options[x],
                    default=list(range(min(3, len(filtered_questions))))
                )

            st.caption(f"📊 Total questions available: {len(filtered_questions)} | Selected: {len(selected_questions_idx)}")

        except FileNotFoundError:
            st.error("❌ cogtale_qa_dataset.json not found!")
            selected_questions_idx = []
    else:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        selected_questions_idx = []

with col2:
    st.header("🚀 Actions")

    if st.button("Initialize RAG Systems", type="primary"):
        if not openai_api_key:
            st.error("❌ OpenAI API key is required!")
        elif not cohere_api_key:
            st.error("❌ Cohere API key is required for reranking!")
        else:
            with st.spinner("Initializing RAG systems..."):
                try:
                    naive_rag, cohere_rag, rrf_rag = initialize_rag_systems(
                        openai_api_key, cohere_api_key,
                        chunk_size, chunk_overlap, top_k
                    )
                    st.session_state.naive_rag = naive_rag
                    st.session_state.cohere_rag = cohere_rag
                    st.session_state.rrf_rag = rrf_rag
                    st.session_state.rag_systems_initialized = True
                    st.success("✅ RAG systems initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing RAG systems: {str(e)}")

    if st.session_state.rag_systems_initialized and selected_questions_idx:
        if st.button("Run Comparison", type="secondary"):
            results = []
            progress_bar = st.progress(0)

            for idx, question_idx in enumerate(selected_questions_idx):
                qa = filtered_questions[question_idx]
                question = qa["question"]
                ground_truth = qa["answer"]

                # Get answers from all three approaches
                with st.spinner(f"Processing question {idx+1}/{len(selected_questions_idx)}..."):
                    try:
                        naive_answer = st.session_state.naive_rag.answer_question(
                            question, model_name
                        )
                        cohere_answer = st.session_state.cohere_rag.answer_question(
                            question, model_name
                        )
                        rrf_answer = st.session_state.rrf_rag.answer_question(
                            question, model_name
                        )

                        # Calculate metrics for each approach
                        naive_metrics = {
                            "cosine": calculate_cosine_similarity(naive_answer, ground_truth, openai_api_key),
                            "rouge": calculate_rouge_scores(naive_answer, ground_truth),
                            "bleu": calculate_bleu_score(naive_answer, ground_truth)
                        }

                        cohere_metrics = {
                            "cosine": calculate_cosine_similarity(cohere_answer, ground_truth, openai_api_key),
                            "rouge": calculate_rouge_scores(cohere_answer, ground_truth),
                            "bleu": calculate_bleu_score(cohere_answer, ground_truth)
                        }

                        rrf_metrics = {
                            "cosine": calculate_cosine_similarity(rrf_answer, ground_truth, openai_api_key),
                            "rouge": calculate_rouge_scores(rrf_answer, ground_truth),
                            "bleu": calculate_bleu_score(rrf_answer, ground_truth)
                        }

                        results.append({
                            "question": question,
                            "ground_truth": ground_truth,
                            "category": qa["category"],
                            "naive_answer": naive_answer,
                            "cohere_answer": cohere_answer,
                            "rrf_answer": rrf_answer,
                            "naive_metrics": naive_metrics,
                            "cohere_metrics": cohere_metrics,
                            "rrf_metrics": rrf_metrics
                        })

                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

                progress_bar.progress((idx + 1) / len(selected_questions_idx))

            st.session_state.results = results
            st.success("✅ Comparison completed!")

# Display results
if st.session_state.results:
    st.header("📊 Results & Comparison")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Detailed Results", "📊 Metrics Comparison",
                                       "🎯 RAGAS Analysis", "📈 Summary Statistics"])

    with tab1:
        st.subheader("Detailed Question-by-Question Results")

        for idx, result in enumerate(st.session_state.results):
            with st.expander(f"Question {idx+1}: {result['question'][:100]}..."):
                st.markdown(f"**Category:** {result['category']}")
                st.markdown(f"**Question:** {result['question']}")
                st.markdown(f"**Ground Truth Answer:**")
                st.info(result['ground_truth'])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**🔵 Naive RAG**")
                    st.write(result['naive_answer'])
                    st.caption(f"Cosine: {result['naive_metrics']['cosine']:.3f} | "
                             f"ROUGE-L: {result['naive_metrics']['rouge']['rougeL']:.3f} | "
                             f"BLEU: {result['naive_metrics']['bleu']:.3f}")

                with col2:
                    st.markdown("**🟢 Cohere Reranking**")
                    st.write(result['cohere_answer'])
                    st.caption(f"Cosine: {result['cohere_metrics']['cosine']:.3f} | "
                             f"ROUGE-L: {result['cohere_metrics']['rouge']['rougeL']:.3f} | "
                             f"BLEU: {result['cohere_metrics']['bleu']:.3f}")

                with col3:
                    st.markdown("**🟣 RRF**")
                    st.write(result['rrf_answer'])
                    st.caption(f"Cosine: {result['rrf_metrics']['cosine']:.3f} | "
                             f"ROUGE-L: {result['rrf_metrics']['rouge']['rougeL']:.3f} | "
                             f"BLEU: {result['rrf_metrics']['bleu']:.3f}")

    with tab2:
        st.subheader("Metrics Comparison Across Approaches")

        # Aggregate metrics
        metrics_data = []
        for result in st.session_state.results:
            metrics_data.append({
                "Question": result['question'][:50] + "...",
                "Naive_Cosine": result['naive_metrics']['cosine'],
                "Cohere_Cosine": result['cohere_metrics']['cosine'],
                "RRF_Cosine": result['rrf_metrics']['cosine'],
                "Naive_ROUGE": result['naive_metrics']['rouge']['rougeL'],
                "Cohere_ROUGE": result['cohere_metrics']['rouge']['rougeL'],
                "RRF_ROUGE": result['rrf_metrics']['rouge']['rougeL'],
                "Naive_BLEU": result['naive_metrics']['bleu'],
                "Cohere_BLEU": result['cohere_metrics']['bleu'],
                "RRF_BLEU": result['rrf_metrics']['bleu']
            })

        df_metrics = pd.DataFrame(metrics_data)

        # Bar chart for average metrics
        avg_metrics = {
            "Naive RAG": [
                df_metrics['Naive_Cosine'].mean(),
                df_metrics['Naive_ROUGE'].mean(),
                df_metrics['Naive_BLEU'].mean()
            ],
            "Cohere Reranking": [
                df_metrics['Cohere_Cosine'].mean(),
                df_metrics['Cohere_ROUGE'].mean(),
                df_metrics['Cohere_BLEU'].mean()
            ],
            "RRF": [
                df_metrics['RRF_Cosine'].mean(),
                df_metrics['RRF_ROUGE'].mean(),
                df_metrics['RRF_BLEU'].mean()
            ]
        }

        fig = go.Figure(data=[
            go.Bar(name='Naive RAG', x=['Cosine Similarity', 'ROUGE-L', 'BLEU'],
                   y=avg_metrics['Naive RAG'], marker_color='blue'),
            go.Bar(name='Cohere Reranking', x=['Cosine Similarity', 'ROUGE-L', 'BLEU'],
                   y=avg_metrics['Cohere Reranking'], marker_color='green'),
            go.Bar(name='RRF', x=['Cosine Similarity', 'ROUGE-L', 'BLEU'],
                   y=avg_metrics['RRF'], marker_color='purple')
        ])
        fig.update_layout(title='Average Metrics Comparison',
                         barmode='group',
                         yaxis_title='Score')
        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics table
        st.dataframe(df_metrics, use_container_width=True)

    with tab3:
        st.subheader("RAGAS Metrics Analysis")

        if openai_api_key and st.button("Calculate RAGAS Metrics"):
            with st.spinner("Calculating RAGAS metrics (this may take a while)..."):
                try:
                    ragas_results = []
                    for result in st.session_state.results:
                        # Calculate RAGAS for each approach
                        naive_ragas = calculate_ragas_metrics(
                            result['question'],
                            result['naive_answer'],
                            result['ground_truth'],
                            st.session_state.naive_rag.get_context(result['question']),
                            openai_api_key
                        )

                        cohere_ragas = calculate_ragas_metrics(
                            result['question'],
                            result['cohere_answer'],
                            result['ground_truth'],
                            st.session_state.cohere_rag.get_context(result['question']),
                            openai_api_key
                        )

                        rrf_ragas = calculate_ragas_metrics(
                            result['question'],
                            result['rrf_answer'],
                            result['ground_truth'],
                            st.session_state.rrf_rag.get_context(result['question']),
                            openai_api_key
                        )

                        ragas_results.append({
                            "question": result['question'],
                            "naive": naive_ragas,
                            "cohere": cohere_ragas,
                            "rrf": rrf_ragas
                        })

                    st.session_state.ragas_results = ragas_results
                    st.success("✅ RAGAS metrics calculated!")

                except Exception as e:
                    st.error(f"Error calculating RAGAS metrics: {str(e)}")

        if 'ragas_results' in st.session_state:
            # Display RAGAS results
            ragas_df = []
            for r in st.session_state.ragas_results:
                ragas_df.append({
                    "Question": r['question'][:50] + "...",
                    "Naive_Faithfulness": r['naive'].get('faithfulness', 0),
                    "Cohere_Faithfulness": r['cohere'].get('faithfulness', 0),
                    "RRF_Faithfulness": r['rrf'].get('faithfulness', 0),
                    "Naive_Relevancy": r['naive'].get('answer_relevancy', 0),
                    "Cohere_Relevancy": r['cohere'].get('answer_relevancy', 0),
                    "RRF_Relevancy": r['rrf'].get('answer_relevancy', 0)
                })

            df_ragas = pd.DataFrame(ragas_df)
            st.dataframe(df_ragas, use_container_width=True)

    with tab4:
        st.subheader("Summary Statistics")

        # Overall statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Questions Evaluated", len(st.session_state.results))

        with col2:
            categories = set([r['category'] for r in st.session_state.results])
            st.metric("Unique Categories", len(categories))

        with col3:
            # Best performing approach
            avg_scores = {
                "Naive": sum([r['naive_metrics']['cosine'] for r in st.session_state.results]) / len(st.session_state.results),
                "Cohere": sum([r['cohere_metrics']['cosine'] for r in st.session_state.results]) / len(st.session_state.results),
                "RRF": sum([r['rrf_metrics']['cosine'] for r in st.session_state.results]) / len(st.session_state.results)
            }
            best_approach = max(avg_scores, key=avg_scores.get)
            st.metric("Best Approach (Avg Cosine)", best_approach, f"{avg_scores[best_approach]:.3f}")

        # Performance by category
        st.subheader("Performance by Category")
        category_performance = {}
        for result in st.session_state.results:
            cat = result['category']
            if cat not in category_performance:
                category_performance[cat] = {
                    'naive': [], 'cohere': [], 'rrf': []
                }
            category_performance[cat]['naive'].append(result['naive_metrics']['cosine'])
            category_performance[cat]['cohere'].append(result['cohere_metrics']['cosine'])
            category_performance[cat]['rrf'].append(result['rrf_metrics']['cosine'])

        category_df = []
        for cat, scores in category_performance.items():
            category_df.append({
                "Category": cat,
                "Naive RAG": sum(scores['naive']) / len(scores['naive']),
                "Cohere Reranking": sum(scores['cohere']) / len(scores['cohere']),
                "RRF": sum(scores['rrf']) / len(scores['rrf'])
            })

        df_category = pd.DataFrame(category_df)
        st.dataframe(df_category, use_container_width=True)

        # Visualization
        fig = px.bar(df_category, x='Category',
                     y=['Naive RAG', 'Cohere Reranking', 'RRF'],
                     title='Average Cosine Similarity by Category',
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure you have set up your API keys correctly and that the `paper_new_1.pdf` file is in the same directory.")
