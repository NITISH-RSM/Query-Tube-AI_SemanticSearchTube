import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import time
import os
import sys
from pathlib import Path

# Page config
st.set_page_config(
    page_title="QueryTube",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0B0F19;
        color: #E0E0E0;
    }
    .stApp {
        background-color: #0B0F19;
    }
    .stTextInput > div > div > input {
        background-color: #121826;
        border: 1px solid #2A3B4B;
        color: #E0E0E0;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
    .video-card {
        background-color: #121826;
        border: 1px solid #2A3B4B;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #121826;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2A3B4B;
    }
    .sidebar .sidebar-content {
        background-color: #121826;
    }
    .stSlider > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
@st.cache_resource
def load_data_and_model():
    """Load data and model with caching."""
    global model, combined_embeddings, df, best_metric
    
    # Load video index
    try:
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        st.success(f"✅ Loaded {len(df)} video records")
    except FileNotFoundError:
        st.error("❌ 'video_index_with_embeddings.parquet' not found!")
        st.stop()

    # Load best model info
    try:
        results_df = pd.read_csv("model_evaluation_summary.csv")
        if results_df["avg_rank"].notna().any():
            best_row = results_df.loc[results_df["avg_rank"].idxmin()]
            best_model_name = best_row["model"]
            best_metric = best_row.get("metric", "cosine")
            st.info(f"🎯 Using best model: {best_model_name} with {best_metric} metric")
        else:
            best_model_name = "paraphrase-mpnet-base-v2"
            best_metric = "cosine"
    except FileNotFoundError:
        best_model_name = "paraphrase-mpnet-base-v2"
        best_metric = "cosine"
        st.warning("⚠️ Using default model: paraphrase-mpnet-base-v2")

    # Load model
    try:
        with st.spinner("Loading AI model..."):
            model = SentenceTransformer(best_model_name)
        st.success(f"✅ Model loaded: {best_model_name}")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

    # Prepare embeddings
    embedding_columns = [col for col in df.columns if col.startswith("emb_")]
    if not embedding_columns:
        st.error("❌ No embedding columns found!")
        st.stop()
    
    combined_embeddings = df[embedding_columns].values
    df["thumbnail"] = df["video_id"].apply(lambda x: f"https://i.ytimg.com/vi/{x}/hqdefault.jpg")
    df["channel"] = "WatchMojo"

    return model, combined_embeddings, df, best_metric

def query_to_top5_videos(query, top_k=5, metric="cosine"):
    """Semantic search function."""
    global model, combined_embeddings, df, best_metric
    
    if not query.strip():
        return pd.DataFrame(columns=["video_id", "title", "score", "thumbnail", "channel"])

    try:
        # Encode query
        query_embedding_title = model.encode([query], show_progress_bar=False)
        query_embedding_transcript = np.zeros_like(query_embedding_title)
        query_embedding = np.hstack((query_embedding_title, query_embedding_transcript))

        # Compute similarity
        if metric == "cosine":
            similarities = cosine_similarity(query_embedding, combined_embeddings)[0]
            scores = similarities
        elif metric == "euclidean":
            distances = euclidean_distances(query_embedding, combined_embeddings)[0]
            scores = 1 / (1 + distances)

        # Get top results
        top_k_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        top_k_scores = scores[top_k_indices]
        
        result_df = pd.DataFrame({
            "video_id": df.iloc[top_k_indices]["video_id"].values,
            "title": df.iloc[top_k_indices]["title"].values,
            "score": top_k_scores,
            "thumbnail": df.iloc[top_k_indices]["thumbnail"].values,
            "channel": df.iloc[top_k_indices]["channel"].values
        })
        
        return result_df
    except Exception as e:
        st.error(f"Search error: {e}")
        return pd.DataFrame()

def display_video_card(video):
    """Display video result as card."""
    embed_url = f"https://www.youtube.com/embed/{video['video_id']}?rel=0&modestbranding=1&iv_load_policy=3&autoplay=0"
    
    st.markdown(f"""
    <div class="video-card">
        <iframe width="100%" height="200" src="{embed_url}" 
                frameborder="0" allowfullscreen></iframe>
        <h3 style="color: #E0E0E0; margin-top: 1rem;">{video["title"]}</h3>
        <p><strong>📺 Channel:</strong> {video["channel"]}</p>
        <p><strong>⭐ Relevance Score:</strong> {video["score"]:.3f}</p>
        <p style="color: #A0AEC0; font-style: italic;">*Click play to watch the video*</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize
model, combined_embeddings, df, best_metric = load_data_and_model()

# Header
st.title("🎥 QueryTube")
st.markdown("### Semantic Search for YouTube Videos")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"**Model:** {model[0].__class__.__name__}")
    st.info(f"**Metric:** {best_metric}")
    st.info(f"**Videos Indexed:** {len(df):,}")
    
    st.markdown("---")
    st.header("📊 Search Options")
    search_sensitivity = st.slider("Search Sensitivity", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("🔄 Reload Data"):
        st.cache_resource.clear()
        st.rerun()

# Main search interface
st.markdown("---")
query = st.text_input(
    "🔍 Enter your search query:",
    placeholder="e.g., 'Top 10 AI breakthroughs', 'Quantum computing explained'",
    label_visibility="collapsed"
)

if st.button("🚀 Search Videos", type="primary"):
    if query:
        with st.spinner("🔍 Searching for relevant videos..."):
            # Simulate processing time
            time.sleep(1.5)
            
            search_results = query_to_top5_videos(query, metric=best_metric)
            
            if not search_results.empty:
                st.success(f"✅ Found {len(search_results)} relevant videos!")
                
                # Display results in columns
                cols = st.columns(min(3, len(search_results)))
                for i, video in enumerate(search_results.itertuples()):
                    with cols[i % len(cols)]:
                        display_video_card(video._asdict())
                        
                # Save results
                search_results.to_csv("last_search_results.csv", index=False)
                st.success("💾 Results saved to last_search_results.csv")
                
                # Metrics
                st.markdown("### 📈 Search Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Top Score", f"{search_results['score'].max():.3f}")
                with col2:
                    st.metric("Avg Score", f"{search_results['score'].mean():.3f}")
                with col3:
                    st.metric("Results", len(search_results))
                    
            else:
                st.error("😔 No videos found. Try a different query!")
    else:
        st.warning("⚠️ Please enter a search query!")

# Footer info
with st.expander("ℹ️ About QueryTube"):
    st.markdown("""
    **QueryTube** is a semantic search engine for YouTube videos using:
    - **Sentence Transformers** for semantic embeddings
    - **Cosine/Euclidean similarity** for ranking
    - **Top-5 results** without threshold filtering
    
    **Required Files:**
    - `video_index_with_embeddings.parquet` - Video embeddings
    - `model_evaluation_summary.csv` - Model evaluation (optional)
    
    **Features:**
    - Dark theme UI
    - Embedded YouTube players
    - Relevance scoring
    - Cached model loading
    """)

# Debug info
if st.checkbox("🐛 Show Debug Info"):
    st.subheader("Debug Information")
    st.json({
        "Total Videos": len(df),
        "Embedding Shape": combined_embeddings.shape,
        "Available Columns": df.columns.tolist(),
        "Memory Usage": df.memory_usage(deep=True).sum() / 1024**2
    })