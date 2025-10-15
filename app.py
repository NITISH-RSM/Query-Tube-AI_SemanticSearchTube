import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Try importing ML libraries with fallback
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ ML libraries not available: {e}")
    st.error("Please install: `pip install sentence-transformers scikit-learn`")
    ML_AVAILABLE = False
    st.stop()

# Check if required files exist
REQUIRED_FILES = ["video_index_with_embeddings.parquet"]
missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing required files: {missing_files}")
    st.info("Upload these files or check your file paths.")
    st.stop()

# Page config
st.set_page_config(
    page_title="QueryTube",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0B0F19; color: #E0E0E0; }
    .stApp { background-color: #0B0F19; }
    .stTextInput > div > div > input {
        background-color: #121826; border: 1px solid #2A3B4B; color: #E0E0E0;
    }
    .stButton > button {
        background-color: #2563EB; color: white; border-radius: 8px;
    }
    .video-card {
        background-color: #121826; border: 1px solid #2A3B4B;
        border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_model():
    """Load data and model with proper error handling."""
    try:
        # Load video index
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        st.success(f"✅ Loaded {len(df)} video records")
        
        # Determine best model
        try:
            results_df = pd.read_csv("model_evaluation_summary.csv")
            if results_df["avg_rank"].notna().any():
                best_row = results_df.loc[results_df["avg_rank"].idxmin()]
                best_model_name = best_row["model"]
                best_metric = best_row.get("metric", "cosine")
            else:
                best_model_name = "all-MiniLM-L6-v2"  # Smaller, faster model
                best_metric = "cosine"
        except FileNotFoundError:
            best_model_name = "all-MiniLM-L6-v2"  # Fallback to smaller model
            best_metric = "cosine"
        
        # Load model with timeout
        with st.spinner(f"Loading model: {best_model_name}"):
            model = SentenceTransformer(best_model_name)
        
        # Prepare embeddings
        embedding_columns = [col for col in df.columns if col.startswith("emb_")]
        if not embedding_columns:
            st.error("No embedding columns found!")
            st.stop()
        
        combined_embeddings = df[embedding_columns].values
        df["thumbnail"] = df["video_id"].apply(lambda x: f"https://i.ytimg.com/vi/{x}/hqdefault.jpg")
        df["channel"] = "WatchMojo"
        
        return model, combined_embeddings, df, best_metric
        
    except Exception as e:
        st.error(f"❌ Failed to load data/model: {str(e)}")
        st.info("Try installing dependencies manually or use a smaller model.")
        st.stop()
        return None, None, None, None

# Initialize
if ML_AVAILABLE:
    model, combined_embeddings, df, best_metric = load_data_and_model()
else:
    model, combined_embeddings, df, best_metric = None, None, None, None

def query_to_top5_videos(query, model, combined_embeddings, df, metric="cosine", top_k=5):
    """Semantic search with error handling."""
    if not query.strip() or model is None:
        return pd.DataFrame()
    
    try:
        # Encode query
        query_embedding_title = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        query_embedding_transcript = np.zeros_like(query_embedding_title)
        query_embedding = np.hstack((query_embedding_title, query_embedding_transcript))
        
        # Compute similarity
        if metric == "cosine":
            similarities = cosine_similarity(query_embedding, combined_embeddings)[0]
            scores = similarities
        else:
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
        st.error(f"Search failed: {e}")
        return pd.DataFrame()

def display_video_card(video):
    """Display video card."""
    embed_url = f"https://www.youtube.com/embed/{video['video_id']}?rel=0&modestbranding=1"
    st.markdown(f"""
    <div class="video-card">
        <iframe width="100%" height="200" src="{embed_url}" frameborder="0" allowfullscreen></iframe>
        <h3 style="color: #E0E0E0;">{video['title']}</h3>
        <p><strong>📺 Channel:</strong> {video['channel']}</p>
        <p><strong>⭐ Score:</strong> {video['score']:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

# UI
st.title("🎥 QueryTube")
st.markdown("### Semantic YouTube Search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if model:
        st.success(f"✅ Model loaded")
        st.info(f"**Metric:** {best_metric}")
        st.info(f"**Videos:** {len(df):,}")
    else:
        st.error("❌ Model not loaded")
    
    if st.button("🔄 Reload"):
        st.cache_resource.clear()
        st.rerun()

# Search
query = st.text_input("🔍 Search videos:", placeholder="e.g., 'AI breakthroughs'")

if st.button("🚀 Search", type="primary") and query and model:
    with st.spinner("Searching..."):
        results = query_to_top5_videos(query, model, combined_embeddings, df, best_metric)
        
        if not results.empty:
            st.success(f"✅ Found {len(results)} videos!")
            cols = st.columns(3)
            for i, video in enumerate(results.itertuples()):
                with cols[i % 3]:
                    display_video_card(video._asdict())
        else:
            st.warning("No results found!")

# Instructions for Streamlit Cloud
with st.expander("📋 Deployment Instructions"):
    st.markdown("""
    **For Streamlit Cloud:**
    1. Upload `video_index_with_embeddings.parquet`
    2. Use smaller model: `all-MiniLM-L6-v2`
    3. Check `packages.txt` for system deps
    
    **packages.txt:**
