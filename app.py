import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from pathlib import Path

# Try importing ML libraries with error handling
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    ML_AVAILABLE = True
    print("✅ ML libraries imported successfully")
except ImportError as e:
    st.error(f"❌ ML libraries import failed: {e}")
    st.info("Install with: `pip install sentence-transformers scikit-learn torch`")
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="QueryTube",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (fixed indentation)
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
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #2563EB; 
        color: white; 
        border-radius: 8px;
        border: none;
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
</style>
""", unsafe_allow_html=True)

# Global variables
model = None
combined_embeddings = None
df = None
best_metric = "cosine"

@st.cache_resource
def load_data_and_model():
    """Load data and model with comprehensive error handling."""
    global model, combined_embeddings, df, best_metric
    
    try:
        # Check and load video index
        if not os.path.exists("video_index_with_embeddings.parquet"):
            st.error("❌ File 'video_index_with_embeddings.parquet' not found!")
            st.info("Please upload this file to the project directory.")
            st.stop()
        
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        st.success(f"✅ Loaded {len(df)} video records")
        
        # Determine best model (use smaller model for cloud)
        model_choices = {
            "Fast (Recommended)": "all-MiniLM-L6-v2",
            "Better Quality": "all-mpnet-base-v2",
            "Original": "paraphrase-mpnet-base-v2"
        }
        
        try:
            results_df = pd.read_csv("model_evaluation_summary.csv")
            if not results_df["avg_rank"].isna().all():
                best_row = results_df.loc[results_df["avg_rank"].idxmin()]
                best_model_name = best_row["model"]
                best_metric = best_row.get("metric", "cosine")
                st.info(f"🎯 Using evaluated model: {best_model_name}")
            else:
                best_model_name = model_choices["Fast (Recommended)"]
        except FileNotFoundError:
            best_model_name = model_choices["Fast (Recommended)"]
            st.warning("⚠️ Using default fast model: all-MiniLM-L6-v2")
        
        # Load model with timeout and memory management
        with st.spinner(f"Loading model: {best_model_name}..."):
            model = SentenceTransformer(best_model_name)
        
        st.success(f"✅ Model loaded successfully: {best_model_name}")
        
        # Validate and prepare embeddings
        embedding_columns = [col for col in df.columns if col.startswith("emb_")]
        if not embedding_columns:
            st.error("❌ No embedding columns found in parquet file!")
            st.stop()
        
        combined_embeddings = df[embedding_columns].values
        df["thumbnail"] = df["video_id"].apply(lambda x: f"https://i.ytimg.com/vi/{x}/hqdefault.jpg")
        df["channel"] = "WatchMojo"  # Default channel
        
        st.info(f"📊 Embeddings shape: {combined_embeddings.shape}")
        return model, combined_embeddings, df, best_metric
        
    except Exception as e:
        st.error(f"❌ Data/Model loading failed: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Check if parquet file has embedding columns (starting with 'emb_')
        2. Try a smaller model: all-MiniLM-L6-v2
        3. Ensure sufficient memory (at least 2GB RAM)
        """)
        st.stop()
        return None, None, None, None

def query_to_top5_videos(query, top_k=5, metric="cosine"):
    """Semantic search function with error handling."""
    global model, combined_embeddings, df
    
    if not query.strip() or model is None or df is None:
        return pd.DataFrame(columns=["video_id", "title", "score", "thumbnail", "channel"])
    
    try:
        st.info(f"🔍 Processing query: '{query}'")
        
        # Encode query
        query_embedding_title = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        # Handle dimension mismatch
        if query_embedding_title.shape[1] != combined_embeddings.shape[1] // 2:
            st.warning("⚠️ Embedding dimension mismatch, padding query embedding")
            padding = np.zeros((1, combined_embeddings.shape[1] // 2))
            query_embedding_title = np.hstack([query_embedding_title, padding])
        
        query_embedding = query_embedding_title
        
        # Compute similarity
        if metric == "cosine":
            similarities = cosine_similarity(query_embedding, combined_embeddings)[0]
            scores = similarities
        elif metric == "euclidean":
            distances = euclidean_distances(query_embedding, combined_embeddings)[0]
            scores = 1 / (1 + distances)
        else:
            st.error(f"Unsupported metric: {metric}")
            return pd.DataFrame()
        
        # Get top-k results
        top_k_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        top_k_scores = scores[top_k_indices]
        
        result_df = pd.DataFrame({
            "video_id": df.iloc[top_k_indices]["video_id"].values,
            "title": df.iloc[top_k_indices]["title"].values,
            "score": top_k_scores,
            "thumbnail": df.iloc[top_k_indices]["thumbnail"].values,
            "channel": df.iloc[top_k_indices]["channel"].values
        })
        
        st.success(f"✅ Found {len(result_df)} relevant videos")
        return result_df
        
    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        return pd.DataFrame()

def display_video_card(video_data):
    """Display video result as HTML card."""
    video_id = video_data["video_id"]
    title = video_data["title"]
    channel = video_data["channel"]
    score = video_data["score"]
    
    embed_url = f"https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1&iv_load_policy=3&autoplay=0"
    
    # Fixed HTML string (no indentation issues)
    html_card = f"""
    <div class="video-card">
        <iframe 
            width="100%" 
            height="200" 
            src="{embed_url}" 
            frameborder="0" 
            allowfullscreen>
        </iframe>
        <h3 style="color: #E0E0E0; margin-top: 1rem; font-size: 1.1rem;">{title}</h3>
        <p style="color: #A0AEC0;"><strong>📺 Channel:</strong> {channel}</p>
        <p style="color: #A0AEC0;"><strong>⭐ Relevance Score:</strong> {score:.3f}</p>
        <p style="color: #A0AEC0; font-style: italic; font-size: 0.9rem;">
            *Click play to watch the video preview*
        </p>
    </div>
    """
    st.markdown(html_card, unsafe_allow_html=True)

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Load data and model
if ML_AVAILABLE and not st.session_state.model_loaded:
    with st.spinner("🚀 Initializing QueryTube..."):
        model, combined_embeddings, df, best_metric = load_data_and_model()
        if model:
            st.session_state.model_loaded = True

# Header
st.title("🎥 QueryTube")
st.markdown("### 🔍 Semantic Search Engine for YouTube Videos")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if st.session_state.model_loaded and model:
        st.success("✅ System Ready")
        st.info(f"**Model:** {model[0].__class__.__name__}")
        st.info(f"**Metric:** {best_metric}")
        st.info(f"**Videos Indexed:** {len(df):,}")
        st.info(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    else:
        st.error("❌ System Not Ready")
        st.info("Check file uploads and dependencies")
    
    st.markdown("---")
    if st.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.session_state.model_loaded = False
        st.rerun()
    
    st.markdown("---")
    st.info("**Required Files:**")
    st.code("video_index_with_embeddings.parquet")

# Main search interface
st.markdown("---")
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., 'Top 10 AI breakthroughs', 'Quantum computing basics'",
        help="Type your query and press Enter or click Search"
    )

with col2:
    search_trigger = st.button("🚀 Search", type="primary", use_container_width=True)

# Process search
if (search_trigger or query) and st.session_state.model_loaded and query.strip():
    with st.spinner("🔍 Searching for relevant videos..."):
        time.sleep(1)  # Simulate processing
        
        results = query_to_top5_videos(query.strip(), metric=best_metric)
        
        if not results.empty:
            st.success(f"✅ Found {len(results)} relevant videos!")
            
            # Display results in responsive columns
            num_cols = min(3, len(results))
            cols = st.columns(num_cols)
            
            for i, (_, video) in enumerate(results.iterrows()):
                with cols[i % num_cols]:
                    display_video_card(video)
            
            # Save results
            try:
                results.to_csv("last_search_results.csv", index=False)
                st.success("💾 Results saved to last_search_results.csv")
            except Exception as e:
                st.warning(f"Could not save results: {e}")
                
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top Score", f"{results['score'].max():.3f}")
            with col2:
                st.metric("Average Score", f"{results['score'].mean():.3f}")
            with col3:
                st.metric("Total Results", len(results))
                
        else:
            st.warning("😔 No videos found matching your query. Try different keywords!")
            st.info("Tips: Use specific topics, avoid very common terms")

# Instructions and debug
with st.expander("ℹ️ About & Troubleshooting"):
    st.markdown("""
    ## 🎯 About QueryTube
    
    **QueryTube** uses semantic search to find relevant YouTube videos based on:
    - **Sentence Transformers** for query understanding
    - **Cosine similarity** for ranking
    - **Pre-computed embeddings** from video titles/transcripts
    
    ## 📋 Requirements
    - `video_index_with_embeddings.parquet` (must have `emb_` columns)
    - Minimum 2GB RAM for model loading
    
    ## 🛠️ Troubleshooting
    1. **Model fails to load**: Use `all-MiniLM-L6-v2` (smaller model)
    2. **Memory errors**: Clear cache and restart
    3. **No embeddings**: Check parquet file structure
    4. **Import errors**: Install `pip install sentence-transformers torch`
    
    ## 🚀 For Streamlit Cloud
    - Upload parquet file to repo
    - Use `packages.txt` for system deps
    - Set model to `all-MiniLM-L6-v2`
    """)

# Debug section
if st.checkbox("🐛 Show Debug Info"):
    st.subheader("Debug Information")
    if df is not None:
        st.json({
            "Total Videos": len(df),
            "Embedding Shape": str(combined_embeddings.shape) if combined_embeddings is not None else "None",
            "Columns": df.columns.tolist(),
            "Sample Data": df.head(2).to_dict()
        })
    else:
        st.error("No data loaded for debugging")
