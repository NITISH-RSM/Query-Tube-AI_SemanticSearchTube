import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# Safe imports with error handling
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
    print("✅ ML libraries loaded")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ ML import error: {e}")

# Page config
st.set_page_config(
    page_title="QueryTube", 
    page_icon="🎥", 
    layout="wide"
)

# Simple CSS without complex indentation
css = """
<style>
.video-container { background: #121826; border-radius: 8px; padding: 1rem; margin: 1rem 0; border: 1px solid #2A3B4B; }
.video-title { color: #E0E0E0; font-size: 1.1rem; margin: 0.5rem 0; }
.video-meta { color: #A0AEC0; font-size: 0.9rem; margin: 0.25rem 0; }
.stTextInput input { background: #121826; color: #E0E0E0; border: 1px solid #2A3B4B; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Session state initialization
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
    st.session_state.df = None
    st.session_state.model = None
    st.session_state.embeddings = None

def check_files():
    """Check if required files exist."""
    parquet_exists = os.path.exists("video_index_with_embeddings.parquet")
    return parquet_exists

def create_test_data():
    """Create minimal test dataset with valid YouTube IDs."""
    try:
        # Real working YouTube video IDs
        test_videos = [
            {"id": "dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up"},
            {"id": "kJQP7kiw5Fk", "title": "The Duck Song"},
            {"id": "9bZkp7q19f0", "title": "Charlie bit my finger"},
            {"id": "rVlgVCQRx88", "title": "Evolution of Dance"},
            {"id": "3JZ_D3ELwOQ", "title": "Badger Badger Badger"},
        ]
        
        n_samples = 20
        video_ids = []
        titles = []
        
        for i in range(n_samples):
            vid = test_videos[i % len(test_videos)]
            video_ids.extend([vid["id"]] * 4)  # Repeat each video
            titles.extend([vid["title"]] * 4)
        
        # Create dummy embeddings (384 dimensions for MiniLM)
        embeddings = np.random.randn(n_samples, 384).astype(np.float32)
        
        # Create DataFrame
        df = pd.DataFrame()
        for i in range(384):
            df[f"emb_{i}"] = embeddings[:, i]
        
        df["video_id"] = video_ids[:n_samples]
        df["title"] = titles[:n_samples]
        
        # Save
        df.to_parquet("video_index_with_embeddings.parquet", index=False)
        st.success(f"✅ Created test data: {n_samples} videos")
        return True
    except Exception as e:
        st.error(f"❌ Test data creation failed: {e}")
        return False

def load_data():
    """Load and validate data."""
    try:
        if not os.path.exists("video_index_with_embeddings.parquet"):
            st.error("❌ Parquet file missing!")
            return False
        
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        
        # Validate columns
        if "video_id" not in df.columns or "title" not in df.columns:
            st.error("❌ Missing 'video_id' or 'title' columns")
            return False
        
        # Check embeddings
        emb_cols = [col for col in df.columns if col.startswith("emb_")]
        if not emb_cols:
            st.error("❌ No embedding columns found")
            return False
        
        # Store in session
        st.session_state.df = df
        st.session_state.embeddings = df[emb_cols].values.astype(np.float32)
        
        # Add metadata
        st.session_state.df["channel"] = "YouTube"
        
        st.success(f"✅ Data loaded: {len(df)} videos")
        return True
    except Exception as e:
        st.error(f"❌ Data load error: {e}")
        return False

def load_model():
    """Load sentence transformer model."""
    try:
        if not ML_AVAILABLE:
            st.error("❌ ML libraries not available")
            return False
        
        with st.spinner("Loading AI model..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            st.session_state.model = model
            st.success("✅ Model loaded")
            return True
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        st.info("Try: pip install sentence-transformers")
        return False

def initialize_system():
    """Initialize complete system."""
    with st.spinner("Setting up QueryTube..."):
        # Load data first
        if not load_data():
            return False
        
        # Then model
        if not load_model():
            return False
        
        st.session_state.system_ready = True
        return True

def search_videos(query):
    """Perform semantic search."""
    try:
        model = st.session_state.model
        embeddings = st.session_state.embeddings
        df = st.session_state.df
        
        # Encode query
        query_emb = model.encode([query.strip()])
        
        # Handle dimension mismatch
        target_dim = embeddings.shape[1]
        if query_emb.shape[1] != target_dim:
            if query_emb.shape[1] < target_dim:
                # Pad with zeros
                padding = np.zeros((1, target_dim - query_emb.shape[1]))
                query_emb = np.hstack([query_emb, padding])
            else:
                # Truncate
                query_emb = query_emb[:, :target_dim]
        
        # Calculate similarity
        similarities = cosine_similarity(query_emb, embeddings)[0]
        
        # Get top 5
        top_indices = np.argsort(similarities)[::-1][:5]
        top_scores = similarities[top_indices]
        
        results = df.iloc[top_indices].copy()
        results["score"] = top_scores
        results = results.sort_values("score", ascending=False)
        
        return results
    except Exception as e:
        st.error(f"❌ Search error: {e}")
        return pd.DataFrame()

def display_video(video_data):
    """Display video result with working embed."""
    video_id = video_data["video_id"]
    title = video_data["title"]
    score = video_data["score"]
    
    # YouTube embed URL
    embed_url = f"https://www.youtube.com/embed/{video_id}?rel=0"
    
    # Simple HTML without complex indentation
    html = f'''
    <div class="video-container">
        <iframe width="100%" height="200" 
                src="{embed_url}" 
                frameborder="0" 
                allowfullscreen>
        </iframe>
        <h3 class="video-title">{title}</h3>
        <p class="video-meta">⭐ Score: {score:.3f}</p>
        <p class="video-meta">
            <a href="https://youtube.com/watch?v={video_id}" target="_blank">
                🔗 Watch on YouTube
            </a>
        </p>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# === MAIN UI ===
st.title("🎥 QueryTube")
st.markdown("**Semantic search for YouTube videos**")

# System initialization
if not st.session_state.system_ready:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_system():
                st.rerun()
    
    with col2:
        if st.button("🧪 Create Test Data"):
            if create_test_data():
                st.info("Test data created! Now click 'Initialize System'")
    
    with col3:
        st.info("**Status:** Not Ready")
    
    # File check
    st.markdown("---")
    st.subheader("📁 File Status")
    if check_files():
        st.success("✅ video_index_with_embeddings.parquet found")
    else:
        st.error("❌ Parquet file missing")
    
    st.stop()

# Search interface (when ready)
st.success("✅ System Ready!")
st.markdown("---")

# Search box
query = st.text_input(
    "🔍 Enter search query:",
    placeholder="e.g., 'programming', 'AI', 'tutorials'"
)

if st.button("🔍 Search Videos", type="primary") and query.strip():
    with st.spinner("Searching..."):
        results = search_videos(query)
        
        if not results.empty:
            st.success(f"✅ Found {len(results)} results!")
            
            for _, video in results.iterrows():
                display_video(video)
                st.markdown("---")
            
            # Download option
            csv_data = results[["video_id", "title", "score"]].to_csv(index=False)
            st.download_button(
                "💾 Download Results",
                csv_data,
                "search_results.csv",
                "text/csv"
            )
        else:
            st.warning("😔 No results found")
            st.info("Try different keywords")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Info")
    if st.session_state.system_ready:
        st.success("🟢 Ready")
        if st.session_state.df is not None:
            st.info(f"📊 {len(st.session_state.df)} videos indexed")
        st.info("🤖 all-MiniLM-L6-v2 model")
    else:
        st.error("🔴 Not Ready")
    
    st.markdown("---")
    if st.button("🔄 Reset System"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Help section
with st.expander("📖 Help & Troubleshooting"):
    st.markdown("""
    ### 🚀 Quick Start
    1. Click **"Create Test Data"** (generates sample videos)
    2. Click **"Initialize System"** 
    3. Search for videos
    
    ### 📁 Requirements
    - `video_index_with_embeddings.parquet` file
    - Columns: `video_id`, `title`, `emb_*` (embeddings)
    
    ### 🔧 Installation
    ```bash
    pip install streamlit pandas numpy
    pip install sentence-transformers torch
    pip install scikit-learn pyarrow
