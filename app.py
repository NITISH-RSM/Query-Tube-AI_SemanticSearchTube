import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

# Safe imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("Install: pip install sentence-transformers scikit-learn")

st.set_page_config(page_title="QueryTube", layout="wide")

# Fixed CSS for video embeds
st.markdown("""
<style>
    .video-container {
        background-color: #121826;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #2A3B4B;
    }
    .video-title {
        color: #E0E0E0;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    .video-meta {
        color: #A0AEC0;
        font-size: 0.9rem;
        margin: 0.25rem 0;
    }
    .stTextInput > div > div > input {
        background-color: #121826;
        color: #E0E0E0;
        border: 1px solid #2A3B4B;
    }
</style>
""", unsafe_allow_html=True)

# Global state
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
    st.session_state.df = None
    st.session_state.model = None
    st.session_state.embeddings = None

def validate_and_load_data():
    """Validate and load data with proper structure."""
    try:
        # Check file
        if not os.path.exists("video_index_with_embeddings.parquet"):
            st.error("❌ video_index_with_embeddings.parquet missing!")
            return False
        
        # Load and validate
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        
        required_cols = ["video_id", "title"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            return False
        
        # Check embeddings
        emb_cols = [col for col in df.columns if col.startswith("emb_")]
        if not emb_cols:
            st.error("❌ No embedding columns (starting with 'emb_') found!")
            return False
        
        # Validate video_ids (YouTube format)
        valid_ids = df["video_id"].str.match(r'^[a-zA-Z0-9_-]{11}$').all()
        if not valid_ids:
            st.warning("⚠️ Some video_ids may not be valid YouTube IDs")
        
        # Prepare data
        st.session_state.df = df.copy()
        st.session_state.embeddings = df[emb_cols].values.astype(np.float32)
        
        # Add metadata
        st.session_state.df["channel"] = "WatchMojo"  # Default
        st.session_state.df["thumbnail"] = st.session_state.df["video_id"].apply(
            lambda x: f"https://img.youtube.com/vi/{x}/hqdefault.jpg"
        )
        
        st.success(f"✅ Data loaded: {len(df)} videos, {st.session_state.embeddings.shape[1]} embedding dims")
        return True
        
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return False

def load_model():
    """Load lightweight model."""
    try:
        with st.spinner("Loading AI model..."):
            # Use small, reliable model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            st.success("✅ Model loaded successfully")
            st.session_state.model = model
            return True
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return False

def initialize_system():
    """Complete system initialization."""
    with st.spinner("🚀 Initializing QueryTube..."):
        if not validate_and_load_data():
            return False
        
        if not ML_AVAILABLE:
            st.error("ML libraries not available")
            return False
        
        if not load_model():
            return False
        
        st.session_state.system_ready = True
        return True

def search_videos(query, top_k=5):
    """Perform semantic search."""
    if not st.session_state.system_ready:
        return pd.DataFrame()
    
    try:
        model = st.session_state.model
        embeddings = st.session_state.embeddings
        df = st.session_state.df
        
        # Encode query
        with st.spinner("Analyzing query..."):
            query_embedding = model.encode([query], show_progress_bar=False)
        
        # Handle dimension mismatch
        if query_embedding.shape[1] != embeddings.shape[1]:
            st.warning(f"⚠️ Dimension mismatch: {query_embedding.shape[1]} vs {embeddings.shape[1]}")
            # Pad or truncate
            if query_embedding.shape[1] < embeddings.shape[1]:
                padding = np.zeros((1, embeddings.shape[1] - query_embedding.shape[1]))
                query_embedding = np.hstack([query_embedding, padding])
            else:
                query_embedding = query_embedding[:, :embeddings.shape[1]]
        
        # Compute similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        results = df.iloc[top_indices].copy()
        results["score"] = top_scores
        
        st.success(f"✅ Found {len(results)} relevant videos")
        return results
        
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        return pd.DataFrame()

def display_video_result(video_data):
    """Display single video with working embed."""
    video_id = video_data["video_id"]
    title = video_data["title"]
    score = video_data["score"]
    channel = video_data.get("channel", "Unknown")
    
    # Test YouTube embed URL
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    
    # Create responsive video card
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Try iframe first
        st.markdown(f"""
        <div class="video-container">
            <iframe 
                width="100%" 
                height="200" 
                src="{embed_url}?rel=0&modestbranding=1" 
                frameborder="0" 
                allowfullscreen
                title="{title}">
            </iframe>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="video-container" style="height: 200px; display: flex; align-items: center;">
            <img src="https://img.youtube.com/vi/{video_id}/mqdefault.jpg" 
                 alt="Thumbnail" 
                 style="width: 100%; border-radius: 8px;">
        </div>
        """, unsafe_allow_html=True)
    
    # Video info
    st.markdown(f"""
    <div class="video-container">
        <h3 class="video-title">{title}</h3>
        <p class="video-meta"><strong>📺 Channel:</strong> {channel}</p>
        <p class="video-meta"><strong>⭐ Relevance:</strong> {score:.3f}</p>
        <p class="video-meta">
            <strong>🔗 Watch:</strong> 
            <a href="https://youtube.com/watch?v={video_id}" target="_blank">
                Open in YouTube
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# === MAIN APP ===
st.title("🎥 QueryTube")
st.markdown("### Semantic search for YouTube videos")

# System status
if not st.session_state.system_ready:
    if st.button("🚀 Initialize System"):
        if initialize_system():
            st.rerun()
    else:
        st.info("👆 Click 'Initialize System' to start")
        
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write("**Files:**")
            for file in ["video_index_with_embeddings.parquet"]:
                st.write(f"- {file}: {'✅' if os.path.exists(file) else '❌'}")
            
            st.write("**ML Libraries:**")
            st.write(f"- sentence-transformers: {'✅' if ML_AVAILABLE else '❌'}")
        
        # Create test data option
        if st.button("🧪 Create Test Data"):
            create_test_data()
            st.success("Test data created! Click 'Initialize System'")
        st.stop()

# Search interface (only if ready)
st.markdown("---")
query = st.text_input(
    "🔍 Search videos:",
    placeholder="e.g., 'Top 10 movies', 'AI explained', 'Tech tutorials'",
    help="Enter your search query"
)

if st.button("🔍 Search", type="primary") and query.strip():
    with st.spinner("Searching..."):
        results = search_videos(query.strip())
        
        if not results.empty:
            st.success(f"✅ Found {len(results)} videos!")
            
            # Display results
            for idx, (_, video) in enumerate(results.iterrows()):
                st.markdown(f"---")
                display_video_result(video)
                
                # Test video availability
                if idx == 0:  # Test first video
                    st.info(f"🧪 Testing video ID: {video['video_id']}")
            
            # Download results
            csv = results[["video_id", "title", "score", "channel"]].to_csv(index=False)
            st.download_button(
                "💾 Download Results",
                csv,
                "search_results.csv",
                "text/csv"
            )
        else:
            st.warning("😔 No videos found. Try different keywords!")
            st.info("💡 Tips: Use specific topics, try synonyms")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Info")
    if st.session_state.system_ready:
        st.success("🟢 Ready")
        st.info(f"**Videos:** {len(st.session_state.df):,}")
        st.info(f"**Model:** all-MiniLM-L6-v2")
    else:
        st.error("🔴 Not Ready")
    
    if st.button("🔄 Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Help
with st.expander("📖 Troubleshooting Videos Not Loading"):
    st.markdown("""
    ## Common Issues & Fixes:
    
    ### 1. **Invalid Video IDs**
    - Check if `video_id` column contains 11-character YouTube IDs
    - Format: `[a-zA-Z0-9_-]{11}`
    
    ### 2. **YouTube Embed Restrictions**
    - Some videos have embedding disabled
    - Solution: Use thumbnail + external link
    
    ### 3. **Network/CORS Issues**
    - Streamlit Cloud may block iframes
    - Solution: Fallback to thumbnails + links
    
    ### 4. **Test Your Data**
    ```python
    # Check video IDs
    df['video_id'].str.len().value_counts()
    df['video_id'].str.match(r'^[a-zA-Z0-9_-]{{11}}$').sum()
