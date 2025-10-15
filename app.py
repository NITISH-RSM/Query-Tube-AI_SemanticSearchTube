import streamlit as st
import pandas as pd
import numpy as np
import os

# Safe imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

st.set_page_config(page_title="QueryTube", layout="wide")

# CSS as single line
st.markdown('<style>.video-box{background:#121826;border:1px solid #2A3B4B;border-radius:8px;padding:1rem;margin:1rem 0;}.title{color:#E0E0E0;font-size:1.1rem;}.meta{color:#A0AEC0;}</style>', unsafe_allow_html=True)

# Session state
if "ready" not in st.session_state:
    st.session_state.ready = False
    st.session_state.df = None
    st.session_state.model = None

def create_test_file():
    """Create simple test data."""
    videos = [
        ("dQw4w9WgXcQ", "Rick Astley Never Gonna Give You Up"),
        ("kJQP7kiw5Fk", "The Duck Song"),
        ("9bZkp7q19f0", "Charlie bit my finger"),
        ("rVlgVCQRx88", "Evolution of Dance")
    ]
    
    data = []
    for i in range(20):
        vid_id, title = videos[i % 4]
        row = {"video_id": vid_id, "title": title}
        for j in range(384):
            row[f"emb_{j}"] = np.random.randn()
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_parquet("video_index_with_embeddings.parquet", index=False)
    return True

def load_data():
    """Load data safely."""
    try:
        if not os.path.exists("video_index_with_embeddings.parquet"):
            return False
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        if "video_id" not in df.columns or "title" not in df.columns:
            return False
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            return False
        st.session_state.df = df
        st.session_state.embeddings = df[emb_cols].values
        return True
    except:
        return False

def load_model_safe():
    """Load model with fallback."""
    if not ML_AVAILABLE:
        return False
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.session_state.model = model
        return True
    except:
        return False

def init_system():
    """Initialize everything."""
    if not load_data():
        st.error("Data load failed")
        return False
    if not load_model_safe():
        st.error("Model load failed")
        return False
    st.session_state.ready = True
    return True

def do_search(query):
    """Simple search function."""
    try:
        model = st.session_state.model
        embeddings = st.session_state.embeddings
        df = st.session_state.df
        
        q_emb = model.encode([query])
        if q_emb.shape[1] != embeddings.shape[1]:
            if q_emb.shape[1] < embeddings.shape[1]:
                pad = np.zeros((1, embeddings.shape[1] - q_emb.shape[1]))
                q_emb = np.hstack([q_emb, pad])
            else:
                q_emb = q_emb[:, :embeddings.shape[1]]
        
        sims = cosine_similarity(q_emb, embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        results = df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results
    except:
        return pd.DataFrame()

def show_video(vid):
    """Show video without complex HTML."""
    vid_id = vid["video_id"]
    title = vid["title"]
    score = vid["score"]
    
    # Simple components instead of raw HTML
    col1, col2 = st.columns([3,1])
    with col1:
        st.video(f"https://www.youtube.com/watch?v={vid_id}")
    with col2:
        st.markdown(f"**{title}**")
        st.markdown(f"Score: {score:.3f}")
        st.markdown(f"[Watch on YouTube](https://youtube.com/watch?v={vid_id})")

# MAIN APP
st.title("🎥 QueryTube")

if not st.session_state.ready:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Test Data"):
            create_test_file()
            st.success("Test data created!")
    with col2:
        if st.button("Initialize"):
            if init_system():
                st.success("System ready!")
                st.rerun()
    st.stop()

# Search when ready
st.success("System Ready!")
query = st.text_input("Search videos")
if st.button("Search") and query:
    results = do_search(query)
    if not results.empty:
        for _, vid in results.iterrows():
            show_video(vid)
    else:
        st.warning("No results")

# Sidebar
with st.sidebar:
    st.info("Click 'Create Test Data' then 'Initialize'")
    if st.button("Reset"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
