import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Safe imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="QueryTube", layout="wide")

# EXACT Gradio CSS
st.markdown("""
<style>
    .main { background-color: #0B0F19 !important; color: #E0E0E0; }
    .stApp { background-color: #0B0F19 !important; }
    
    /* Header */
    .header-main {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.75rem 1.5rem; background-color: #121826 !important;
        border-bottom: 1px solid #2A3B4B; margin-bottom: 1rem;
    }
    .logo-text { font-size: 1.5rem; font-weight: bold; color: #FFFFFF; }
    
    /* Sidebar */
    .sidebar-nav {
        background-color: #121826 !important; padding: 1rem; border-radius: 12px;
        border: 1px solid #2A3B4B; min-height: 80vh;
    }
    .nav-button {
        background: transparent !important; color: #A0AEC0 !important; 
        text-align: left !important; border: none !important; 
        width: 100% !important; font-size: 1rem !important; 
        padding: 0.75rem !important; margin: 0.25rem 0 !important;
    }
    .nav-button:hover { background-color: #2A3B4B !important; }
    .nav-button.primary { background-color: #2563EB !important; color: white !important; }
    
    /* Result Cards */
    .result_card {
        background: #121826 !important; border-radius: 8px !important; 
        overflow: hidden; border: 1px solid #2A3B4B !important; 
        height: 100%; margin: 1rem 0 !important;
    }
    .result_card iframe { width: 100%; height: 180px; border: none; display: block; }
    .card-content { padding: 1rem; }
    .card-title { font-weight: 600; font-size: 1.1rem; color: #E0E0E0; margin-bottom: 0.5rem; }
    .card-channel { font-size: 0.9rem; color: #A0AEC0; }
    
    /* Skeleton */
    .skeleton_box {
        height: 280px; border-radius: 8px; border: 1px solid #2A3B4B;
        background: linear-gradient(90deg, rgba(42,59,75,0.2) 25%, rgba(42,59,75,0.4) 50%, rgba(42,59,75,0.2) 75%);
        background-size: 200% 100%; animation: shimmer 1.5s infinite linear;
    }
    @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #121826 !important; border: 1px solid #2A3B4B !important;
        color: #E0E0E0 !important; border-radius: 8px !important;
    }
    .stButton > button {
        background-color: #2563EB !important; color: white !important;
        border-radius: 8px !important; border: none !important;
    }
    
    /* User Profile */
    .user-profile {
        border-top: 1px solid #2A3B4B; margin-top: 2rem; padding-top: 1rem;
        display: flex; align-items: center; gap: 0.75rem;
    }
    .user-avatar { width: 40px; height: 40px; border-radius: 50%; background: #E0E0E0; }
</style>
""", unsafe_allow_html=True)

# Session state
if "page" not in st.session_state: st.session_state.page = "search"
if "system_ready" not in st.session_state: st.session_state.system_ready = False

# Auto-initialize
@st.cache_resource
def load_system():
    try:
        # Load data
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        
        # Model selection
        try:
            results_df = pd.read_csv("model_evaluation_summary.csv")
            if results_df["avg_rank"].notna().any():
                best_row = results_df.loc[results_df["avg_rank"].idxmin()]
                best_model = best_row["model"]
                best_metric = best_row.get("metric", "cosine")
            else:
                best_model = "all-MiniLM-L6-v2"
                best_metric = "cosine"
        except:
            best_model = "all-MiniLM-L6-v2"
            best_metric = "cosine"
        
        model = SentenceTransformer(best_model)
        embedding_columns = [col for col in df.columns if col.startswith("emb_")]
        combined_embeddings = df[embedding_columns].values
        df["thumbnail"] = df["video_id"].apply(lambda x: f"https://i.ytimg.com/vi/{x}/hqdefault.jpg")
        df["channel"] = "WatchMojo"
        
        st.session_state.model = model
        st.session_state.embeddings = combined_embeddings
        st.session_state.df = df
        st.session_state.metric = best_metric
        st.session_state.system_ready = True
        
        return True
    except Exception as e:
        st.error(f"Load failed: {e}")
        return False

# Auto-start
if not st.session_state.system_ready:
    with st.spinner("🚀 Initializing QueryTube..."):
        load_system()
    st.rerun()

# Search function (same logic)
def query_to_top5_videos(query):
    model = st.session_state.model
    embeddings = st.session_state.embeddings
    df = st.session_state.df
    metric = st.session_state.metric
    
    query_embedding_title = model.encode([query], show_progress_bar=False)
    query_embedding_transcript = np.zeros_like(query_embedding_title)
    query_embedding = np.hstack((query_embedding_title, query_embedding_transcript))
    
    if metric == "cosine":
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        scores = similarities
    else:
        distances = euclidean_distances(query_embedding, embeddings)[0]
        scores = 1 / (1 + distances)
    
    top_indices = np.argsort(scores)[::-1][:5]
    results = df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return results

def format_result_card(video):
    embed_url = f"https://www.youtube.com/embed/{video['video_id']}?rel=0&modestbranding=1"
    return f"""
    <div class='result_card'>
        <iframe src='{embed_url}' allowfullscreen></iframe>
        <div class='card-content'>
            <h3 class='card-title'>{video['title']}</h3>
            <p class='card-channel'><strong>Channel:</strong> {video['channel']}</p>
            <p class='card-channel'><strong>Score:</strong> {video['score']:.3f}</p>
        </div>
    </div>
    """

# === EXACT GRADIO UI STRUCTURE ===

# Header
header_html = """
<div class='header-main'>
    <div class='logo-text'>QueryTube</div>
    <div style='flex: 1; max-width: 500px; margin: 0 2rem;'>
        <input type='text' placeholder='Search...' style='
            background: #0B0F19; border: 1px solid #2A3B4B; border-radius: 8px;
            padding: 0.5rem 1rem; color: white; width: 100%;
        '>
    </div>
    <div style='display: flex; align-items: center; gap: 1rem;'>
        <span style='font-size: 1.2rem;'>🔔</span>
        <div style='width: 32px; height: 32px; background: #3B82F6; 
                    border-radius: 50%; color: white; display: flex; 
                    align-items: center; justify-content: center;'>U</div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Main Layout (Row = Sidebar + Content)
col1, col2 = st.columns([1, 4])

with col1:
    # Sidebar (EXACT Gradio)
    st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    
    # Navigation Buttons
    if st.button("🔍 Search", key="nav_search"):
        st.session_state.page = "search"
        st.rerun()
    st.button("📜 Search History", key="nav_history")
    st.button("⚙️ Settings", key="nav_settings")
    
    # User Profile (Bottom of sidebar)
    st.markdown("""
    <div class='user-profile'>
        <div class='user-avatar' style='
            width: 40px; height: 40px; background: #E0E0E0; 
            border-radius: 50%; display: flex; align-items: center; 
            justify-content: center; color: #0B0F19; font-weight: bold;
        '>JD</div>
        <div>
            <div style='font-weight: bold; color: #E0E0E0;'>John Doe</div>
            <div style='font-size: 0.8rem; color: #A0AEC0;'>johndoe@email.com</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Content Area
    if st.session_state.page == "search":
        # Search Page (EXACT Gradio)
        
        # Query + Search Button Row
        search_row = st.columns([5, 1])
        with search_row[0]:
            query = st.text_input("", placeholder="Semantic search for YouTube videos", 
                                label_visibility="collapsed")
        with search_row[1]:
            if st.button("🔍 Search", type="primary"):
                if query:
                    # Show skeletons
                    skeleton_cols = st.columns(5)
                    for col in skeleton_cols:
                        with col:
                            st.markdown("<div class='skeleton_box'></div>", unsafe_allow_html=True)
                    
                    with st.spinner("Searching..."):
                        time.sleep(1.5)
                        results = query_to_top5_videos(query)
                    
                    # Clear skeletons and show results
                    for col in st.columns(5):
                        with col:
                            st.empty()
                    
                    if not results.empty:
                        result_cols = st.columns(5)
                        for i, (_, video) in enumerate(results.iterrows()):
                            with result_cols[i % 5]:
                                st.markdown(format_result_card(video), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='text-align: center; padding: 3rem; color: #A0AEC0; 
                                   font-size: 1.2rem;'>😔 No videos found</div>
                        """, unsafe_allow_html=True)
        
        # Sort Buttons Row (EXACT Gradio)
        st.markdown("<div style='margin-bottom: 1.5rem;'><strong>Sort by:</strong></div>", unsafe_allow_html=True)
        sort_cols = st.columns(4)
        with sort_cols[0]: st.button("Relevance", disabled=True)
        with sort_cols[1]: st.button("Date", disabled=True)
        with sort_cols[2]: st.button("Duration", disabled=True)
        with sort_cols[3]: st.button("Views", disabled=True)
    
    elif st.session_state.page == "history":
        # History Page
        st.markdown("<h2 style='color: #E0E0E0;'>Search History</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: #121826; border: 1px solid #2A3B4B; border-radius: 8px; padding: 1rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
                <h3 style='color: #E0E0E0;'>Recent Searches</h3>
                <a href='#' style='color: #3B82F6;'>Clear all</a>
            </div>
            <div style='padding: 1rem 0; border-bottom: 1px solid #2A3B4B; display: flex; gap: 0.75rem;'>
                <span style='color: #A0AEC0;'>🔍</span>
                <div>
                    <p style='color: #E0E0E0; margin: 0;'>AI in education</p>
                    <p style='color: #A0AEC0; font-size: 0.85rem; margin: 0;'>July 14, 2024</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.page == "settings":
        # Settings Page
        st.markdown("<h2 style='color: #E0E0E0;'>Settings</h2>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #E0E0E0;'>Data Sources</h4>", unsafe_allow_html=True)
        st.text_input("YouTube Channels", placeholder="e.g., 'TechCrunch'")
        
        st.markdown("<strong>Active Sources:</strong>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap;'>
            <div style='background: #2A3B4B; color: #E0E0E0; padding: 0.5rem 1rem; 
                       border-radius: 20px; display: flex; align-items: center; gap: 0.5rem;'>
                WatchMojo <span style='color: #EF4444; cursor: pointer;'>&times;</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #E0E0E0;'>Semantic Search</h4>", unsafe_allow_html=True)
        st.slider("Search Sensitivity", 0.0, 1.0, 0.5)
        
        col1, col2 = st.columns(2)
        with col1: st.checkbox("Dark Mode", value=True)
        with col2: st.checkbox("Email Notifications")
        
        st.columns(2)[1].button("Save Changes", type="primary")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #A0AEC0;'>Powered by Sentence Transformers</p>", unsafe_allow_html=True)
