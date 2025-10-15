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

# Page config
st.set_page_config(page_title="QueryTube", layout="wide")

# Gradio-style CSS
st.markdown("""
<style>
    .main { background-color: #0B0F19; color: #E0E0E0; }
    .stApp { background-color: #0B0F19; }
    .header { 
        display: flex; justify-content: space-between; align-items: center;
        padding: 1rem; background-color: #121826; border-bottom: 1px solid #2A3B4B;
    }
    .logo { font-size: 2rem; font-weight: bold; color: #FFFFFF; }
    .search-header input { 
        background: #0B0F19; border: 1px solid #2A3B4B; border-radius: 8px; 
        padding: 0.75rem; color: white; width: 400px;
    }
    .user-menu { display: flex; align-items: center; gap: 1rem; color: #A0AEC0; }
    .sidebar { background-color: #121826; border: 1px solid #2A3B4B; border-radius: 12px; }
    .nav-btn { 
        background: transparent; color: #A0AEC0; text-align: left; border: none; 
        width: 100%; padding: 1rem; font-size: 1rem;
    }
    .nav-btn:hover { background-color: #2A3B4B; }
    .nav-btn.primary { background-color: #2563EB; color: white; }
    .result-card { 
        background: #121826; border: 1px solid #2A3B4B; border-radius: 8px; 
        padding: 1rem; margin: 1rem 0; height: 100%;
    }
    .result-card iframe { width: 100%; height: 180px; border: none; }
    .title { font-weight: 600; font-size: 1.1rem; color: #E0E0E0; margin-bottom: 0.5rem; }
    .channel { font-size: 0.9rem; color: #A0AEC0; }
    .skeleton { 
        height: 280px; background: linear-gradient(90deg, #2A3B4B 25%, #374151 50%, #2A3B4B 75%);
        background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 8px;
    }
    @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
    .stTextInput input { background-color: #121826; border: 1px solid #2A3B4B; color: #E0E0E0; }
    .status-bar { background: #121826; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Session state
if "page" not in st.session_state:
    st.session_state.page = "search"
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "metric" not in st.session_state:
    st.session_state.metric = "cosine"

# Auto-initialization function
@st.cache_resource
def load_system():
    """Auto-load system on startup"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        progress_bar.progress(10)
        status_text.text("🔍 Loading video index...")
        
        # Load data
        if not os.path.exists("video_index_with_embeddings.parquet"):
            raise FileNotFoundError("video_index_with_embeddings.parquet not found")
        
        df = pd.read_parquet("video_index_with_embeddings.parquet")
        progress_bar.progress(30)
        
        # Best model logic (same as Gradio)
        status_text.text("🎯 Selecting best model...")
        try:
            results_df = pd.read_csv("model_evaluation_summary.csv")
            if results_df["avg_rank"].notna().any():
                best_row = results_df.loc[results_df["avg_rank"].idxmin()]
                best_model_name = best_row["model"]
                best_metric = best_row.get("metric", "cosine")
                print(f"Using best model: {best_model_name}")
            else:
                best_model_name = "all-MiniLM-L6-v2"  # Smaller for faster load
                best_metric = "cosine"
        except FileNotFoundError:
            best_model_name = "all-MiniLM-L6-v2"
            best_metric = "cosine"
        
        progress_bar.progress(50)
        status_text.text(f"🤖 Loading model: {best_model_name}...")
        
        # Load model
        model = SentenceTransformer(best_model_name)
        
        # Prepare embeddings
        embedding_columns = [col for col in df.columns if col.startswith("emb_")]
        if not embedding_columns:
            raise ValueError("No embedding columns found")
        
        combined_embeddings = df[embedding_columns].values
        df["thumbnail"] = df["video_id"].apply(lambda x: f"https://i.ytimg.com/vi/{x}/hqdefault.jpg")
        df["channel"] = "WatchMojo"
        
        progress_bar.progress(90)
        status_text.text("✅ System ready!")
        
        # Store in session
        st.session_state.model = model
        st.session_state.embeddings = combined_embeddings
        st.session_state.df = df
        st.session_state.metric = best_metric
        st.session_state.system_ready = True
        
        progress_bar.progress(100)
        time.sleep(0.5)  # Show completion
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Auto-init failed: {str(e)}")
        st.info("Required files: video_index_with_embeddings.parquet")
        progress_bar.empty()
        status_text.empty()
        return False

# AUTO-INITIALIZE ON STARTUP
if not st.session_state.system_ready:
    st.markdown("""
    <div class='status-bar'>
        <h3 style='color: #E0E0E0; margin: 0;'>🚀 Initializing QueryTube...</h3>
        <p style='color: #A0AEC0;'>Loading AI model and video index automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    load_system()
    st.rerun()

# Header (Gradio-style)
st.markdown("""
<div class='header'>
    <div class='logo'>🎥 QueryTube</div>
    <input class='search-header' type='text' placeholder='Search videos...'>
    <div class='user-menu'>
        <span>🔔</span>
        <div style='width: 32px; height: 32px; background: #3B82F6; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;'>U</div>
    </div>
</div>
""", unsafe_allow_html=True)

# System status indicator
st.markdown(f"""
<div class='status-bar'>
    <span style='color: #10B981;'>✅ Ready</span> | 
    Model: {st.session_state.model[0].__class__.__name__} | 
    Videos: {len(st.session_state.df):,} | 
    <span style='color: #3B82F6;'>{st.session_state.metric.upper()} Metric</span>
</div>
""", unsafe_allow_html=True)

def query_to_top5_videos(query, top_k=5, metric="cosine"):
    """Same exact logic from Gradio code"""
    try:
        if not query.strip():
            return pd.DataFrame(columns=["video_id", "title", "score", "thumbnail", "channel"])
        
        model = st.session_state.model
        combined_embeddings = st.session_state.embeddings
        
        query_embedding_title = model.encode([query], show_progress_bar=False)
        query_embedding_transcript = np.zeros_like(query_embedding_title)
        query_embedding = np.hstack((query_embedding_title, query_embedding_transcript))
        
        if metric == "cosine":
            similarities = cosine_similarity(query_embedding, combined_embeddings)[0]
            scores = similarities
        elif metric == "euclidean":
            distances = euclidean_distances(query_embedding, combined_embeddings)[0]
            scores = 1 / (1 + distances)
        
        top_k_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        top_k_scores = scores[top_k_indices]
        
        df = st.session_state.df
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

def format_result_card(video):
    """Same HTML formatting as Gradio"""
    embed_url = f"https://www.youtube.com/embed/{video['video_id']}?rel=0&modestbranding=1&iv_load_policy=3&autoplay=0"
    return f"""
    <div class='result_card'>
        <iframe src='{embed_url}' allowfullscreen></iframe>
        <div style='padding: 1rem;'>
            <h3 class='title'>{video["title"]}</h3>
            <p class='channel'><strong>Channel:</strong> {video["channel"]}</p>
            <p class='channel'><strong>Relevance Score:</strong> {video["score"]:.3f}</p>
            <p style='color: #A0AEC0; font-style: italic;'>*Click play to watch*</p>
        </div>
    </div>
    """

# Sidebar Navigation (Gradio-style)
with st.sidebar:
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #E0E0E0;'>Navigation</h3>", unsafe_allow_html=True)
    
    if st.button("🔍 Search", key="nav_search"):
        st.session_state.page = "search"
        st.rerun()
    if st.button("📜 History", key="nav_history"):
        st.session_state.page = "history"
        st.rerun()
    if st.button("⚙️ Settings", key="nav_settings"):
        st.session_state.page = "settings"
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 0.75rem; padding-top: 1rem; border-top: 1px solid #2A3B4B;'>
        <div style='width: 40px; height: 40px; background: #E0E0E0; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #0B0F19;'>JD</div>
        <div>
            <p style='font-weight: bold; color: #E0E0E0; margin: 0;'>John Doe</p>
            <p style='font-size: 0.8rem; color: #A0AEC0; margin: 0;'>johndoe@email.com</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Page Content
if st.session_state.page == "search":
    # Search Page
    st.markdown("<h2 style='color: #E0E0E0;'>Search Videos</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("", placeholder="Semantic search for YouTube videos", key="search_query")
    with col2:
        if st.button("Search", type="primary"):
            if query:
                # Loading skeleton
                skeleton_cols = st.columns(5)
                for col in skeleton_cols:
                    with col:
                        st.markdown('<div class="skeleton"></div>', unsafe_allow_html=True)
                
                with st.spinner("Searching..."):
                    time.sleep(1.5)
                    results = query_to_top5_videos(query, metric=st.session_state.metric)
                
                # Clear skeletons
                for col in st.columns(5):
                    with col:
                        st.empty()
                
                if not results.empty:
                    st.success(f"✅ Found {len(results)} results!")
                    result_cols = st.columns(5)
                    for i, (_, video) in enumerate(results.iterrows()):
                        with result_cols[i % 5]:
                            st.markdown(format_result_card(video), unsafe_allow_html=True)
                            
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button("💾 Download Results", csv, "search_results.csv", "text/csv")
                else:
                    st.markdown("""
                    <div style='text-align: center; padding: 3rem; color: #A0AEC0;'>
                        😔 No videos found. Try another search!
                    </div>
                    """, unsafe_allow_html=True)
    
    # Sort buttons (placeholder)
    st.markdown("<strong>Sort by:</strong>", unsafe_allow_html=True)
    col_sort1, col_sort2, col_sort3, col_sort4 = st.columns(4)
    with col_sort1: st.button("Relevance", disabled=True)
    with col_sort2: st.button("Date", disabled=True)
    with col_sort3: st.button("Duration", disabled=True)
    with col_sort4: st.button("Views", disabled=True)

elif st.session_state.page == "history":
    st.markdown("<h2 style='color: #E0E0E0;'>Search History</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #121826; border: 1px solid #2A3B4B; border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
            <h3 style='color: #E0E0E0; margin: 0;'>Recent Searches</h3>
            <a href='#' style='color: #3B82F6;'>Clear all</a>
        </div>
        <div style='padding: 1rem 0; border-bottom: 1px solid #2A3B4B; display: flex; align-items: center; gap: 0.75rem;'>
            <div style='color: #A0A0A0;'>🔍</div>
            <div>
                <p style='color: #E0E0E0; font-weight: 500; margin: 0;'>AI in education</p>
                <p style='font-size: 0.85rem; color: #A0AEC0; margin: 0;'>July 14, 2024</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "settings":
    st.markdown("<h2 style='color: #E0E0E0;'>Settings</h2>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #E0E0E0;'>Data Sources</h4>", unsafe_allow_html=True)
    st.text_input("YouTube Channels", placeholder="e.g., 'TechCrunch', 'Science & Technology'")
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1rem;'>
        <div style='background: #2A3B4B; color: #E0E0E0; padding: 0.5rem 1rem; border-radius: 20px; display: flex; align-items: center; gap: 0.5rem;'>
            WatchMojo <span style='cursor: pointer; color: #EF4444;'>&times;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #E0E0E0;'>Semantic Search</h4>", unsafe_allow_html=True)
    st.slider("Search Sensitivity", 0.0, 1.0, 0.5, 0.1)
    
    col1, col2 = st.columns(2)
    with col1: st.checkbox("Dark Mode", value=True)
    with col2: st.checkbox("Email Notifications", value=False)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1: st.button("Cancel")
    with col_btn2: st.button("Save Changes", type="primary")

# Footer
st.markdown("---")
st.markdown("*Powered by Sentence Transformers & Streamlit*")
