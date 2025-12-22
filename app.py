"""
Content Analytics Platform — Demo App
A Netflix-style analytics dashboard demonstrating:
  - Distributed pipeline execution (master/worker pattern)
  - SQL-backed metric layer (Database Systems)
  - ML-powered retention scoring & segmentation (ML/DS)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
import threading
import queue
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Content Analytics Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e0e0e; color: #e5e5e5; }
    .stApp { background-color: #141414; }
    .metric-card {
        background: #1f1f1f;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e50914; }
    .metric-label { font-size: 0.85rem; color: #999; margin-top: 4px; }
    .pipeline-step {
        background: #1f1f1f;
        border-left: 3px solid #e50914;
        padding: 8px 14px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .pipeline-step.done { border-left-color: #46d369; }
    .pipeline-step.running { border-left-color: #f5c518; }
    .worker-box {
        background: #1a1a2e;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 10px;
        margin: 4px;
        font-family: monospace;
        font-size: 0.78rem;
    }
    h1, h2, h3 { color: #e5e5e5 !important; }
    .stTabs [data-baseweb="tab"] { color: #999; }
    .stTabs [aria-selected="true"] { color: #e50914 !important; border-bottom-color: #e50914 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data generation (synthetic, realistic) ───────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_users=5000, n_content=500, seed=42):
    rng = np.random.default_rng(seed)
    genres = ["Drama", "Action", "Comedy", "Thriller", "Sci-Fi", "Romance", "Documentary", "Horror"]
    content = pd.DataFrame({
        "content_id": range(n_content),
        "title": [f"Title_{i}" for i in range(n_content)],
        "genre": rng.choice(genres, n_content),
        "release_year": rng.integers(2015, 2025, n_content),
        "duration_min": rng.integers(20, 180, n_content),
    })

    join_days_ago = rng.integers(1, 730, n_users)
    users = pd.DataFrame({
        "user_id": range(n_users),
        "join_date": [datetime.now() - timedelta(days=int(d)) for d in join_days_ago],
        "country": rng.choice(["US","UK","IN","BR","DE","JP","FR","CA"], n_users),
        "plan": rng.choice(["Standard","Premium","Basic"], n_users, p=[0.5,0.35,0.15]),
    })

