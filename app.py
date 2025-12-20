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
