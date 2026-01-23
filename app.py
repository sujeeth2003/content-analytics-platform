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

    n_events = 80000
    user_ids = rng.integers(0, n_users, n_events)
    content_ids = rng.integers(0, n_content, n_events)
    watch_pct = np.clip(rng.beta(2, 1.5, n_events), 0.01, 1.0)
    days_ago = rng.integers(0, 90, n_events)

    events = pd.DataFrame({
        "user_id": user_ids,
        "content_id": content_ids,
        "watch_pct": watch_pct,
        "rating": np.where(rng.random(n_events) < 0.4,
                           rng.integers(1, 6, n_events).astype(float), np.nan),
        "event_date": [datetime.now() - timedelta(days=int(d)) for d in days_ago],
        "device": rng.choice(["TV","Mobile","Desktop","Tablet"], n_events, p=[0.45,0.3,0.18,0.07]),
    })
    return users, content, events

# ── SQLite metric layer ───────────────────────────────────────────────────────
@st.cache_resource
def build_db(users, content, events):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    users.to_sql("users", conn, index=False, if_exists="replace")
    content.to_sql("content", conn, index=False, if_exists="replace")
    events["event_date"] = events["event_date"].astype(str)
    events.to_sql("events", conn, index=False, if_exists="replace")

    conn.executescript("""
    CREATE VIEW IF NOT EXISTS user_metrics AS
    SELECT
        e.user_id,
        COUNT(DISTINCT e.content_id)                          AS titles_watched,
        AVG(e.watch_pct)                                      AS avg_completion,
        SUM(CASE WHEN e.watch_pct >= 0.85 THEN 1 ELSE 0 END) AS completed_titles,
        SUM(CASE WHEN e.watch_pct < 0.25  THEN 1 ELSE 0 END) AS dropped_titles,
        AVG(e.rating)                                         AS avg_rating,
        COUNT(DISTINCT DATE(e.event_date))                    AS active_days,
        MAX(e.event_date)                                     AS last_seen
    FROM events e
    GROUP BY e.user_id;

    CREATE VIEW IF NOT EXISTS genre_performance AS
    SELECT
        c.genre,
        COUNT(*)                    AS total_views,
        AVG(e.watch_pct)            AS avg_completion,
        AVG(e.rating)               AS avg_rating,
        COUNT(DISTINCT e.user_id)   AS unique_viewers
    FROM events e
    JOIN content c ON e.content_id = c.content_id
    GROUP BY c.genre
    ORDER BY total_views DESC;

    CREATE VIEW IF NOT EXISTS daily_activity AS
    SELECT
        DATE(event_date)            AS day,
        COUNT(*)                    AS total_views,
        COUNT(DISTINCT user_id)     AS dau,
        AVG(watch_pct)              AS avg_completion
    FROM events
    GROUP BY DATE(event_date)
    ORDER BY day;
    """)
    return conn

# ── Distributed pipeline (master/worker with threads + queue) ─────────────────
class PipelineMaster:
    def __init__(self, n_workers=4):
        self.n_workers = n_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.log = []
        self.lock = threading.Lock()

    def _worker(self, worker_id, tasks_fn):
        while True:
            try:
                task = self.task_queue.get(timeout=0.5)
                if task is None:
                    break
                t0 = time.time()
                result = tasks_fn(task)
                elapsed = round(time.time() - t0, 3)
                with self.lock:
                    self.log.append({
                        "worker": worker_id, "task": task["name"],
                        "status": "✅ done", "elapsed_s": elapsed,
                        "output": result
                    })
                self.result_queue.put(result)
                self.task_queue.task_done()
            except queue.Empty:
                break

    def run(self, tasks, tasks_fn):
        for t in tasks:
            self.task_queue.put(t)
        workers = [
            threading.Thread(target=self._worker, args=(f"W-{i+1}", tasks_fn), daemon=True)
            for i in range(self.n_workers)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results, self.log

def run_ml_pipeline(conn, events, users):
    """Full ML pipeline: feature eng → cohort → cluster → retention score"""
    metrics_df = pd.read_sql("SELECT * FROM user_metrics", conn)
    user_df = users.merge(metrics_df, on="user_id", how="left").fillna(0)

    # Cohort assignment
    user_df["days_since_join"] = (
        datetime.now() - pd.to_datetime(user_df["join_date"])
    ).dt.days
    user_df["cohort"] = pd.cut(
        user_df["days_since_join"],
        bins=[0, 30, 90, 365, 9999],
        labels=["New (0-30d)", "Growing (30-90d)", "Retained (90-365d)", "Veteran (365d+)"]
    )

    # Simple retention label: active in last 14 days
    user_df["last_seen"] = pd.to_datetime(user_df["last_seen"].replace(0, pd.NaT))
    user_df["is_retained"] = (
        (datetime.now() - user_df["last_seen"]).dt.days < 14
    ).astype(int)

    # Retention score (logistic-style from features)
    feats = ["avg_completion","completed_titles","active_days","titles_watched"]
    for f in feats:
        user_df[f] = pd.to_numeric(user_df[f], errors="coerce").fillna(0)
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    raw = user_df[feats].values
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0) + 1e-9
    normed = (raw - mins) / (maxs - mins)
    user_df["retention_score"] = np.clip((normed * weights).sum(axis=1), 0, 1)

    # K-Means segmentation (manual, no sklearn needed for demo)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    seg_feats = user_df[feats].values
    scaler = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_feats)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_df["segment"] = km.fit_predict(seg_scaled)
    seg_names = {0: "Power Viewers", 1: "Casual Browsers", 2: "Engaged Critics", 3: "At-Risk"}
    # remap by avg retention score per cluster
    cluster_score = user_df.groupby("segment")["retention_score"].mean().sort_values(ascending=False)
    remap = {old: list(seg_names.values())[i] for i, old in enumerate(cluster_score.index)}
    user_df["segment"] = user_df["segment"].map(remap)

    return user_df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Content Analytics")
    st.markdown("*Netflix-style analytics platform*")
    st.divider()
    st.markdown("**Architecture**")
    st.markdown("""
- 🗄️ SQL metric layer (SQLite views)
- ⚡ Distributed pipeline (4 workers)
- 🤖 ML retention scoring
- 📊 Self-service BI dashboard
    """)
    st.divider()
    n_users = st.slider("Simulated Users", 1000, 10000, 5000, 500)
    st.caption("Adjust to simulate scale")
    st.divider()
    st.markdown("**[GitHub Repo](https://github.com/sujeeth2003)**")
    st.markdown("**[Portfolio](https://sujeeth2003.github.io/Portfolio/)**")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    users, content, events = generate_synthetic_data(n_users=n_users)
    conn = build_db(users, content, events)
    user_df = run_ml_pipeline(conn, events, users)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎬 Content Analytics Platform")
st.markdown("*Scalable analytics infrastructure for content engagement, retention, and audience intelligence*")
st.divider()

# ── Top KPI row ───────────────────────────────────────────────────────────────
total_users = len(user_df)
retained = int(user_df["is_retained"].sum())
avg_completion = float(user_df["avg_completion"].mean())
avg_score = float(user_df["retention_score"].mean())

