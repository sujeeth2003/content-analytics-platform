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

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total_users:,}</div>
        <div class="metric-label">Total Members</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{retained:,}</div>
        <div class="metric-label">Retained (14d)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_completion:.0%}</div>
        <div class="metric-label">Avg Completion Rate</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_score:.2f}</div>
        <div class="metric-label">Avg Retention Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Engagement Overview",
    "🎯 Audience Segments",
    "🎬 Content Performance",
    "⚡ Pipeline Engine",
    "🗄️ SQL Metric Layer",
])

# ─────────────────────────── TAB 1: Engagement ───────────────────────────────
with tab1:
    st.markdown("### Daily Active Users & Watch Trends")
    daily = pd.read_sql("SELECT * FROM daily_activity ORDER BY day", conn)
    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.tail(60)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Daily Active Users (DAU)", "Avg Content Completion Rate"),
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["dau"],
                             fill="tozeroy", line=dict(color="#e50914", width=2),
                             fillcolor="rgba(229,9,20,0.15)", name="DAU"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["avg_completion"],
                             line=dict(color="#46d369", width=2), name="Completion"), row=2, col=1)
    fig.update_layout(height=400, paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
                      font=dict(color="#e5e5e5"), showlegend=False,
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.update_xaxes(gridcolor="#333", zeroline=False)
    fig.update_yaxes(gridcolor="#333", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cohort Retention Analysis")
    cohort_summary = user_df.groupby("cohort").agg(
        users=("user_id", "count"),
        retention_rate=("is_retained", "mean"),
        avg_completion=("avg_completion", "mean"),
        avg_active_days=("active_days", "mean")
    ).reset_index()

    fig2 = px.bar(cohort_summary, x="cohort", y="retention_rate",
                  color="retention_rate", color_continuous_scale=["#e50914","#f5c518","#46d369"],
                  labels={"retention_rate": "Retention Rate", "cohort": "User Cohort"},
                  text=cohort_summary["retention_rate"].apply(lambda x: f"{x:.0%}"))
    fig2.update_layout(height=300, paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
                       font=dict(color="#e5e5e5"), coloraxis_showscale=False,
                       margin=dict(l=0, r=0, t=10, b=0))
    fig2.update_traces(textposition="outside")
    fig2.update_xaxes(gridcolor="#333")
    fig2.update_yaxes(gridcolor="#333", tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────── TAB 2: Segments ─────────────────────────────────
with tab2:
    st.markdown("### Audience Segmentation (K-Means, K=4)")
    st.caption("Users clustered by behavioral signals: completion rate, active days, titles watched, engagement depth")

    col1, col2 = st.columns([1, 1])
    with col1:
        seg_counts = user_df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        colors = {"Power Viewers": "#e50914", "Engaged Critics": "#f5c518",
                  "Casual Browsers": "#46d369", "At-Risk": "#888"}
        fig3 = px.pie(seg_counts, names="segment", values="count",
                      color="segment", color_discrete_map=colors,
                      hole=0.5)
        fig3.update_layout(height=320, paper_bgcolor="#141414",
                           font=dict(color="#e5e5e5"), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        seg_profile = user_df.groupby("segment").agg(
            avg_completion=("avg_completion","mean"),
            avg_active_days=("active_days","mean"),
            avg_titles=("titles_watched","mean"),
            retention_score=("retention_score","mean"),
        ).reset_index()
        fig4 = px.bar(seg_profile.melt(id_vars="segment"),
                      x="variable", y="value", color="segment",
                      barmode="group",
                      color_discrete_map=colors,
                      labels={"variable":"Metric","value":"Score","segment":"Segment"})
        fig4.update_layout(height=320, paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
                           font=dict(color="#e5e5e5"), margin=dict(l=0,r=0,t=10,b=0))
        fig4.update_xaxes(gridcolor="#333", tickvals=["avg_completion","avg_active_days","avg_titles","retention_score"],
                          ticktext=["Completion","Active Days","Titles","Retention"])
        fig4.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Retention Score Distribution by Segment")
    fig5 = px.violin(user_df, x="segment", y="retention_score",
                     color="segment", color_discrete_map=colors,
                     box=True, points=False)
    fig5.update_layout(height=320, paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
                       font=dict(color="#e5e5e5"), showlegend=False,
                       margin=dict(l=0,r=0,t=10,b=0))
    fig5.update_xaxes(gridcolor="#333")
    fig5.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────── TAB 3: Content ──────────────────────────────────
with tab3:
    st.markdown("### Genre Performance")
    genre_df = pd.read_sql("SELECT * FROM genre_performance", conn)

    fig6 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Views by Genre", "Avg Completion by Genre"))
    fig6.add_trace(go.Bar(x=genre_df["genre"], y=genre_df["total_views"],
                          marker_color="#e50914", name="Views"), row=1, col=1)
    fig6.add_trace(go.Bar(x=genre_df["genre"], y=genre_df["avg_completion"],
                          marker_color="#46d369", name="Completion"), row=1, col=2)
    fig6.update_layout(height=320, paper_bgcolor="#141414", plot_bgcolor="#1a1a1a",
                       font=dict(color="#e5e5e5"), showlegend=False,
                       margin=dict(l=0,r=0,t=40,b=0))
    fig6.update_xaxes(gridcolor="#333", tickangle=30)
    fig6.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### Device Mix")
    device_df = events.groupby("device").size().reset_index(name="views")
    fig7 = px.pie(device_df, names="device", values="views",
                  color_discrete_sequence=["#e50914","#f5c518","#46d369","#1f8ef1"],
                  hole=0.4)
    fig7.update_layout(height=280, paper_bgcolor="#141414",
                       font=dict(color="#e5e5e5"), margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig7, use_container_width=True)

# ─────────────────────────── TAB 4: Pipeline ─────────────────────────────────
with tab4:
    st.markdown("### ⚡ Distributed Pipeline Engine")
    st.markdown("""
The platform uses a **master/worker architecture** to process analytics tasks in parallel.
The master distributes tasks across worker threads; each worker processes independently and reports results back.
This pattern scales horizontally — add more workers to handle larger data volumes.
    """)

    if st.button("▶ Run Pipeline (Live Demo)", type="primary"):
        tasks = [
            {"name": "ingest::events_table",    "partition": "all",   "rows": len(events)},
            {"name": "transform::user_metrics", "partition": "users", "rows": len(users)},
            {"name": "transform::genre_rollup", "partition": "content","rows": len(content)},
            {"name": "score::retention_model",  "partition": "ml",    "rows": len(user_df)},
            {"name": "export::dashboard_views", "partition": "bi",    "rows": 3},
            {"name": "validate::schema_check",  "partition": "qa",    "rows": 0},
        ]

        def process_task(task):
            time.sleep(np.random.uniform(0.1, 0.5))
            return {"task": task["name"], "rows_processed": task["rows"], "status": "ok"}

        progress = st.progress(0, text="Initializing master...")
        log_box = st.empty()

        master = PipelineMaster(n_workers=4)
        results, logs = master.run(tasks, process_task)

        for i, log in enumerate(logs):
            progress.progress((i+1)/len(logs), text=f"Processing: {log['task']}")
            time.sleep(0.05)

        progress.progress(1.0, text="✅ Pipeline complete")

        st.markdown("#### Worker Execution Log")
        log_df = pd.DataFrame(logs)[["worker","task","status","elapsed_s"]]
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        total_rows = sum(r["rows_processed"] for r in results)
        workers_used = len(set(l["worker"] for l in logs))
        total_time = sum(l["elapsed_s"] for l in logs)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tasks Completed", len(results))
        c2.metric("Workers Used", workers_used)
        c3.metric("Total Wall Time (parallel)", f"{max(l['elapsed_s'] for l in logs):.2f}s")

    st.markdown("#### Architecture Diagram")
    st.code("""
  ┌──────────────────────────────────────────────────┐
  │                  MASTER NODE                     │
  │   - Receives pipeline DAG                        │
  │   - Partitions tasks into queue                  │
  │   - Monitors worker health & collects results    │
  └────────────┬─────────────────────────────────────┘
               │  task_queue (thread-safe)
       ┌───────┴────────┐
       ▼                ▼
  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
  │Worker-1 │      │Worker-2 │      │Worker-3 │      │Worker-4 │
  │ingest   │      │transform│      │score    │      │validate │
  └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘
       └────────────────┴────────────────┴────────────────┘
                              │  result_queue
                              ▼
                    ┌─────────────────┐
                    │  SQL Metric DB  │
                    │  (SQLite views) │
                    └─────────────────┘
    """, language="text")

# ─────────────────────────── TAB 5: SQL Layer ────────────────────────────────
with tab5:
    st.markdown("### 🗄️ SQL Metric Layer")
    st.markdown("""
All metrics are computed via **SQL views** — reusable, version-controlled definitions
that any analyst can query without touching raw tables. This is the foundation of
scalable self-service analytics infrastructure.
    """)

    queries = {
        "User Metrics View": "SELECT * FROM user_metrics LIMIT 10",
        "Genre Performance": "SELECT * FROM genre_performance",
        "Daily Activity (last 14 days)": """
            SELECT * FROM daily_activity
            WHERE day >= DATE('now', '-14 days')
            ORDER BY day DESC
        """,
        "Top Retained Users": """
            SELECT u.user_id, u.plan, u.country,
                   m.avg_completion, m.active_days, m.titles_watched
            FROM user_metrics m
            JOIN users u ON m.user_id = u.user_id
            ORDER BY m.active_days DESC
            LIMIT 10
        """,
    }

    selected = st.selectbox("Select a query to run:", list(queries.keys()))
    st.code(queries[selected].strip(), language="sql")

    if st.button("▶ Execute Query"):
        result = pd.read_sql(queries[selected], conn)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.caption(f"{len(result)} rows returned")

    st.markdown("#### View Definitions")
    st.code("""
-- user_metrics: per-user behavioral summary (reusable across all downstream analysis)
CREATE VIEW user_metrics AS
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

-- genre_performance: content-level rollup for editorial decisions
CREATE VIEW genre_performance AS
SELECT c.genre,
       COUNT(*)                  AS total_views,
       AVG(e.watch_pct)          AS avg_completion,
       AVG(e.rating)             AS avg_rating,
       COUNT(DISTINCT e.user_id) AS unique_viewers
FROM events e
JOIN content c ON e.content_id = c.content_id
GROUP BY c.genre;
    """, language="sql")
