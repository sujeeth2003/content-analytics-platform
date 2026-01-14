# 🎬 Content Analytics Platform

A Netflix-style analytics engineering platform demonstrating scalable data infrastructure, distributed pipeline execution, SQL-backed metric layers, and ML-powered audience intelligence — built to mirror the architecture of production content analytics systems.

**[▶ Live Demo](https://content-analytics-platform.streamlit.app/)** | **[Portfolio](https://sujeeth2003.github.io/Portfolio/)**

---

## What This Platform Does

Raw viewing events (80k+ interactions across 5k users × 500 titles) flow through a full analytics engineering stack:

```
Raw Events (80k interactions)
        │
        ▼
┌─────────────────────────────────────┐
│  Distributed Ingest (4 workers)     │  Master/worker pipeline, parallel ETL
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  SQL Metric Layer (SQLite Views)    │  Reusable views: user_metrics,
│                                     │  genre_performance, daily_activity
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  ML Pipeline                        │  Cohort analysis, K-Means segmentation,
│                                     │  retention scoring (LightGBM-backed)
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Self-Service BI Dashboard          │  Streamlit — 5 interactive tabs,
│                                     │  live SQL query runner, pipeline demo
└─────────────────────────────────────┘
```

---

## Architecture Highlights

### 1. Distributed Pipeline Engine
- **Master/worker pattern** using Python `threading` + `queue`
- Master partitions tasks into a thread-safe queue; 4 workers consume independently
- Results aggregated back through result queue with structured execution logs
- Demonstrates horizontal scaling concept: more workers → more throughput

### 2. SQL Metric Layer
Three reusable SQL views that standardize metrics across all downstream analysis:

| View | Purpose |
|------|---------|
| `user_metrics` | Per-user behavioral summary (completion, active days, drop rate) |
| `genre_performance` | Content-level rollup for editorial decisions |
| `daily_activity` | DAU, total views, avg completion over time |

Any analyst can query these views without touching raw tables — this is the foundation of self-service analytics infrastructure.

### 3. ML Audience Intelligence
- **Cohort analysis**: users segmented by lifecycle stage (New / Growing / Retained / Veteran)
- **K-Means segmentation** (K=4): behavioral personas — Power Viewers, Engaged Critics, Casual Browsers, At-Risk
- **Retention scoring**: normalized feature-weighted score (completion rate, active days, titles watched) exposed as reusable pipeline output

### 4. Self-Service BI Dashboard (Streamlit)
Five interactive tabs:
- 📊 **Engagement Overview** — DAU trends, cohort retention bar chart
- 🎯 **Audience Segments** — Donut chart, segment profiles, violin distributions
- 🎬 **Content Performance** — Genre views, completion rates, device mix
- ⚡ **Pipeline Engine** — Live pipeline execution demo with worker log
- 🗄️ **SQL Metric Layer** — Interactive SQL query runner on live data

---

## Running Locally

```bash
# Clone
git clone https://github.com/sujeeth2003/content-analytics-platform.git
cd content-analytics-platform

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

The app generates synthetic data automatically — no dataset download needed.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Engineering | Python, Pandas, SQLite (in-memory) |
| SQL Layer | SQLite views, window functions, multi-table joins |
| Distributed Execution | Python threading, queue (master/worker) |
| ML | Scikit-learn (KMeans), feature engineering, retention scoring |
| Visualization | Plotly, Streamlit |
| Deployment | Streamlit Cloud (one-click) |

---

## Deploying to Streamlit Cloud (Free, 5 minutes)

1. Fork this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py` → Deploy

Your live URL will be: `https://your-username-content-analytics-platform-app-xxxxx.streamlit.app`

---

## Key Design Decisions

**Why SQL views instead of computed DataFrames?**
Views are version-controlled, reusable definitions. Any new analysis starts from the same metric definitions — no inconsistency between teams. This mirrors how production analytics teams standardize their metric layer.

**Why master/worker over simple `map()`?**
The pattern demonstrates explicit task distribution, worker health monitoring, and result aggregation — the conceptual foundation of distributed systems like Spark executors or Airflow workers. It's the right mental model even when threads are the implementation.

**Why synthetic data?**
The pipeline architecture is what matters — not the specific dataset. Synthetic data lets reviewers run the demo instantly without credential setup, and the schema mirrors real streaming platform event logs.

---

## Author

**Sujeeth Sukumar** — M.S. Data Science, University of Maryland  
[Portfolio](https://sujeeth2003.github.io/Portfolio/) · [LinkedIn](https://www.linkedin.com/in/sujeeth73/) · [GitHub](https://github.com/sujeeth2003)
