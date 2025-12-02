# 🎬 Content Analytics Platform

A Netflix-style analytics engineering platform demonstrating scalable data infrastructure, distributed pipeline execution, SQL-backed metric layers, and ML-powered audience intelligence — built to mirror the architecture of production content analytics systems.

**[▶ Live Demo](https://your-streamlit-app-url)** | **[Portfolio](https://sujeeth2003.github.io/Portfolio/)**

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

