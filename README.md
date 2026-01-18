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

