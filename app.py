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

