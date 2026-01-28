"""
Interactive Dashboard Page
===========================
Handles dynamic filtering, KPI calculation, and visual regeneration 
using the Python logic generated in the App page.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pathlib import Path
import json
import os
import re
import warnings
import sys
import io
import math

# ==================== CONFIGURATION ====================
# Suppress warnings for a clean UI
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Interactive Dashboard", layout="wide", page_icon="📊")

# ==================== SETUP & PATHS ====================
TEMP_CHARTS_DIR = Path("temp_charts")
TEMP_DATA_DIR = Path("temp_data")
DATA_PATH = TEMP_DATA_DIR / "source.csv"
CONFIG_PATH = TEMP_CHARTS_DIR / "kpi_config.json"
DASHBOARD_HTML = TEMP_CHARTS_DIR / "interactive_dashboard.html"

# Ensure directories exist
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def load_data():
    """Load the source data from session state or disk with encoding fallback."""
    if st.session_state.get("dataframe") is not None:
        return st.session_state.dataframe
    
    if DATA_PATH.exists():
        try: 
            return pd.read_csv(DATA_PATH, encoding='utf-8')
        except: 
            return pd.read_csv(DATA_PATH, encoding='latin1')
    return None

def calculate_kpi(df, formula, agg):
    """
    Calculates KPI values dynamically from the formulas provided in config.
    Supports simple column names and complex math expressions like 'price * quantity'.
    """
    if df is None or df.empty: 
        return 0
        
    try:
        col_map = {c.lower(): c for c in df.columns}
        clean_formula = str(formula).strip().lower()
        
        # 1. Try direct column access
        actual_col = col_map.get(clean_formula)
        if actual_col:
            series = df[actual_col]
        else:
            # 2. Fallback to evaluating the formula (e.g., "price * quantity")
            # Only allow standard alphanumeric and math operators for security
            if re.match(r'^[a-zA-Z0-9_\s\+\-\*\/\(\)\.]+$', str(formula)):
                series = df.eval(formula)
            else:
                return 0
        
        # 3. Handle Aggregations
        agg = str(agg).lower()
        if agg in ["count_distinct", "nunique", "distinct"]: 
            return series.nunique()
        if agg == "count": 
            return len(series)
        
        # Force numeric for math-based aggregations
        s_num = pd.to_numeric(series, errors='coerce').fillna(0)
        if agg == "mean": 
            return s_num.mean()
        if agg == "min": 
            return s_num.min()
        if agg == "max": 
            return s_num.max()
            
        return s_num.sum() # Default to sum
    except: 
        return 0

def currency_fmt_func(x, pos):
    """Formatted axis labels showing K (Thousands) or M (Millions)."""
    if x >= 1e6: return f'${x*1e-6:.1f}M'
    if x >= 1e3: return f'${x*1e-3:.0f}K'
    return f'${x:.0f}'

def rerun_visuals(df_filtered):
    """
    Re-executes the Python analysis code using the FILTERED dataset.
    This ensures Plotly charts update based on sidebar selections.
    """
    if not st.session_state.get("last_analysis_code"):
        return False
    
    code = st.session_state.last_analysis_code
    
    # Execution context mirroring app.py
    # We pass df_filtered as 'df' so the code operates on the current subset
    local_vars = {
        'df': df_filtered,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'go': go,
        'px': px,
        'make_subplots': make_subplots,
        'json': json,
        'Path': Path,
        'FuncFormatter': FuncFormatter,
        're': re,
        'os': os,
        'math': math,
        'currency_fmt': FuncFormatter(currency_fmt_func),
        'TEMP_CHARTS_DIR': TEMP_CHARTS_DIR,
        'glob': __import__('glob')
    }
    
    # Capture output to suppress technical noise
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        exec(code, local_vars)
        sys.stdout = old_stdout
        return DASHBOARD_HTML.exists()
    except Exception as e:
        sys.stdout = old_stdout
        # st.error(f"Error updating visuals: {e}") # Debugging
        return False

# ==================== MAIN DASHBOARD LOGIC ====================

st.header("📊 Interactive Analytics Dashboard")

# Robust readiness check
is_ready = st.session_state.get('dashboard_ready') or DASHBOARD_HTML.exists()
df = load_data()

if is_ready and df is not None:
    
    # 1. LOAD KPI CONFIGURATION
    config = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except:
            st.error("⚠️ Failed to parse KPI configuration.")
            
    if not config:
        st.warning("⚠️ No configuration found. Please run an analysis in the App page first.")
        st.stop()

    st.markdown("---")
    
    # 2. SIDEBAR SLICERS (FILTERS)
    st.sidebar.header("🔍 Dynamic Filters")
    df_filtered = df.copy()
    active_filters = {}
    
    slicers = config.get("slicers", [])
    
    for col_raw in slicers:
        # Case-insensitive matching for robust column handling
        col_map = {c.lower(): c for c in df.columns}
        col = col_map.get(str(col_raw).lower())
        
        if col and col in df.columns:
            # Sort unique values for cleaner dropdowns
            try:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
            except:
                options = df[col].dropna().astype(str).unique().tolist()
                
            selected = st.sidebar.multiselect(
                f"📍 {col.replace('_', ' ').title()}", 
                options,
                key=f"filter_{col}"
            )
            
            if selected:
                active_filters[col] = selected
                df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected)]

    # 3. KPI METRICS (TOP ROW)
    st.markdown("### 📈 Dashboard KPIs")
    kpis = config.get("kpis", [])
    
    if kpis:
        # Create dynamic column grid (max 4 KPIs per row)
        cols = st.columns(min(len(kpis), 4))
        for idx, kpi in enumerate(kpis[:4]):
            val = calculate_kpi(df_filtered, kpi.get("formula"), kpi.get("agg", "sum"))
            fmt = kpi.get("fmt", "num")
            label = kpi.get("label", "Metric")
            
            # Format the metric display
            if isinstance(val, (int, float)):
                if fmt == "$":
                    disp = f"${val:,.0f}" if val % 1 == 0 else f"${val:,.2f}"
                elif fmt == "%":
                    disp = f"{val:.1f}%"
                else:
                    disp = f"{val:,.0f}" if val % 1 == 0 else f"{val:,.2f}"
            else:
                disp = str(val)
            
            cols[idx].metric(label, disp)
    else:
        st.info("ℹ️ No KPIs available in config.")

    # 4. INTERACTIVE VISUAL ANALYSIS
    st.markdown("---")
    st.subheader("📊 Dynamic Visual Insights")
    
    # Filter Change Hash Logic (prevents unnecessary re-runs)
    current_filter_hash = json.dumps(active_filters, sort_keys=True)
    if "last_filter_hash" not in st.session_state:
        st.session_state.last_filter_hash = ""
    
    # Trigger visual update only if filters changed
    if current_filter_hash != st.session_state.last_filter_hash:
        with st.spinner("🔄 Recalculating charts..."):
            success = rerun_visuals(df_filtered)
            if success:
                st.session_state.last_filter_hash = current_filter_hash
    
    # Render the Plotly Dashboard HTML
    if DASHBOARD_HTML.exists():
        try:
            with open(DASHBOARD_HTML, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Error loading visualization components: {e}")
    else:
        st.error("❌ dashboard file missing. Please re-run the analysis in the App.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    AI Data Analyst Agent | Dashboard Engine
    </div>
    """, unsafe_allow_html=True)

else:
    # No data loaded state
    st.warning("⚠️ No analysis data available.")
    st.markdown("""
    ### Getting Started:
    1. Navigate to the **App** page.
    2. Upload your CSV dataset.
    3. Ask an analysis question to generate the dashboard logic.
    4. Explore your interactive insights here.
    """)