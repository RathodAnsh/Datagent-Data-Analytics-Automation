import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import warnings
import io
import sys
import shutil
import glob
from pathlib import Path
import json
import math
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
from matplotlib.ticker import FuncFormatter

# ==================== CONFIGURATION ====================
# Suppress warnings and configure Streamlit defaults
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
load_dotenv()

# ==================== SESSION STATE INITIALIZATION ====================
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "profile_data" not in st.session_state:
    st.session_state.profile_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dashboard_ready" not in st.session_state:
    st.session_state.dashboard_ready = False
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = os.getenv("GOOGLE_API_KEY", "")
if "last_analysis_code" not in st.session_state:
    st.session_state.last_analysis_code = None
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False

# ==================== SETUP DIRECTORIES ====================
TEMP_CHARTS_DIR = Path("temp_charts")
TEMP_DATA_DIR = Path("temp_data")
PROFILES_DIR = Path("data_profiles")

# Initialize directories only once per session (NOT on every page load)
if not st.session_state.session_initialized:
    for d in [TEMP_CHARTS_DIR, TEMP_DATA_DIR, PROFILES_DIR]:
        if d.exists():
            try:
                shutil.rmtree(d)
            except:
                pass
        d.mkdir(parents=True, exist_ok=True)
    st.session_state.session_initialized = True
else:
    # Ensure directories exist but DON'T delete them
    for d in [TEMP_CHARTS_DIR, TEMP_DATA_DIR, PROFILES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide", page_icon="🤖")
st.title("🤖 AI Data Analyst Agent (Python Edition)")
st.markdown("""
**System Workflow:**
1. **Profile:** Auto-cleans data (Smart Detection) & saves JSON profile.
2. **Analysis:** AI writes robust Python code to generate Insights & Charts based on your questions.
3. **Dual Output:** Generates a **Static Report** in chat and an **Interactive Dashboard**.
""")

# ==================== SIDEBAR & API KEY ====================
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Google API Key", 
    value=st.session_state.api_key_saved, 
    type="password"
)

if not api_key:
    st.error("❌ Please provide a Google API Key in the sidebar.")
    st.stop()

st.session_state.api_key_saved = api_key
genai.configure(api_key=api_key)

# ==================== HELPER FUNCTIONS: CLEANING & PROFILING ====================

def clean_column_names(df):
    """Normalize column names to snake_case for consistent coding."""
    df.columns = [
        re.sub(r'[^a-zA-Z0-9]', '_', str(c).strip().lower()).strip('_')
        for c in df.columns
    ]
    return df

def convert_smart_types(df):
    """
    Intelligently converts data types.
    Handles mixed date separators, currency, and already-clean data.
    More robust for different retail sales datasets.
    """
    conversions = []
    
    for col in df.columns:
        # Skip if already proper type
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        non_null_series = df[col].dropna().astype(str)
        if len(non_null_series) == 0:
            continue
        
        # Skip if mostly empty or all None/null strings
        if non_null_series.str.lower().isin(['none', 'null', 'nan', '']).sum() > len(non_null_series) * 0.7:
            continue
            
        sample_size = min(len(non_null_series), 200)
        sample = non_null_series.sample(sample_size, random_state=42)
        
        # 1. NUMERIC CHECK (currency and commas)
        numeric_pattern = r'^[\$\€\£\s]*[\d\,]+(\.\d+)?[\s]*$'
        is_numeric = sample.str.match(numeric_pattern).sum() / len(sample) > 0.7
        
        if is_numeric:
            try:
                # Clean: remove non-numeric characters except decimal point and negative sign
                clean_col = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                converted_col = pd.to_numeric(clean_col, errors='coerce')
                
                # Only convert if > 70% of non-null values converted successfully
                if converted_col.notna().sum() / len(non_null_series) > 0.7:
                    df[col] = converted_col
                    conversions.append(f"💰 Converted '{col}' to Numeric")
                    continue
            except:
                pass

        # 2. DATE CHECK (ISO format, mixed separators, and common formats)
        is_date_col = any(k in col.lower() for k in ['date', 'time', 'day', 'year', 'month'])
        
        try:
            # Check if column looks like dates (contains common date patterns)
            date_pattern = r'(\d{1,4}[\-/\.]\d{1,2}[\-/\.]\d{1,4}|\d{4}-\d{2}-\d{2})'
            has_date_pattern = sample.str.contains(date_pattern, regex=True).sum() / len(sample) > 0.5
            
            if is_date_col or has_date_pattern:
                # Try multiple date formats
                converted_col = None
                
                # Try ISO format first (2024-08-04 format)
                try:
                    converted_col = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                    success_rate = converted_col.notna().sum() / len(non_null_series)
                    if success_rate > 0.7:
                        df[col] = converted_col
                        conversions.append(f"📅 Converted '{col}' to DateTime (ISO)")
                        continue
                except:
                    pass
                
                # Try with dayfirst for mixed separators (19-04-2019 or 19/04/2019)
                try:
                    normalized = df[col].astype(str).str.replace('/', '-', regex=False).str.replace('.', '-', regex=False)
                    converted_col = pd.to_datetime(normalized, errors='coerce', dayfirst=True)
                    success_rate = converted_col.notna().sum() / len(non_null_series)
                    if success_rate > 0.7:
                        df[col] = converted_col
                        conversions.append(f"📅 Converted '{col}' to DateTime (Mixed)")
                        continue
                except:
                    pass
                
                # Try inference (most flexible but slower)
                try:
                    converted_col = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    success_rate = converted_col.notna().sum() / len(non_null_series)
                    if success_rate > 0.7:
                        df[col] = converted_col
                        conversions.append(f"📅 Converted '{col}' to DateTime (Inferred)")
                        continue
                except:
                    pass
        except:
            pass
        
        # Ensure string columns don't have None/null string representations
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).replace(['None', 'none', 'null', 'NULL', 'NONE'], '')
                
    return df, conversions

def generate_profile(df):
    """Generates the JSON summary used by the AI as context."""
    profile = {
        "rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "column_list": list(df.columns),
        "column_details": {}
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique())
        missing = int(df[col].isna().sum())
        
        col_info = {"dtype": dtype, "unique_values": unique_count, "missing_values": missing}
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isna().all() else 0.0
            col_info["max"] = float(df[col].max()) if not df[col].isna().all() else 0.0
            col_info["mean"] = float(df[col].mean()) if not df[col].isna().all() else 0.0
        
        if pd.api.types.is_object_dtype(df[col]):
            try:
                col_info["samples"] = df[col].dropna().unique()[:5].tolist()
            except:
                col_info["samples"] = []
            
        profile["column_details"][col] = col_info

    with open(PROFILES_DIR / "data_profiling.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
        
    return profile

def load_csv_and_clean(file):
    """Loads, standardizes, and de-duplicates the dataset."""
    try:
        try:
            df = pd.read_csv(file)
        except:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')
        
        # Validate non-empty
        if df.empty:
            st.error("❌ The uploaded CSV file is empty.")
            return None, []
        
        logs = []
        
        # Clean column names
        df = clean_column_names(df)
        
        # Remove exact duplicates (keep first occurrence)
        duplicates_before = len(df)
        df = df.drop_duplicates(keep='first')
        duplicates_removed = duplicates_before - len(df)
        if duplicates_removed > 0:
            logs.append(f"🔁 Removed {duplicates_removed} duplicate row(s)")
        
        # Smart type conversion
        df, type_logs = convert_smart_types(df)
        logs.extend(type_logs)
        
        # Save to temp_data
        df.to_csv(TEMP_DATA_DIR / "source.csv", index=False)
        
        return df, logs
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, []

# ==================== MAIN UI: PHASE 1 (UPLOAD) ====================

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type="csv")

if uploaded_file is not None:
    if st.session_state.dataframe is None:
        with st.spinner("🧹 Cleaning & Profiling Data..."):
            df, changes = load_csv_and_clean(uploaded_file)
            
            if df is not None:
                st.session_state.dataframe = df
                st.session_state.profile_data = generate_profile(df)
                st.sidebar.success(f"✅ Ready: {df.shape[0]} rows")
                if changes:
                    with st.sidebar.expander("🛠️ Cleaning Log"):
                        for c in changes: st.write(c)

    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        profile = st.session_state.profile_data
        
        st.subheader("📊 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        if profile:
            c1.metric("Rows", f"{profile.get('rows', 0):,}")
            c2.metric("Columns", profile.get('num_columns', 0))
            details = profile.get('column_details', {})
            total_missing = sum(d.get('missing_values', 0) for d in details.values())
            c3.metric("Missing Values", f"{total_missing:,}")
        
        tab1, tab2 = st.tabs(["🔍 Data Preview", "📋 Data Types"])
        with tab1:
            st.dataframe(df.head(), use_container_width=True)
        with tab2:
            dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).astype(str)
            st.dataframe(dtype_df, use_container_width=True)

# ==================== MAIN UI: PHASE 2 (ANALYSIS) ====================

if st.session_state.dataframe is not None:
    st.markdown("---")
    st.header("💬 AI Data Analyst Chat")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "charts" in msg:
                for chart in msg["charts"]:
                    st.image(chart)
            if "code" in msg:
                with st.expander("📜 View Python Logic"):
                    st.code(msg["code"], language="python")

    user_query = st.chat_input("Ex: Show total sales by month and region, and top 5 products")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("🧠 Analyst is thinking..."):
            profile_json = json.dumps(st.session_state.profile_data, indent=2)
            
            # --- STRICT SYSTEM PROMPT FOR ROBUST PYTHON AUTOMATION ---
            prompt = f"""
You are an expert Python Data Analyst. 
**Context:** The variable `df` is already loaded in the environment and contains the dataset.

**Dataset Profile:**
{profile_json}

**User Query:** "{user_query}"

### MANDATORY INSTRUCTIONS:
1. **FILE PATHS & KPI CONFIG:** Use `TEMP_CHARTS_DIR` (a Path object) for all file saves. Convert to string with `str()`. ALWAYS populate and save `kpi_config.json` with real column names and formulas from the dataset. This file MUST contain valid "slicers" (2-3 categorical columns) and "kpis" (3-4 objects with label, formula, agg, fmt).
2. **SMART KPI GENERATION:** Analyze the Dataset Profile (column names, data types) + User Query to intelligently suggest:
   - **Slicers:** 2-3 categorical/string columns (case-insensitive) suitable for filtering
   - **KPIs:** 3-4 numeric-based KPIs with formulas using ACTUAL column names from the dataset. Use aggregations like sum, mean, count, min, max based on column type. Examples: "revenue", "price * quantity", "order_id" (for count).
   - **Formats:** "currency" for monetary values, "number" for counts/quantities, "percentage" for rates.
3. **NO DUMMY DATA:** Never create mock dictionaries or dataframes. Use `df` directly.
4. **AGGREGATION FIRST:** Always use `groupby` before plotting. Never plot raw, unaggregated rows.
5. **ONE QUESTION = ONE CHART:** Analyze the query. If 3 questions are asked, generate 3 distinct charts.
6. **SYNC VISUALS:** Static plots (Matplotlib) and Interactive plots (Plotly) must use the SAME aggregated data.
7. **DYNAMIC GRID:** Calculate dashboard grid rows as `math.ceil(num_plots / 2)`.
8. **NO SET(DICTS):** Do not use `set()` to remove duplicates from KPI lists as dictionaries are unhashable.
9. **TRACE TYPE SAFETY:** Only use XY-compatible trace types in subplots (bar, line, scatter, box, histogram). Create non-XY charts (pie, sunburst) as separate standalone figures.

### OUTPUT STRUCTURE (Return ONLY raw Python code):

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import math
from matplotlib.ticker import FuncFormatter

# Axis Formatter (K/M suffixes)
def currency_fmt(x, pos):
    if x >= 1e6: return f'${{x*1e-6:.1f}}M'
    if x >= 1e3: return f'${{x*1e-3:.0f}}K'
    return f'${{x:.0f}}'

# STEP 1: DATA PREPARATION (Aggregations for EACH question)
# df_q1 = df.groupby('col')['val'].sum().reset_index()

# STEP 2: STATIC VISUALIZATIONS (Matplotlib)
# [For each insight, create a plt.figure(figsize=(10, 6)), save to TEMP_CHARTS_DIR / "plot_X.png", and plt.close()]
# Example: plt.savefig(str(TEMP_CHARTS_DIR / "plot_1.png"), bbox_inches="tight", dpi=100)

# STEP 3: INTERACTIVE DASHBOARD (Plotly)
# num_plots = X
# rows = math.ceil(num_plots / 2)
# fig = make_subplots(rows=rows, cols=2, subplot_titles=(...))
# [Add traces matching static charts using SAME data. Ensure y-axis format matches static charts.]
# fig.write_html(str(TEMP_CHARTS_DIR / "interactive_dashboard.html"))

# STEP 4: KPI CONFIGURATION (JSON)
# Analyze the dataset profile and user query to suggest relevant KPIs and slicers.
# Use ACTUAL column names from the dataset in the formulas.
# IMPORTANT: Populate slicers with 2-3 categorical column names and kpis with 3-4 KPI objects.
# Each KPI must have: label, formula (using real columns), agg (sum/mean/count/min/max/nunique), fmt (currency/number/percentage)
kpi_dict = {{
    "slicers": [col for col in df.columns if df[col].dtype == 'object'][:3],  # First 3 categorical columns
    "kpis": [
        {{"label": "Total Count", "formula": "index", "agg": "count", "fmt": "number"}},
        {{"label": "Unique Items", "formula": df.columns[0], "agg": "nunique", "fmt": "number"}}
    ]
}}
# REPLACE THE ABOVE with intelligently selected KPIs based on the dataset profile and user query
with open(str(TEMP_CHARTS_DIR / "kpi_config.json"), "w") as f:
    json.dump(kpi_dict, f, indent=2)

# STEP 5: SUMMARY TEXT
result_text = "### Analysis Summary\\n* Insight 1...\\n* Insight 2..."
"""
            
            # 1. Call Gemini model
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            code = response.text.replace("```python", "").replace("```", "").strip()

            # 2. Strict Sanitization (Security & Data Integrity)
            clean_lines = []
            for line in code.split('\n'):
                # Block attempts to reload data or create fake data
                if any(x in line for x in ['read_csv', 'read_excel', 'pd.DataFrame({', 'data = {']):
                    if len(line) > 50: continue # Skip large mock dictionaries
                clean_lines.append(line)
            code = '\n'.join(clean_lines)
            
            # 3. Setup Execution Environment
            local_vars = {
                "df": st.session_state.dataframe, 
                "pd": pd, "plt": plt, "sns": sns, 
                "go": go, "px": px, "make_subplots": make_subplots, 
                "json": json, "math": math, "FuncFormatter": FuncFormatter,
                "os": os, "glob": glob, "Path": Path,
                "TEMP_CHARTS_DIR": TEMP_CHARTS_DIR
            }
            
            output_text = ""
            generated_charts = []
            
            try:
                # EXECUTE GENERATED CODE
                exec(code, local_vars)
                output_text = local_vars.get("result_text", "Analysis complete.")
                
                # Verify KPI config was created and is valid
                config_path = TEMP_CHARTS_DIR / "kpi_config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                            if config_data.get("slicers") and config_data.get("kpis"):
                                st.session_state.dashboard_ready = True
                    except:
                        pass
                
                # Load Static Charts for UI display
                chart_files = sorted(glob.glob(str(TEMP_CHARTS_DIR / "*.png")))
                for f in chart_files:
                    with open(f, "rb") as img:
                        generated_charts.append(img.read())

                # Confirm Dashboard Readiness (also check interactive HTML)
                if (TEMP_CHARTS_DIR / "interactive_dashboard.html").exists():
                    st.session_state.dashboard_ready = True
                    st.session_state.last_analysis_code = code

            except Exception as e:
                output_text = f"❌ Error executing analysis: {e}"
            
            # 4. Display Results in Chat
            with st.chat_message("assistant"):
                st.markdown(output_text)
                for chart in generated_charts:
                    st.image(chart)
                with st.expander("📜 View Python Logic"):
                    st.code(code, language="python")

            # 5. Persistent History
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": output_text,
                "charts": generated_charts,
                "code": code
            })
            
            st.rerun()
else: 
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to get started.")

# ==================== FOOTER ====================
st.markdown("---") 
st.markdown("<div style='text-align: center; color: gray;'>AI Data Analyst Agent | Python Automation</div>", unsafe_allow_html=True)