from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from numbers import Number
import matplotlib.dates as mdates
import os
import warnings
import io
import shutil
import glob
import json
import math
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
from matplotlib.ticker import FuncFormatter
import base64
import tempfile
from fpdf import FPDF


# ==================== CONFIGURATION ====================
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
load_dotenv()

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception:
        pass

# Relative paths — resolves to whatever directory uvicorn is run from
# Run uvicorn from the ROOT of the project: uvicorn backend.main:app --reload
TEMP_CHARTS_DIR = Path("temp_charts")
TEMP_DATA_DIR = Path("temp_data")
PROFILES_DIR = Path("data_profiles")

for d in [TEMP_CHARTS_DIR, TEMP_DATA_DIR, PROFILES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clear temp_charts and temp_data on every server startup
# so stale files from previous sessions are removed
for _cleanup_dir in [TEMP_CHARTS_DIR, TEMP_DATA_DIR]:
    for _f in _cleanup_dir.glob("*"):
        try:
            if _f.is_file():
                _f.unlink()
        except Exception:
            pass


# ==================== HELPER FUNCTIONS (BLACK-BOX LOGIC) ====================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^a-zA-Z0-9]", "_", str(c).strip().lower()).strip("_")
        for c in df.columns
    ]
    return df


def convert_smart_types(df: pd.DataFrame):
    conversions: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        non_null_series = df[col].dropna().astype(str)
        if len(non_null_series) == 0:
            continue

        if non_null_series.str.lower().isin(["none", "null", "nan", ""]).sum() > len(non_null_series) * 0.7:
            continue

        sample_size = min(len(non_null_series), 200)
        sample = non_null_series.sample(sample_size, random_state=42)

        numeric_pattern = r"^[\$\€\£\s]*[\d\,]+(\.\d+)?[\s]*$"
        is_numeric = sample.str.match(numeric_pattern).sum() / len(sample) > 0.7

        if is_numeric:
            try:
                clean_col = df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
                converted_col = pd.to_numeric(clean_col, errors="coerce")
                if converted_col.notna().sum() / len(non_null_series) > 0.7:
                    df[col] = converted_col
                    conversions.append(f"💰 Converted '{col}' to Numeric")
                    continue
            except Exception:
                pass

        is_date_col = any(k in col.lower() for k in ["date", "time", "day", "year", "month"])

        try:
            date_pattern = r"(\d{1,4}[\-/\.]\d{1,2}[\-/\.]\d{1,4}|\d{4}-\d{2}-\d{2})"
            has_date_pattern = sample.str.contains(date_pattern, regex=True).sum() / len(sample) > 0.5

            if is_date_col or has_date_pattern:
                converted_col = None

                try:
                    converted_col = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
                    success_rate = converted_col.notna().sum() / len(non_null_series)
                    if success_rate > 0.7:
                        df[col] = converted_col
                        conversions.append(f"📅 Converted '{col}' to DateTime (ISO)")
                        continue
                except Exception:
                    pass

                try:
                    normalized = (
                        df[col]
                        .astype(str)
                        .str.replace("/", "-", regex=False)
                        .str.replace(".", "-", regex=False)
                    )
                    converted_col = pd.to_datetime(normalized, errors="coerce", dayfirst=True)
                    success_rate = converted_col.notna().sum() / len(non_null_series)
                    if success_rate > 0.7:
                        df[col] = converted_col
                        conversions.append(f"📅 Converted '{col}' to DateTime (Mixed)")
                        continue
                except Exception:
                    pass

                try:
                    converted_col = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    success_rate = converted_col.notna().sum() / len(non_null_series)
                    if success_rate > 0.7:
                        df[col] = converted_col
                        conversions.append(f"📅 Converted '{col}' to DateTime (Inferred)")
                        continue
                except Exception:
                    pass
        except Exception:
            pass

        if df[col].dtype == "object":
            df[col] = df[col].astype(str).replace(
                ["None", "none", "null", "NULL", "NONE"], ""
            )

    return df, conversions


def generate_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "column_list": list(df.columns),
        "column_details": {},
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = int(df[col].nunique())
        missing = int(df[col].isna().sum())

        col_info: Dict[str, Any] = {
            "dtype": dtype,
            "unique_values": unique_count,
            "missing_values": missing,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if not df[col].isna().all() else 0.0
            col_info["max"] = float(df[col].max()) if not df[col].isna().all() else 0.0
            col_info["mean"] = (
                float(df[col].mean()) if not df[col].isna().all() else 0.0
            )

        if pd.api.types.is_object_dtype(df[col]):
            try:
                col_info["samples"] = df[col].dropna().unique()[:5].tolist()
            except Exception:
                col_info["samples"] = []

        profile["column_details"][col] = col_info

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_DIR / "data_profiling.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)

    return profile


def load_csv_and_clean(file_bytes: bytes) -> (pd.DataFrame, List[str]):
    try:
        buffer = io.BytesIO(file_bytes)
        try:
            df = pd.read_csv(buffer)
        except Exception:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding="latin1")

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        logs: List[str] = []

        df = clean_column_names(df)

        duplicates_before = len(df)
        df = df.drop_duplicates(keep="first")
        duplicates_removed = duplicates_before - len(df)
        if duplicates_removed > 0:
            logs.append(f"🔁 Removed {duplicates_removed} duplicate row(s)")

        df, type_logs = convert_smart_types(df)
        logs.extend(type_logs)

        TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TEMP_DATA_DIR / "source.csv", index=False)

        return df, logs
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def _count_questions(user_query: str) -> int:
    """Count how many distinct questions/tasks the user asked."""
    # Split by common delimiters: numbered lists, question marks, newlines, 'and'
    parts = re.split(r'\n|\?\s*\d+[\.\)]|\d+[\.\)]\s', user_query)
    questions = [p.strip() for p in parts if len(p.strip()) > 10]
    if len(questions) <= 1:
        # Try splitting by question marks
        questions = [p.strip() for p in user_query.split('?') if len(p.strip()) > 10]
    return max(len(questions), 1)


def _select_kpis_and_slicers(df: pd.DataFrame, profile_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Use AI to select RELEVANT KPIs and slicers based on dataset + user questions.
    Returns structured config that backend calculates deterministically.
    """
    col_summary = []
    for col, info in profile_data.get("column_details", {}).items():
        col_summary.append(f"- {col}: dtype={info['dtype']}, unique={info['unique_values']}, missing={info['missing_values']}")

    sample_rows = df.head(3).to_dict(orient="records")

    kpi_prompt = f"""You are a data analysis expert. Select the most relevant KPIs and slicers for a dashboard.

DATASET COLUMNS:
{chr(10).join(col_summary)}

SAMPLE DATA (3 rows):
{json.dumps(sample_rows, default=str)}

USER'S ANALYSIS QUESTIONS:
{user_query}

YOUR TASK:
1. Select 3-5 KPIs that are DIRECTLY relevant to the user's questions above.
   - Each KPI must use a REAL column name from the dataset
   - KPI formula must be one of: sum, mean, count, nunique, max, min
   - For ratio KPIs use: ratio (numerator_col / denominator_col)
   - label should be human readable
   - fmt: "currency" for money, "number" for counts, "percentage" for rates

2. Select 2-4 slicers that are RELEVANT to filtering the analysis:
   - Must be categorical (object/string) columns
   - Must have reasonable cardinality (< 50 unique values preferred)
   - Must be useful for the user's specific questions

RETURN ONLY THIS EXACT JSON (no explanation, no markdown):
{{
  "kpis": [
    {{"label": "Human Readable Name", "column": "actual_column_name", "agg": "sum", "fmt": "currency"}},
    {{"label": "Another KPI", "column": "actual_column_name", "agg": "count", "fmt": "number"}}
  ],
  "slicers": ["column_name_1", "column_name_2"]
}}"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(kpi_prompt)
        raw = resp.text.replace("```json", "").replace("```", "").strip()
        config = json.loads(raw)

        # Validate columns actually exist in df
        valid_kpis = []
        for kpi in config.get("kpis", []):
            col = kpi.get("column", "")
            agg = kpi.get("agg", "sum")
            if col == "index" or col in df.columns:
                valid_kpis.append(kpi)
        config["kpis"] = valid_kpis[:5]

        valid_slicers = [s for s in config.get("slicers", []) if s in df.columns]
        config["slicers"] = valid_slicers[:4]

        # Fallback if AI returned empty
        if not config["kpis"]:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            config["kpis"] = [{"label": col.replace("_", " ").title(), "column": col, "agg": "sum", "fmt": "number"} for col in numeric_cols[:4]]
        if not config["slicers"]:
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            config["slicers"] = cat_cols[:3]

        return config

    except Exception:
        # Safe fallback
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        return {
            "kpis": [{"label": col.replace("_", " ").title(), "column": col, "agg": "sum", "fmt": "number"} for col in numeric_cols[:4]],
            "slicers": cat_cols[:3],
        }


def _calculate_kpis(df: pd.DataFrame, kpi_config: List[Dict]) -> List[Dict]:
    """
    Calculate KPI values deterministically from config.
    No AI involved — pure pandas operations.
    """
    results = []
    for kpi in kpi_config:
        col = kpi.get("column", "")
        agg = kpi.get("agg", "sum")
        label = kpi.get("label", col)
        fmt = kpi.get("fmt", "number")

        try:
            if col == "index" or agg == "count":
                value = len(df)
            elif col not in df.columns:
                continue
            elif agg == "sum":
                value = float(df[col].sum())
            elif agg == "mean":
                value = float(df[col].mean())
            elif agg == "nunique":
                value = int(df[col].nunique())
            elif agg == "max":
                value = float(df[col].max())
            elif agg == "min":
                value = float(df[col].min())
            elif agg == "ratio":
                cols = kpi.get("columns", [col])
                if len(cols) == 2 and cols[0] in df.columns and cols[1] in df.columns:
                    denom = df[cols[1]].sum()
                    value = float(df[cols[0]].sum() / denom) if denom != 0 else 0.0
                else:
                    value = float(df[col].sum())
            else:
                value = float(df[col].sum())

            if math.isnan(value) or math.isinf(value):
                value = 0.0

            results.append({"label": label, "value": value, "fmt": fmt, "column": col, "agg": agg})
        except Exception:
            continue

    return results


def run_analysis(df: pd.DataFrame, profile_data: Dict[str, Any], user_query: str):
    if not GENAI_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set on the backend."
        )

    # ── STEP A: Count exact number of questions to enforce chart count ──
    num_questions = _count_questions(user_query)

    # ── STEP B: Select relevant KPIs + slicers via structured AI call ──
    kpi_slicer_config = _select_kpis_and_slicers(df, profile_data, user_query)

    # ── STEP C: Calculate KPI values deterministically (no AI, pure pandas) ──
    kpi_values = _calculate_kpis(df, kpi_slicer_config.get("kpis", []))

    # Save kpi_config.json with both config + calculated values
    full_kpi_config = {
        "slicers": kpi_slicer_config.get("slicers", []),
        "kpis": kpi_slicer_config.get("kpis", []),
        "kpi_values": kpi_values,
    }
    TEMP_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(TEMP_CHARTS_DIR / "kpi_config.json"), "w") as f:
        json.dump(full_kpi_config, f, indent=2)

    profile_json = json.dumps(profile_data, indent=2)

    prompt = f"""
You are an expert Python Data Analyst. 
**Context:** The variable `df` is already loaded in the environment and contains the dataset.

**Dataset Profile:**
{profile_json}

**User Query:** "{user_query}"

**CRITICAL - Number of questions detected: {num_questions}**
**KPIs already selected (DO NOT change these):** {json.dumps(kpi_slicer_config.get('kpis', []))}
**Slicers already selected (DO NOT change these):** {json.dumps(kpi_slicer_config.get('slicers', []))}

### MANDATORY INSTRUCTIONS:

1. **EXHAUSTIVE VISUALIZATION — STRICT ENFORCEMENT:**
   - The user asked EXACTLY {num_questions} question(s).
   - You MUST generate EXACTLY {num_questions} Matplotlib chart(s).
   - Each question maps to exactly one chart. No bundling. No skipping.
   - At the end of your code, set: `num_charts_generated = {num_questions}`

2. **KPI CONFIG — DO NOT REGENERATE:**
   - KPIs and slicers are already saved. DO NOT overwrite `kpi_config.json`.

3. **DO NOT WRITE ANY PLOTLY CODE — CRITICAL:**
   - Do NOT import plotly. Do NOT create any plotly figures.
   - Do NOT save any .json plotly files.
   - Backend automatically converts matplotlib charts to interactive Plotly charts.
   - ONLY write matplotlib/seaborn code.

4. **NO DUMMY DATA:** Use `df` directly. Never create mock dataframes.

5. **AGGREGATION FIRST:**
   - Aggregate ONCE per question: `df_q1 = df.groupby(...).sum().reset_index()`
   - NEVER plot raw unaggregated rows.
   - Store each aggregated df as `df_q1`, `df_q2`, `df_q3` etc.

6. **NO SET(DICTS):** Do not use `set()` on dictionaries.

7. **AUTOMATIC COLUMN TYPE DETECTION:**
   - Date/Time: use existing datetime dtypes or try `pd.to_datetime`.
   - Categorical: object/string columns with low cardinality (< 20 unique).
   - Numeric: int/float columns.

8. **SMART TIME AGGREGATION:**
    - unique time values > 50 → aggregate by month.
    - unique time values > 20 → aggregate by week.
    - else → use original values.

9. **CHART TYPE SELECTION:**
    - trend/over time → line chart
    - category comparison → bar chart
    - distribution → histogram
    - relationship/correlation → scatter plot
    - proportion/share → pie chart

10. **X-AXIS READABILITY:**
    - labels > 20 → rotate 45°
    - labels > 40 → rotate 90°

11. **AUTOMATIC DATA CLEANING (CRITICAL):**
    - Drop nulls in key columns before each chart.
    - You MUST explicitly convert any date/time column using `pd.to_datetime(df['col'], errors="coerce")` BEFORE using `.dt` accessor. Do not assume the column is already a datetime object.

12. STRICT SAVE PATH (CRITICAL):
    - You MUST use `TEMP_CHARTS_DIR`. Example: `plt.savefig(TEMP_CHARTS_DIR / "matplotlib_chart_1.png")`
    - NEVER save directly to a string filename like `plt.savefig("chart.png")`.
    - DO NOT define or import TEMP_CHARTS_DIR. DO NOT write try/except blocks for it. It is already injected.

13. **NO GUI WINDOWS OR CLOSING (CRITICAL):**
    - NEVER call `plt.show()`, `fig.show()`, or `plt.close()`. 
    - This code runs on a headless server. The backend needs the figures to remain open in memory to convert them to interactive dashboard charts.

### OUTPUT STRUCTURE (Return ONLY raw Python code, no markdown):

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
from matplotlib.ticker import FuncFormatter

def currency_fmt(x, pos):
    if x >= 1e6: return f'${{x*1e-6:.1f}}M'
    if x >= 1e3: return f'${{x*1e-3:.0f}}K'
    return f'${{x:.0f}}'

# STEP 1: DATA PREPARATION — df_q1, df_q2, df_q3 etc. (one per question)
# STEP 2: MATPLOTLIB CHARTS — exactly {num_questions} charts
#         plt.savefig(TEMP_CHARTS_DIR / "matplotlib_chart_N.png")
# NOTE: Do NOT call plt.close(). Do NOT write Plotly code.
# STEP 3: num_charts_generated = {num_questions}
# STEP 4: result_text = "### Analysis Summary\n* ..."
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    code = response.text.replace("```python", "").replace("```", "").strip()

    # Fix invisible characters (non-breaking spaces) that AI sometimes hallucinates
    code = code.replace("\xa0", " ")

    # Strip any lines that try to load data, overwrite kpi_config,
    # OR redefine TEMP_CHARTS_DIR (AI sometimes redefines it as a string
    # using os.path.join which breaks the Path / operator and crashes silently)
    clean_lines = []
    for line in code.split("\n"):
        # Block data loading attempts
        if any(x in line for x in ["read_csv", "read_excel", "pd.DataFrame({"]):
            if len(line) > 50:
                continue
        # Block overwriting kpi_config.json
        if "kpi_config.json" in line and ("open(" in line or "json.dump" in line):
            continue
        # Block AI from redefining or re-creating TEMP_CHARTS_DIR
        if "TEMP_CHARTS_DIR" in line and any(x in line for x in ["os.path", "os.getcwd", "os.path.join", "= Path(", "makedirs", ".mkdir("]):
            indent = line[:len(line) - len(line.lstrip())]
            clean_lines.append(indent + "pass")
            continue
        # MUST BLOCK GUI popup commands and block closing figures
        if ".show(" in line or "plt.close" in line:
            continue
        clean_lines.append(line)
    code = "\n".join(clean_lines)

    # Fix unterminated triple-quoted strings — AI sometimes forgets the closing """
    # causing a SyntaxError so exec() fails silently and no charts are saved
    if code.count('"""') % 2 != 0:
        code += '\n"""'
    if code.count("'''") % 2 != 0:
        code += "\n'''"

    # Resolve to absolute path so exec() saves files to the correct location
    # regardless of working directory inside exec
    abs_temp_charts = TEMP_CHARTS_DIR.resolve()
    abs_temp_charts.mkdir(parents=True, exist_ok=True)

    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "go": go,
        "px": px,
        "make_subplots": make_subplots,
        "json": json,
        "math": math,
        "FuncFormatter": FuncFormatter,
        "os": os,
        "glob": glob,
        "Path": Path,
        "re": re,
        "TEMP_CHARTS_DIR": abs_temp_charts,
    }

    # Clear old chart files using abs_temp_charts (consistent absolute path)
    for f in glob.glob(str(abs_temp_charts / "*.png")):
        try:
            os.remove(f)
        except Exception:
            pass
    for f in glob.glob(str(abs_temp_charts / "*.html")):
        try:
            os.remove(f)
        except Exception:
            pass
    for f in glob.glob(str(abs_temp_charts / "plotly_chart_*.json")):
        try:
            os.remove(f)
        except Exception:
            pass

    output_text = ""
    plt.close("all")
    try:
        exec(code, local_vars)
        output_text = local_vars.get("result_text", "Analysis complete.")

        # ── VALIDATION: Ensure chart count matches question count ──
        charts_generated = local_vars.get("num_charts_generated", None)
        png_files = sorted(glob.glob(str(abs_temp_charts / "*.png")))
        plotly_files = sorted(glob.glob(str(abs_temp_charts / "plotly_chart_*.json")))

        if len(png_files) < num_questions:
            # Retry once with a stricter prompt
            retry_prompt = f"""
The previous code only generated {len(png_files)} static chart(s).
The user asked EXACTLY {num_questions} question(s). You MUST generate EXACTLY {num_questions} Matplotlib chart(s).

User Query: "{user_query}"
Dataset columns: {list(df.columns)}

Regenerate the COMPLETE code. Every question must have its own chart.
Return ONLY raw Python code. No markdown.
"""
            retry_response = model.generate_content(retry_prompt)
            retry_code = retry_response.text.replace("```python", "").replace("```", "").strip()
            retry_code = retry_code.replace("\xa0", " ") # Fix spaces again
            retry_clean = []
            for line in retry_code.split("\n"):
                if any(x in line for x in ["read_csv", "read_excel", "pd.DataFrame({"]):
                    if len(line) > 50:
                        continue
                if "kpi_config.json" in line and ("open(" in line or "json.dump" in line):
                    continue
                if "TEMP_CHARTS_DIR" in line and any(x in line for x in ["os.path", "os.getcwd", "os.path.join", "= Path(", "makedirs", ".mkdir("]):
                    indent = line[:len(line) - len(line.lstrip())]
                    retry_clean.append(indent + "pass")
                    continue
                if ".show(" in line or "plt.close" in line:
                    continue
                retry_clean.append(line)
            retry_code = "\n".join(retry_clean)
            if retry_code.count('"""') % 2 != 0:
                retry_code += '\n"""'
            if retry_code.count("'''") % 2 != 0:
                retry_code += "\n'''"
            plt.close("all")
            try:
                exec(retry_code, local_vars)
                output_text = local_vars.get("result_text", output_text)
                code = retry_code  # use retry code for persistence
            except Exception:
                pass  # keep original output if retry also fails

    except Exception as e:
        import traceback
        output_text = f"❌ Error executing analysis: {e}\n\nTraceback:\n{traceback.format_exc()}"

    # ── POST-EXEC: Convert matplotlib figures → Plotly (100% match guaranteed) ──
    try:
        _mpl_charts: List[Dict[str, Any]] = []
        _fig_nums = plt.get_fignums()
        for _i, _fig_num in enumerate(_fig_nums):
            try:
                _mpl_fig = plt.figure(_fig_num)
                try: _mpl_fig.canvas.draw()
                except Exception: pass
                
                _axes = _mpl_fig.get_axes()
                if not _axes: continue
                _ax = _axes[0]

                _title = _ax.get_title() or f"Chart {_i + 1}"
                _xlabel = _ax.get_xlabel() or ""
                _ylabel = _ax.get_ylabel() or ""

                _lines = _ax.get_lines()
                _patches = _ax.patches
                _plotly_fig = go.Figure()
                has_data = False

                # ── LINE CHART ──
                if _lines and not _patches:
                    _line = _lines[0]
                    _xdata = list(_line.get_xdata())
                    _ydata = list(_line.get_ydata())
                    _x_formatted = []
                    is_date_axis = ("Date" in type(_ax.xaxis.get_major_formatter()).__name__ or "Date" in type(_ax.xaxis.get_major_locator()).__name__)
                    for x in _xdata:
                        if isinstance(x, pd.Timestamp): _x_formatted.append(x.strftime('%Y-%m-%d'))
                        elif is_date_axis and isinstance(x, Number):
                            try: _x_formatted.append(mdates.num2date(x).strftime('%Y-%m-%d'))
                            except Exception: _x_formatted.append(str(x))
                        else: _x_formatted.append(str(x))
                    if _x_formatted and _ydata:
                        _plotly_fig.add_trace(go.Scatter(x=_x_formatted, y=_ydata, mode='lines+markers', name=_ylabel))
                        has_data = True

                # ── BAR CHART FIX ──
                elif _patches:
                    _x_labels = [t.get_text().replace('\n', ' ').strip() for t in _ax.get_xticklabels()]
                    xticks = list(_ax.get_xticks())
                    _valid_xs, _valid_ys = [], []
                    
                    for p in _patches:
                        h, w = p.get_height(), p.get_width()
                        if h == 0 and w == 0: continue
                        
                        px_val = p.get_x() + w / 2.0
                        label = str(px_val)
                        
                        if xticks and _x_labels:
                            try:
                                c_idx = min(range(len(xticks)), key=lambda idx: abs(xticks[idx] - px_val))
                                if abs(xticks[c_idx] - px_val) < 0.6 and c_idx < len(_x_labels) and _x_labels[c_idx]:
                                    label = _x_labels[c_idx]
                            except Exception: pass
                            
                        _valid_xs.append(label)
                        _valid_ys.append(h)
                        
                    if _valid_xs and _valid_ys:
                        _plotly_fig.add_trace(go.Bar(x=_valid_xs, y=_valid_ys, name=_ylabel))
                        has_data = True

                if has_data:
                    _plotly_fig.update_layout(title=_title, xaxis_title=_xlabel, yaxis_title=_ylabel)
                    _js = _plotly_fig.to_json()
                    _fp = abs_temp_charts / f"plotly_chart_{_i + 1}.json"
                    with open(str(_fp), "w") as _pf: _pf.write(_js)
                    _mpl_charts.append({"title": _title, "plotly_json": _js})
            except Exception: 
                continue

        if _mpl_charts:
            with open(str(abs_temp_charts / "dashboard_charts.json"), "w") as _f:
                json.dump({"charts": _mpl_charts}, _f)
    except Exception:
        pass  # fallback: existing charts used

    # Collect ALL png files
    chart_files = sorted(glob.glob(str(abs_temp_charts / "*.png")))
    charts_base64: List[str] = []
    for fpath in chart_files:
        try:
            with open(fpath, "rb") as img:
                encoded = base64.b64encode(img.read()).decode("utf-8")
                charts_base64.append(encoded)
        except Exception:
            continue

    # Build dashboard_charts.json from plotly JSON files if AI didn't create it
    dashboard_charts_path = abs_temp_charts / "dashboard_charts.json"
    if not dashboard_charts_path.exists():
        plotly_files = sorted(glob.glob(str(abs_temp_charts / "plotly_chart_*.json")))
        if plotly_files:
            charts_list = []
            for i, fp in enumerate(plotly_files):
                try:
                    with open(fp, "r") as pf:
                        pjson = pf.read()
                    charts_list.append({"title": f"Chart {i+1}", "plotly_json": pjson})
                except Exception:
                    continue
            with open(str(dashboard_charts_path), "w") as f:
                json.dump({"charts": charts_list}, f)

    dashboard_ready = False
    config_path = TEMP_CHARTS_DIR / "kpi_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                if config_data.get("slicers") and config_data.get("kpis"):
                    dashboard_ready = True
        except Exception:
            pass

    if dashboard_charts_path.exists():
        dashboard_ready = True

    # Persist last analysis code for dashboard slicer updates
    try:
        last_code_path = abs_temp_charts / "last_analysis.txt"
        with open(last_code_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception:
        pass

    # CRITICAL CLEANUP: Clear memory so server doesn't crash from open figures
    plt.close("all")

    return {
        "summary": output_text,
        "charts": charts_base64,
        "dashboard_ready": dashboard_ready,
        "last_analysis_code": code,
        "kpi_values": kpi_values,
    }

# ==================== FASTAPI APP & ENDPOINTS ====================

app = FastAPI(title="AI Data Analyst Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── NEW: Request Model for PDF Generation ──
class PDFDownloadRequest(BaseModel):
    dashboard_screenshot: str


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df, logs = load_csv_and_clean(contents)
        profile = generate_profile(df)
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "logs": logs,
            "profile": profile,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/analyze")
async def analyze(payload: Dict[str, Any]):
    user_query = payload.get("question") or payload.get("query")
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing 'question' field.")

    data_path = TEMP_DATA_DIR / "source.csv"
    if not data_path.exists():
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")

    try:
        try:
            df = pd.read_csv(data_path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(data_path, encoding="latin1")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load stored dataset: {e}")

    profile_path = PROFILES_DIR / "data_profiling.json"
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile_data = json.load(f)
    else:
        profile_data = generate_profile(df)

    try:
        result = run_analysis(df, profile_data, user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


@app.get("/kpi-config")
async def get_kpi_config():
    config_path = TEMP_CHARTS_DIR / "kpi_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="KPI configuration not found.")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read KPI configuration: {e}")


@app.get("/dashboard-config")
async def dashboard_config():
    """
    Returns slicer metadata (column names and distinct values) and the current
    interactive dashboard HTML, if available.
    """
    config_path = TEMP_CHARTS_DIR / "kpi_config.json"
    slicer_meta: List[Dict[str, Any]] = []
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            slicer_cols = config.get("slicers", []) or []
            data_path = TEMP_DATA_DIR / "source.csv"
            df = None
            if data_path.exists():
                try:
                    try:
                        df = pd.read_csv(data_path, encoding="utf-8")
                    except Exception:
                        df = pd.read_csv(data_path, encoding="latin1")
                except Exception:
                    df = None

            for raw_col in slicer_cols:
                if df is None:
                    # If we cannot load data, still expose slicer column names with empty values
                    slicer_meta.append({"name": str(raw_col), "values": []})
                    continue

                col_map = {c.lower(): c for c in df.columns}
                col = col_map.get(str(raw_col).lower())
                if col and col in df.columns:
                    try:
                        values = (
                            df[col]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                        )
                    except Exception:
                        values = []
                    slicer_meta.append({"name": col, "values": values})
        except Exception:
            pass

    # Load calculated KPI values (deterministically calculated, not AI-guessed)
    kpi_values: List[Dict[str, Any]] = []
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            kpi_values = cfg.get("kpi_values", [])
        except Exception:
            pass

    # Load Plotly chart JSONs for Next.js rendering (replaces raw HTML)
    dashboard_charts: List[Dict[str, Any]] = []
    dashboard_charts_path = TEMP_CHARTS_DIR / "dashboard_charts.json"
    if dashboard_charts_path.exists():
        try:
            with open(dashboard_charts_path, "r", encoding="utf-8") as f:
                dc = json.load(f)
            dashboard_charts = dc.get("charts", [])
        except Exception:
            pass

    # Load PNG chart images (matplotlib/seaborn) as base64 for dashboard display
    png_charts: List[Dict[str, Any]] = []
    abs_charts = TEMP_CHARTS_DIR.resolve()
    for fpath in sorted(glob.glob(str(abs_charts / "*.png"))):
        try:
            with open(fpath, "rb") as img:
                encoded = base64.b64encode(img.read()).decode("utf-8")
                png_charts.append({
                    "filename": Path(fpath).name,
                    "image_b64": encoded,
                })
        except Exception:
            continue

    return {"slicers": slicer_meta, "kpi_values": kpi_values, "dashboard_charts": dashboard_charts, "png_charts": png_charts}


@app.post("/dashboard-update")
async def dashboard_update(payload: Dict[str, Any]):
    filters: Dict[str, List[str]] = payload.get("filters") or {}
    data_path = TEMP_DATA_DIR / "source.csv"
    df_filtered = None
    if data_path.exists():
        try:
            try: df = pd.read_csv(data_path, encoding="utf-8")
            except Exception: df = pd.read_csv(data_path, encoding="latin1")
            df_filtered = df.copy()
            for raw_col, values in filters.items():
                if not values: continue
                col_map = {c.lower(): c for c in df_filtered.columns}
                col = col_map.get(str(raw_col).lower())
                if col and col in df_filtered.columns:
                    try: df_filtered = df_filtered[df_filtered[col].astype(str).isin(values)]
                    except Exception: continue
        except Exception: df_filtered = None

    kpi_values = []
    config_path = TEMP_CHARTS_DIR / "kpi_config.json"
    if config_path.exists() and df_filtered is not None:
        try: kpi_values = _calculate_kpis(df_filtered, json.load(open(config_path, "r")).get("kpis", []))
        except Exception: pass

    last_code_path = TEMP_CHARTS_DIR / "last_analysis.txt"
    dashboard_charts = []
    exec_success = False

    if last_code_path.exists() and df_filtered is not None:
        try: code = open(last_code_path, "r", encoding="utf-8").read()
        except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to load analysis code: {e}")
        
        sanitized_lines = []
        for line in code.split("\n"):
            if "TEMP_CHARTS_DIR" in line and any(x in line for x in ["os.path", "os.getcwd", "os.path.join", "= Path(", "makedirs"]): 
                indent = line[:len(line) - len(line.lstrip())]
                sanitized_lines.append(indent + "pass")
                continue
            if ".show(" in line or "plt.close" in line: 
                indent = line[:len(line) - len(line.lstrip())]
                sanitized_lines.append(indent + "pass")
                continue
            sanitized_lines.append(line)
        code = "\n".join(sanitized_lines)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            local_vars = {
                "df": df_filtered.copy(), "pd": pd, "plt": plt, "sns": sns, "go": go, "px": px,
                "make_subplots": make_subplots, "json": json, "Path": Path, "FuncFormatter": FuncFormatter,
                "re": re, "os": os, "math": math, "glob": glob, "TEMP_CHARTS_DIR": tmp_path,
            }
            
            plt.close("all") # CRITICAL: Pehle ke saare open charts memory se hatane ke liye
            try: 
                exec(code, local_vars)
                exec_success = True
            except Exception as e: 
                pass 
            
            try:
                _mpl_charts: List[Dict[str, Any]] = []
                _fig_nums = plt.get_fignums()
                target_dir = tmp_path  
                
                for _i, _fig_num in enumerate(_fig_nums):
                    try:
                        _mpl_fig = plt.figure(_fig_num)
                        try:
                            _mpl_fig.canvas.draw()  # CRITICAL FIX 1: Force draw to load X/Y labels perfectly
                        except Exception: pass
                        
                        _axes = _mpl_fig.get_axes()
                        if not _axes: continue
                        _ax = _axes[0]

                        _title = _ax.get_title() or f"Chart {_i + 1}"
                        _xlabel = _ax.get_xlabel() or ""
                        _ylabel = _ax.get_ylabel() or ""

                        _lines = _ax.get_lines()
                        _patches = _ax.patches
                        _plotly_fig = None
                        has_data = False

                        # ── LINE CHART FIX ──
                        if _lines and not _patches:
                            _line = _lines[0]
                            _xdata_raw = list(_line.get_xdata())
                            _ydata = list(_line.get_ydata())
                            _xdata = []
                            
                            is_date_axis = ("Date" in type(_ax.xaxis.get_major_formatter()).__name__ or 
                                            "Date" in type(_ax.xaxis.get_major_locator()).__name__)
                            
                            for x in _xdata_raw:
                                if is_date_axis and isinstance(x, Number):
                                    try: _xdata.append(mdates.num2date(x).strftime('%Y-%m-%d'))
                                    except Exception: _xdata.append(str(x))
                                elif isinstance(x, pd.Timestamp):
                                    _xdata.append(x.strftime('%Y-%m-%d'))
                                else:
                                    try:
                                        if "datetime" in str(type(x)): _xdata.append(pd.to_datetime(x).strftime('%Y-%m-%d'))
                                        else: _xdata.append(str(x))
                                    except Exception: _xdata.append(str(x))

                            if _xdata and _ydata:
                                _plotly_fig = go.Figure()
                                _plotly_fig.add_trace(go.Scatter(x=_xdata, y=_ydata, mode='lines+markers', name=_ylabel))
                                has_data = True
                                
                        # ── BAR CHART FIX (Solves 5 Categories & Wrong Hover Values) ──
                        elif _patches:
                            _x_labels = [t.get_text().replace('\n', ' ').strip() for t in _ax.get_xticklabels()]
                            xticks = list(_ax.get_xticks())
                            _valid_xs, _valid_ys = [], []
                            
                            for p in _patches:
                                h, w = p.get_height(), p.get_width()
                                
                                # Ignore extra background patches or legend artifacts 
                                if h == 0 and w == 0: continue
                                
                                px_val = p.get_x() + w / 2.0
                                label = str(px_val)
                                
                                if xticks and _x_labels:
                                    try:
                                        c_idx = min(range(len(xticks)), key=lambda idx: abs(xticks[idx] - px_val))
                                        # Strict matching to ensure we don't pick up random boxes
                                        if abs(xticks[c_idx] - px_val) < 0.6 and c_idx < len(_x_labels) and _x_labels[c_idx]:
                                            label = _x_labels[c_idx]
                                    except Exception: pass
                                    
                                _valid_xs.append(label)
                                _valid_ys.append(h)
                            
                            if _valid_xs and _valid_ys:
                                _plotly_fig = go.Figure()
                                _plotly_fig.add_trace(go.Bar(x=_valid_xs, y=_valid_ys, name=_ylabel))
                                has_data = True

                        if has_data and _plotly_fig is not None:
                            _plotly_fig.update_layout(title=_title, xaxis_title=_xlabel, yaxis_title=_ylabel)
                            _js = _plotly_fig.to_json()
                            _fp = target_dir / f"plotly_chart_{_i + 1}.json"
                            with open(str(_fp), "w") as _pf: _pf.write(_js)
                            _mpl_charts.append({"title": _title, "plotly_json": _js})
                    except Exception: 
                        continue

                if _mpl_charts:
                    with open(str(target_dir / "dashboard_charts.json"), "w") as _f:
                        json.dump({"charts": _mpl_charts}, _f)
            except Exception: 
                pass

            tmp_dc_path = tmp_path / "dashboard_charts.json"
            if tmp_dc_path.exists():
                try: dashboard_charts = json.load(open(tmp_dc_path, "r")).get("charts", [])
                except Exception: pass
            if not dashboard_charts:
                for fp in sorted(glob.glob(str(tmp_path / "plotly_chart_*.json"))):
                    try: dashboard_charts.append({"title": Path(fp).stem, "plotly_json": open(fp, "r").read()})
                    except Exception: continue
            plt.close("all")

    return {"kpi_values": kpi_values, "dashboard_charts": dashboard_charts}


# ── NEW: Custom PDF Class for Watermark ──
class PDFWithWatermark(FPDF):
    def header(self):
        # Save the current Y position so we don't mess up the normal text flow
        current_y = self.get_y()
        
        # Set a very large, light gray font for the watermark
        self.set_font("helvetica", "B", 80)
        self.set_text_color(235, 235, 235) # Very light gray
        
        # Move down to roughly the middle of the page
        self.set_y(140)
        
        # Print the watermark text centered
        self.cell(0, 10, "Datagent", align="C")
        
        # Reset font color to black and restore the original Y position
        self.set_text_color(0, 0, 0)
        self.set_y(current_y)


# ── NEW: PDF Generation Endpoint (With Watermark) ──
@app.post("/download-pdf")
async def generate_pdf_report(request: PDFDownloadRequest):
    try:
        # Use our new custom class instead of the standard FPDF
        pdf = PDFWithWatermark()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Manually calculate effective width
        epw = pdf.w - 30 

        # --- TITLE ---
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 10, "Automated Data Analysis Report", ln=1, align="C")
        pdf.ln(10)

        # --- 1. ANALYSIS TEXT & CODE ---
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "1. AI Insights & Generated Code", ln=1)
        pdf.set_font("helvetica", "", 10)
        
        analysis_path = TEMP_CHARTS_DIR / "last_analysis.txt"
        if analysis_path.exists():
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis_text = f.read()
                # Clean characters for fpdf compatibility
                clean_text = analysis_text.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 6, clean_text)
        else:
            pdf.multi_cell(0, 6, "No recent analysis text found.")
        pdf.ln(10)

        # --- 2. STATIC MATPLOTLIB VISUALS ---
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "2. Static Visualizations", ln=1)
        pdf.ln(5)

        if TEMP_CHARTS_DIR.exists():
            for filename in sorted(os.listdir(TEMP_CHARTS_DIR)):
                if filename.startswith("matplotlib_chart_") and filename.endswith(".png"):
                    img_path = str(TEMP_CHARTS_DIR / filename)
                    pdf.image(img_path, w=epw)
                    pdf.ln(5)

        # --- 3. DASHBOARD SCREENSHOT ---
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "3. Interactive Dashboard Snapshot", ln=1)
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 10, "Reflects the exact filters applied at the time of download.", ln=1)
        pdf.ln(5)

        base64_data = request.dashboard_screenshot
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        image_bytes = base64.b64decode(base64_data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(image_bytes)
            tmp_img_path = tmp_img.name

        pdf.image(tmp_img_path, w=epw)
        os.remove(tmp_img_path)

        # Legacy fpdf output handling
        pdf_string = pdf.output(dest='S')
        if isinstance(pdf_string, str):
            pdf_bytes = pdf_string.encode('latin1')
        else:
            pdf_bytes = bytes(pdf_string)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="Data_Analysis_Report.pdf"'}
        )

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}