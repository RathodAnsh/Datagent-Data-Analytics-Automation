from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any
from fpdf import FPDF
import pandas as pd
from dotenv import load_dotenv
import os
import warnings
import io
import shutil
import glob
import json
import math
import re
import subprocess
import base64
import tempfile
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px

# ==================== CONFIGURATION ====================
warnings.filterwarnings("ignore")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception:
        pass

TEMP_CHARTS_DIR = Path("temp_charts")
TEMP_DATA_DIR   = Path("temp_data")
PROFILES_DIR    = Path("data_profiles")

for d in [TEMP_CHARTS_DIR, TEMP_DATA_DIR, PROFILES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clear temp dirs on every server startup (same as Python main.py)
for _cleanup_dir in [TEMP_CHARTS_DIR, TEMP_DATA_DIR]:
    for _f in _cleanup_dir.glob("*"):
        try:
            if _f.is_file():
                _f.unlink()
        except Exception:
            pass


# ==================== HELPER: Find Rscript ====================

def _find_rscript() -> str:
    """Locate Rscript binary cross-platform (same approach as original app.py)."""
    r_exec = shutil.which("Rscript")
    if not r_exec and os.name == "nt":
        common_paths = sorted(
            glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True
        )
        if common_paths:
            r_exec = common_paths[0]
    return r_exec or ""


def _run_rscript(r_code: str, r_file_path: str) -> tuple[str, bool]:
    """
    R EXECUTION
    1. Auto-installs missing packages in the exact R environment Python is using.
    2. Never swallows real error messages.
    """
    # Force R to install missing packages internally so no path mismatch occurs
    install_script = """
    options(repos = c(CRAN = "https://cloud.r-project.org"))
    needed <- c("tidyverse", "ggplot2", "plotly", "scales", "lubridate", "jsonlite")
    missing <- needed[!(needed %in% installed.packages()[,"Package"])]
    if(length(missing)) install.packages(missing, quiet=TRUE)
    """
    
    # Clean all invisible characters to prevent syntax crashes
    clean_r_code = r_code.replace('\xa0', ' ').replace('\u200b', '')
    final_code = install_script + "\n" + clean_r_code

    with open(r_file_path, "w", encoding="utf-8") as f:
        f.write(final_code)

    r_exec = _find_rscript()
    if not r_exec:
        return "❌ Error: 'Rscript' not found. Please check R installation.", False

    try:
        result = subprocess.run(
            [r_exec, r_file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        # Stop Python from swallowing real errors!
        # We only ignore useless startup noise, NOT real warnings/errors.
        clean_err = []
        for line in (result.stderr or "").splitlines():
            if any(noise in line for noise in ["Attaching package", "masked", "Welcome to R"]):
                continue
            if line.strip():
                clean_err.append(line)
        
        err_msg = "\n".join(clean_err).strip()

        if result.returncode != 0:
            return f"{err_msg}\n\nStdout: {result.stdout}", False

        return result.stdout, True
    except Exception as e:
        return f"System Error: {str(e)}", False


# ==================== SAME HELPER FUNCTIONS AS Python main.py ====================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"[^a-zA-Z0-9]", "_", str(c).strip().lower()).strip("_")
        for c in df.columns
    ]
    return df


def convert_smart_types(df: pd.DataFrame):
    """Exact same logic as Python main.py — preserve 100%."""
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

        numeric_pattern = r"^[$€£\s]*[\d,]+(\.\d+)?[\s]*$"
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
                for fmt, label in [
                    ("%Y-%m-%d", "ISO"),
                    (None, "Mixed"),
                    (None, "Inferred"),
                ]:
                    try:
                        if label == "Mixed":
                            normalized = (
                                df[col].astype(str)
                                .str.replace("/", "-", regex=False)
                                .str.replace(".", "-", regex=False)
                            )
                            converted_col = pd.to_datetime(normalized, errors="coerce", dayfirst=True)
                        elif label == "Inferred":
                            converted_col = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                        else:
                            converted_col = pd.to_datetime(df[col], format=fmt, errors="coerce")

                        if converted_col.notna().sum() / len(non_null_series) > 0.7:
                            df[col] = converted_col
                            conversions.append(f"📅 Converted '{col}' to DateTime ({label})")
                            break
                    except Exception:
                        continue
        except Exception:
            pass

        if df[col].dtype == "object":
            df[col] = df[col].astype(str).replace(["None", "none", "null", "NULL", "NONE"], "")

    return df, conversions


def generate_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Same as Python main.py — generates profile JSON."""
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
            col_info["min"]  = float(df[col].min())  if not df[col].isna().all() else 0.0
            col_info["max"]  = float(df[col].max())  if not df[col].isna().all() else 0.0
            col_info["mean"] = float(df[col].mean()) if not df[col].isna().all() else 0.0

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


def load_csv_and_clean(file_bytes: bytes):
    """Same as Python main.py."""
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
        removed = duplicates_before - len(df)
        if removed > 0:
            logs.append(f"🔁 Removed {removed} duplicate row(s)")

        df, type_logs = convert_smart_types(df)
        logs.extend(type_logs)

        TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TEMP_DATA_DIR / "source.csv", index=False)

        return df, logs
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


def _count_questions(user_query: str) -> int:
    """Same as Python main.py."""
    parts = re.split(r'\n|\?\s*\d+[\.\)]|\d+[\.\)]\s', user_query)
    questions = [p.strip() for p in parts if len(p.strip()) > 10]
    if len(questions) <= 1:
        questions = [p.strip() for p in user_query.split('?') if len(p.strip()) > 10]
    return max(len(questions), 1)


def _select_kpis_and_slicers(df: pd.DataFrame, profile_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Same AI-driven KPI/slicer selection as Python main.py.
    Uses Gemini to select relevant KPIs and slicers.
    """
    col_summary = []
    for col, info in profile_data.get("column_details", {}).items():
        col_summary.append(
            f"- {col}: dtype={info['dtype']}, unique={info['unique_values']}, missing={info['missing_values']}"
        )

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

        valid_kpis = []
        for kpi in config.get("kpis", []):
            col = kpi.get("column", "")
            if col == "index" or col in df.columns:
                valid_kpis.append(kpi)
        config["kpis"] = valid_kpis[:5]

        valid_slicers = [s for s in config.get("slicers", []) if s in df.columns]
        config["slicers"] = valid_slicers[:4]

        if not config["kpis"]:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            config["kpis"] = [
                {"label": c.replace("_", " ").title(), "column": c, "agg": "sum", "fmt": "number"}
                for c in numeric_cols[:4]
            ]
        if not config["slicers"]:
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            config["slicers"] = cat_cols[:3]

        return config

    except Exception:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        return {
            "kpis": [
                {"label": c.replace("_", " ").title(), "column": c, "agg": "sum", "fmt": "number"}
                for c in numeric_cols[:4]
            ],
            "slicers": cat_cols[:3],
        }


def _calculate_kpis(df: pd.DataFrame, kpi_config: List[Dict]) -> List[Dict]:
    """
    Same pure-pandas KPI calculation as Python main.py.
    No AI — deterministic.
    """
    results = []
    for kpi in kpi_config:
        col   = kpi.get("column", "")
        agg   = kpi.get("agg", "sum")
        label = kpi.get("label", col)
        fmt   = kpi.get("fmt", "number")

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


# ==================== R ANALYSIS ENGINE ====================

def _inline_html_dependencies(html_path: Path) -> bool:
    """
    Same inline_html_dependencies logic from original app.py / dashboard.py.
    Fixes blank dashboards by embedding JS/CSS directly into HTML.
    """
    if not html_path.exists():
        return False
    try:
        content = html_path.read_text(encoding="utf-8")
        parent_dir = html_path.parent

        def replace_script(match):
            src = match.group(1)
            if src.startswith("http") or src.startswith("//"):
                return match.group(0)
            local = parent_dir / src
            if local.exists():
                try:
                    return f"<script>\n{local.read_text(encoding='utf-8')}\n</script>"
                except Exception:
                    pass
            return match.group(0)

        def replace_style(match):
            href = match.group(1)
            if href.startswith("http") or href.startswith("//"):
                return match.group(0)
            local = parent_dir / href
            if local.exists():
                try:
                    return f"<style>\n{local.read_text(encoding='utf-8')}\n</style>"
                except Exception:
                    pass
            return match.group(0)

        content = re.sub(r'<script src="([^"]+)"></script>', replace_script, content)
        content = re.sub(r'<link href="([^"]+)" rel="stylesheet" />', replace_style, content)
        html_path.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def run_analysis(df: pd.DataFrame, profile_data: Dict[str, Any], user_query: str):
    """
    R Analysis Engine.
    Mirrors the Python main.py run_analysis() flow exactly:
      STEP A → count questions
      STEP B → select KPIs + slicers via Gemini
      STEP C → calculate KPI values (pure pandas, no AI)
      STEP D → Gemini writes R code → Rscript executes it → PNGs saved
      STEP E → Read PNGs → base64
      STEP F → Build dashboard_charts.json from plotly JSON files R generates
    """
    if not GENAI_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set on the backend.")

    # ── STEP A: Count questions ──────────────────────────────────────────────
    num_questions = _count_questions(user_query)

    # ── STEP B: Select KPIs + slicers (Gemini, same as Python main.py) ──────
    kpi_slicer_config = _select_kpis_and_slicers(df, profile_data, user_query)

    # ── STEP C: Calculate KPI values (pure pandas, same as Python main.py) ──
    kpi_values = _calculate_kpis(df, kpi_slicer_config.get("kpis", []))

    full_kpi_config = {
        "slicers":    kpi_slicer_config.get("slicers", []),
        "kpis":       kpi_slicer_config.get("kpis", []),
        "kpi_values": kpi_values,
    }
    TEMP_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEMP_CHARTS_DIR / "kpi_config.json", "w") as f:
        json.dump(full_kpi_config, f, indent=2)

    # ── STEP D: Gemini writes R code ─────────────────────────────────────────
    abs_temp_charts = TEMP_CHARTS_DIR.resolve()
    abs_temp_charts.mkdir(parents=True, exist_ok=True)

    source_csv_path = str((TEMP_DATA_DIR / "source.csv").resolve()).replace("\\", "/")
    charts_dir_path = str(abs_temp_charts).replace("\\", "/")
    profile_json    = json.dumps(profile_data, indent=2)

    r_prompt = f"""You are an expert R Data Analyst.

**Context:**
- Dataset is already saved at: '{source_csv_path}'
- Charts must be saved to: '{charts_dir_path}/'

**Dataset Profile:**
{profile_json}

**User Query:** "{user_query}"

**CRITICAL — Number of questions detected: {num_questions}**
**You MUST generate EXACTLY {num_questions} chart(s). Not more, not less.**
**KPIs already selected (DO NOT change):** {json.dumps(kpi_slicer_config.get('kpis', []))}
**Slicers already selected (DO NOT change):** {json.dumps(kpi_slicer_config.get('slicers', []))}

### MANDATORY INSTRUCTIONS:

1. **INITIAL SETUP (ALWAYS at the top):**
options(warn=-1)
options(scipen=999)
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(plotly))
suppressPackageStartupMessages(library(scales))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(jsonlite))
df <- read.csv('{source_csv_path}', stringsAsFactors=FALSE)
CHARTS_DIR <- '{charts_dir_path}'

2. **EXHAUSTIVE VISUALIZATION — STRICT:**
   - The user asked EXACTLY {num_questions} question(s).
   - You MUST generate EXACTLY {num_questions} ggplot2 chart(s) saved as PNG.
   - Each question maps to exactly ONE chart. No bundling. No skipping.
   - Count your charts before finishing. If you have {num_questions - 1}, add one more.
   - Save each chart:
     ggsave(file.path(CHARTS_DIR, "matplotlib_chart_1.png"), plot_1, width=10, height=6, dpi=120)
     ggsave(file.path(CHARTS_DIR, "matplotlib_chart_2.png"), plot_2, width=10, height=6, dpi=120)
     (continue for all {num_questions} charts)

3. **PLOTLY INTERACTIVE CHARTS — for every chart immediately after ggsave:**
   Convert each ggplot to plotly and save as JSON like this:
     plotly_1 <- ggplotly(plot_1)
     plotly_list_1 <- list(data=plotly_1$x$data, layout=plotly_1$x$layout)
     writeLines(toJSON(plotly_list_1, auto_unbox=TRUE, pretty=FALSE, null="null"), file.path(CHARTS_DIR, "plotly_chart_1.json"))

     plotly_2 <- ggplotly(plot_2)
     plotly_list_2 <- list(data=plotly_2$x$data, layout=plotly_2$x$layout)
     writeLines(toJSON(plotly_list_2, auto_unbox=TRUE, pretty=FALSE, null="null"), file.path(CHARTS_DIR, "plotly_chart_2.json"))

     (do this for ALL {num_questions} charts — always use plotly_N$x$data and plotly_N$x$layout)

4. **DASHBOARD CHARTS JSON (CRITICAL) — at the very end:**
all_charts <- list()
for (i in seq_len({num_questions})) {{
  json_file <- file.path(CHARTS_DIR, paste0("plotly_chart_", i, ".json"))
  if (file.exists(json_file)) {{
    plotly_str <- paste(readLines(json_file, warn=FALSE), collapse="")
    all_charts[[length(all_charts)+1]] <- list(title=paste("Chart", i), plotly_json=plotly_str)
  }}
}}
write_json(list(charts=all_charts), file.path(CHARTS_DIR, "dashboard_charts.json"), auto_unbox=TRUE)

5. **DO NOT OVERWRITE kpi_config.json.** It is already saved.

6. **NO DUMMY DATA.** Use `df` directly. Never create mock data frames.

7. **AGGREGATION FIRST:**
   - Always aggregate before plotting:
     df_q1 <- df %>% filter(!is.na(col)) %>% group_by(...) %>% summarise(Total=sum(..., na.rm=TRUE)) %>% ungroup()

8. **DATE HANDLING:**
   - Convert date cols: df$date_col <- suppressWarnings(parse_date_time(df$date_col, orders=c("ymd","dmy","mdy")))

9. **SMART TIME AGGREGATION:**
   - unique time values > 50 → aggregate by month using floor_date(date_col, "month")
   - unique time values > 20 → aggregate by week using floor_date(date_col, "week")
   - else → use original values

10. **CHART TYPE SELECTION:**
    - trend/over time → geom_line()
    - category comparison → geom_col()
    - distribution → geom_histogram()
    - relationship → geom_point()
    - proportion → geom_col()
    - CRITICAL: NEVER use coord_polar() or pie charts (ggplotly crashes on them).
    - CRITICAL: NEVER use geom_smooth(), stat_smooth(), or trendlines.
    
11. **THEME:**
    - Use theme_minimal() for all charts
    - Use scale_y_continuous(labels=scales::comma) for large numbers
    - Use scale_y_continuous(labels=scales::dollar) for currency
    - Always add labs(title="...", x="...", y="...")

12. **ANALYSIS SUMMARY TEXT:**
    - At the end, print a summary using cat():
      cat("--- Analysis Summary ---\n")
      cat("Key finding 1\n")
      cat("Key finding 2\n")

13. **FILTER NULL VALUES before each chart:**
    df_q1 <- df %>% filter(!is.na(column_name))

### OUTPUT: Return ONLY clean R code. No markdown fences. No triple backticks. No explanation.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(r_prompt)
    r_code = response.text

    # Extract R code if wrapped in markdown fences
    match = re.search(r"```[rR]\n([\s\S]*?)```", r_code)
    if match:
        r_code = match.group(1).strip()
    elif "```" in r_code:
        r_code = r_code.replace("```", "").strip()

    # Clear old chart files
    for pattern in ["*.png", "*.html", "plotly_chart_*.json", "dashboard_charts.json"]:
        for f in glob.glob(str(abs_temp_charts / pattern)):
            try:
                os.remove(f)
            except Exception:
                pass

    # Execute R code
    r_file_path = str((Path(".") / "temp_analysis.R").resolve())
    output_text, success = _run_rscript(r_code, r_file_path)

    # ── RETRY if chart count wrong (mirrors Python main.py retry logic) ──────────
    png_files = sorted(glob.glob(str(abs_temp_charts / "matplotlib_chart_*.png")))
    if len(png_files) < num_questions:
        retry_prompt = f"""The previous R code only generated {len(png_files)} chart(s).
The user asked EXACTLY {num_questions} question(s). You MUST generate EXACTLY {num_questions} chart(s).
User Query: "{user_query}"
Dataset columns: {list(df.columns)}
Charts dir: '{charts_dir_path}'
Source CSV: '{source_csv_path}'
Write COMPLETE R code with EXACTLY {num_questions} ggsave() + ggplotly() + writeLines() calls.
Also write dashboard_charts.json at the end combining all plotly JSONs.
Use: p <- ggplotly(plot_N); writeLines(toJSON(list(data=p$x$data, layout=p$x$layout), auto_unbox=TRUE, null="null"), file.path(CHARTS_DIR, paste0("plotly_chart_",N,".json")))
RETURN ONLY RAW R CODE. No markdown."""
        retry_response = model.generate_content(retry_prompt)
        retry_code = retry_response.text
        m2 = re.search(r"```[rR]\n([\s\S]*?)```", retry_code)
        if m2:
            retry_code = m2.group(1).strip()
        elif "```" in retry_code:
            retry_code = retry_code.replace("```", "").strip()
        retry_out, retry_ok = _run_rscript(retry_code, r_file_path)
        if retry_ok:
            r_code = retry_code
            output_text = retry_out

    # Persist R code for dashboard slicer updates (same as Python main.py saves last_analysis.txt)
    # 🚨 CRITICAL FIX: Agar R crash ho gaya aur PNG nahi bana, toh exact error throw karo!
    final_png_files = sorted(glob.glob(str(abs_temp_charts / "matplotlib_chart_*.png")))
    if len(final_png_files) == 0:
        raise ValueError(f"R Engine Crashed! Failed to make charts.\nExact Error:\n{output_text}")
    last_code_path = abs_temp_charts / "last_analysis.txt"
    try:
        last_code_path.write_text(r_code, encoding="utf-8")
    except Exception:
        pass

    # Inline HTML if R generated an HTML file (same as original app.py)
    for html_file in abs_temp_charts.glob("*.html"):
        _inline_html_dependencies(html_file)

    # ── STEP E: Read PNGs → base64 (same as Python main.py) ─────────────────
    chart_files = sorted(glob.glob(str(abs_temp_charts / "matplotlib_chart_*.png")))
    # Fallback: any PNG
    if not chart_files:
        chart_files = sorted(glob.glob(str(abs_temp_charts / "*.png")))

    charts_base64: List[str] = []
    for fpath in chart_files:
        try:
            with open(fpath, "rb") as img:
                charts_base64.append(base64.b64encode(img.read()).decode("utf-8"))
        except Exception:
            continue

    # ── STEP F: Build dashboard_charts.json (same as Python main.py) ─────────
    dashboard_charts_path = abs_temp_charts / "dashboard_charts.json"

    # If R didn't build it, build from individual plotly JSON files
    if not dashboard_charts_path.exists():
        plotly_files = sorted(glob.glob(str(abs_temp_charts / "plotly_chart_*.json")))
        if plotly_files:
            charts_list = []
            for i, fp in enumerate(plotly_files):
                try:
                    pjson = Path(fp).read_text(encoding="utf-8")
                    charts_list.append({"title": f"Chart {i+1}", "plotly_json": pjson})
                except Exception:
                    continue
            with open(str(dashboard_charts_path), "w") as f:
                json.dump({"charts": charts_list}, f)

    # Check dashboard_ready (same logic as Python main.py)
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

    return {
        "summary": output_text or "✅ R Analysis complete.",
        "charts":  charts_base64,
        "dashboard_ready": dashboard_ready,
        "last_analysis_code": r_code,   # shown as "Generated R Code" in frontend
        "kpi_values": kpi_values,
    }


# ==================== FASTAPI APP & ENDPOINTS ====================

app = FastAPI(title="AI Data Analyst Agent API — R Edition")

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
            "rows":    int(df.shape[0]),
            "columns": int(df.shape[1]),
            "logs":    logs,
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


@app.get("/dashboard-config")
async def dashboard_config():
    """Same response shape as Python main.py — Next.js dashboard works unchanged."""
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
                    slicer_meta.append({"name": str(raw_col), "values": []})
                    continue
                col_map = {c.lower(): c for c in df.columns}
                col = col_map.get(str(raw_col).lower())
                if col and col in df.columns:
                    try:
                        values = df[col].dropna().astype(str).unique().tolist()
                    except Exception:
                        values = []
                    slicer_meta.append({"name": col, "values": values})
        except Exception:
            pass

    kpi_values: List[Dict[str, Any]] = []
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            kpi_values = cfg.get("kpi_values", [])
        except Exception:
            pass

    dashboard_charts: List[Dict[str, Any]] = []
    dashboard_charts_path = TEMP_CHARTS_DIR / "dashboard_charts.json"
    if dashboard_charts_path.exists():
        try:
            with open(dashboard_charts_path, "r", encoding="utf-8") as f:
                dc = json.load(f)
            dashboard_charts = dc.get("charts", [])
        except Exception:
            pass

    # PNG charts as base64 (same as Python main.py)
    png_charts: List[Dict[str, Any]] = []
    abs_charts = TEMP_CHARTS_DIR.resolve()
    for fpath in sorted(glob.glob(str(abs_charts / "*.png"))):
        try:
            with open(fpath, "rb") as img:
                encoded = base64.b64encode(img.read()).decode("utf-8")
                png_charts.append({"filename": Path(fpath).name, "image_b64": encoded})
        except Exception:
            continue

    return {
        "slicers":          slicer_meta,
        "kpi_values":       kpi_values,
        "dashboard_charts": dashboard_charts,
        "png_charts":       png_charts,
    }


@app.post("/dashboard-update")
async def dashboard_update(payload: Dict[str, Any]):
    """
    Same as Python main.py /dashboard-update.
    Applies filters → recalculates KPIs → re-runs R code on filtered data → returns updated charts.
    """
    filters: Dict[str, List[str]] = payload.get("filters") or {}
    data_path = TEMP_DATA_DIR / "source.csv"
    df_filtered = None

    if data_path.exists():
        try:
            try:
                df = pd.read_csv(data_path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(data_path, encoding="latin1")
            df_filtered = df.copy()
            for raw_col, values in filters.items():
                if not values:
                    continue
                col_map = {c.lower(): c for c in df_filtered.columns}
                col = col_map.get(str(raw_col).lower())
                if col and col in df_filtered.columns:
                    try:
                        df_filtered = df_filtered[df_filtered[col].astype(str).isin(values)]
                    except Exception:
                        continue
        except Exception:
            df_filtered = None

    # Recalculate KPIs on filtered data (pure pandas, same as Python main.py)
    kpi_values = []
    config_path = TEMP_CHARTS_DIR / "kpi_config.json"
    if config_path.exists() and df_filtered is not None:
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            kpi_values = _calculate_kpis(df_filtered, cfg.get("kpis", []))
        except Exception:
            pass

    # Re-run R code on filtered data (equivalent of exec(last_analysis.txt, df=df_filtered) in Python)
    dashboard_charts = []
    last_code_path = TEMP_CHARTS_DIR / "last_analysis.txt"

    if last_code_path.exists() and df_filtered is not None:
        try:
            r_code = last_code_path.read_text(encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load R code: {e}")

        # Save filtered CSV
        filtered_csv_path = str((TEMP_DATA_DIR / "filtered_dashboard.csv").resolve()).replace("\\", "/")
        df_filtered.to_csv(filtered_csv_path, index=False)

        # Use persistent folder instead of tempfile.TemporaryDirectory
        filter_out_dir = (TEMP_CHARTS_DIR / "filtered_run").resolve()
        filter_out_dir.mkdir(parents=True, exist_ok=True)
        # Clear previous run files
        for old_f in filter_out_dir.glob("*"):
            try:
                old_f.unlink()
            except Exception:
                pass
        filter_out_str = str(filter_out_dir).replace("\\", "/")

        # Patch 1: read.csv → filtered CSV
        patched_code = re.sub(
            r"read\.csv\s*\(\s*['\"].*?['\"]",
            f"read.csv('{filtered_csv_path}'",
            r_code, count=1, flags=re.IGNORECASE
        )

        # Patch 2: CHARTS_DIR → persistent filtered_run dir
        patched_code = re.sub(
            r"CHARTS_DIR\s*<-\s*['\"][^\'\"].*?['\"]",
            f"CHARTS_DIR <- '{filter_out_str}'",
            patched_code
        )

        r_filter_path = str((Path(".") / "temp_analysis_filtered.R").resolve())
        _run_rscript(patched_code, r_filter_path)

        # Read updated Plotly charts from persistent filtered dir
        tmp_dc = filter_out_dir / "dashboard_charts.json"
        if tmp_dc.exists():
            try:
                with open(str(tmp_dc), "r", encoding="utf-8") as f:
                    dashboard_charts = json.load(f).get("charts", [])
            except Exception:
                pass

        # Fallback: read individual plotly JSON files
        if not dashboard_charts:
            for fp in sorted(glob.glob(str(filter_out_dir / "plotly_chart_*.json"))):
                try:
                    pjson = Path(fp).read_text(encoding="utf-8")
                    dashboard_charts.append({"title": Path(fp).stem, "plotly_json": pjson})
                except Exception:
                    continue

    return {"kpi_values": kpi_values, "dashboard_charts": dashboard_charts}


# ── NEW: PDF Generation Endpoint ──
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
    return {"status": "ok", "engine": "R"}