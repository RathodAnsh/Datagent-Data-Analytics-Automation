from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import shutil
import os
import polars as pl

# Internal imports
from service import clean_dataset_stream
from datagent import load_memory, save_memory
from measure_service import prepare_dashboard_stream
from api_adapter import dataset_info_from_csv
from measure_creation import (
    ColumnRelevanceAgent,
    MeasureCreationAgent,  # used by /create-measures and /suggest-measures endpoints
    DashboardConfig
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# UPLOAD CONFIG
# =========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# REQUEST MODELS
# =========================
class MeasureRequest(BaseModel):
    cleaned_file: str
    measures: List[Dict]

class DashboardStreamRequest(BaseModel):
    cleaned_file: str
    business_requirements: str
    pinned_columns: List[str] = []

# =========================
# ENDPOINTS
# =========================

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload raw dataset and return dataset metadata + preview
    """
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return dataset_info_from_csv(path)


@app.get("/check-dataset")
def check_dataset(file_path: str):
    """
    Quick heuristic check: is this dataset already clean?

    Returns:
      is_clean   bool   — True if the dataset looks clean enough to skip cleaning
      reason     str    — Human-readable explanation
      stats      dict   — null_cells, null_pct, duplicate_rows, total_rows, total_cols
    """
    try:
        df = pl.read_csv(file_path, infer_schema_length=500, ignore_errors=True)
    except Exception as e:
        return {"is_clean": False, "reason": f"Could not read file: {e}", "stats": {}}

    total_rows = len(df)
    total_cols = len(df.columns)

    # Count actual nulls
    null_cells = sum(df[col].null_count() for col in df.columns)
    null_pct   = round(null_cells / max(total_rows * total_cols, 1) * 100, 2)

    # Count exact duplicate rows
    dup_rows   = int(df.is_duplicated().sum())
    dup_pct    = round(dup_rows / max(total_rows, 1) * 100, 2)

    # Quick string-null probe (values that look like "null", "N/A", "?", "")
    import pandas as pd
    _STR_NULLS = {"", "null", "none", "na", "n/a", "nan", "?", "-", "missing", "unknown"}
    str_null_count = 0
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            vals = df[col].drop_nulls().to_pandas().astype(str).str.strip().str.lower()
            str_null_count += int(vals.isin(_STR_NULLS).sum())
    str_null_pct = round(str_null_count / max(total_rows * total_cols, 1) * 100, 2)

    total_null_pct = null_pct + str_null_pct

    stats = {
        "total_rows":    total_rows,
        "total_cols":    total_cols,
        "null_cells":    null_cells + str_null_count,
        "null_pct":      round(total_null_pct, 2),
        "duplicate_rows": dup_rows,
        "duplicate_pct": dup_pct,
    }

    # Decision thresholds — relaxed so genuinely clean datasets pass
    CLEAN_NULL_PCT  = 1.0   # ≤ 1 % null/placeholder cells → clean
    CLEAN_DUP_PCT   = 0.5   # ≤ 0.5 % duplicate rows        → clean

    if total_null_pct > CLEAN_NULL_PCT:
        return {
            "is_clean": False,
            "reason": (
                f"Dataset has {stats['null_cells']:,} missing / placeholder values "
                f"({total_null_pct:.1f}% of all cells). Please run data cleaning first."
            ),
            "stats": stats,
        }

    if dup_pct > CLEAN_DUP_PCT:
        return {
            "is_clean": False,
            "reason": (
                f"Dataset has {dup_rows:,} duplicate rows ({dup_pct:.1f}%). "
                "Please run data cleaning first to remove them."
            ),
            "stats": stats,
        }

    return {
        "is_clean": True,
        "reason": (
            f"Dataset looks clean — {total_null_pct:.1f}% missing values, "
            f"{dup_pct:.1f}% duplicates. Ready for feature selection."
        ),
        "stats": stats,
    }


@app.get("/clean-stream")
async def clean_stream(file_path: str):
    """
    Stream real-time cleaning steps to frontend
    """
    return StreamingResponse(
        clean_dataset_stream(file_path),
        media_type="application/x-ndjson"
    )


@app.post("/analyze-columns")
def analyze_columns(file_path: str):
    """
    Analyze cleaned dataset columns for dashboard relevance
    """
    api_key = DashboardConfig.load_api_key()
    agent = ColumnRelevanceAgent(api_key)

    df = pl.read_csv(file_path)
    analysis = agent.analyze_columns(df, requirements="")

    response = []
    for col, info in analysis.items():
        response.append({
            "name": col,
            "dataType": info["data_type"],
            "relevanceScore": int(info["relevance_score"] * 100),
            "suggestedRole": (
                "kpi" if "KPI" in info["role"]
                else "dimension" if "Dimension" in info["role"]
                else "filter"
            ),
            "reason": info["reasoning"],
            "selected": info["relevant"]
        })

    return response


@app.post("/suggest-measures")
def suggest_measures(file_path: str):
    """
    Suggest KPI / calculated measures based on cleaned dataset
    """
    api_key = DashboardConfig.load_api_key()
    agent = MeasureCreationAgent(api_key)

    df = pl.read_csv(file_path)
    measures = agent.identify_measures(
        df,
        kept_cols=list(df.columns),
        requirements=""
    )

    response = []
    for idx, m in enumerate(measures):
        response.append({
            "id": f"measure_{idx}",
            "name": m["name"],
            "description": m.get("business_value", ""),
            "formula": m.get("formula_desc", ""),
            "category": "KPI",
            "selected": False
        })

    return response


@app.post("/create-measures")
def create_measures(req: MeasureRequest):
    """
    Apply selected measures to the cleaned dataset and return dashboard-ready file.
    req.cleaned_file  — path to cleaned_dataset.csv
    req.measures      — list of measure dicts selected by the user on FeaturesPage
    """
    api_key = DashboardConfig.load_api_key()
    agent = MeasureCreationAgent(api_key)

    import pandas as pd
    df = pd.read_csv(req.cleaned_file)

    # create_measures() applies the selected measure formulas as new columns
    df_with_measures = agent.create_measures(df, req.measures)

    output_file = "dashboard_dataset.csv"
    df_with_measures.to_csv(output_file, index=False)

    return dataset_info_from_csv(output_file)

@app.post("/prepare-dashboard-stream")
def prepare_dashboard_stream_api(req: DashboardStreamRequest):
    return StreamingResponse(
        prepare_dashboard_stream(req.cleaned_file, req.business_requirements, req.pinned_columns),
        media_type="application/x-ndjson"
    )

@app.get("/get-dashboard-context")
def get_dashboard_context(requirements: str = ""):
    """
    Returns the three things needed for dashboard/insight generation:
      1. dataset_profile  — full column-level profile of dashboard_dataset.csv (JSON)
      2. requirements     — the user's business requirements string
      3. dataset_file     — path to the final dashboard-ready CSV

    Called when the user clicks "Proceed to Dashboard" on FeaturesPage.
    The profile was written to dashboard_profile.json by measure_service.py
    at the end of the feature-selection pipeline.
    """
    import json as _json

    profile_path = "dashboard_profile.json"
    dataset_path = "dashboard_dataset.csv"

    if not os.path.exists(profile_path):
        return {"error": "Profile not found. Please run feature selection first."}

    if not os.path.exists(dataset_path):
        return {"error": "Dashboard dataset not found. Please run feature selection first."}

    with open(profile_path, "r") as f:
        profile = _json.load(f)

    return {
        "dataset_profile":  profile,
        "requirements":     requirements,
        "dataset_file":     dataset_path,
        "dataset_file_url": f"http://localhost:8000/download?path={dataset_path}",
    }


@app.get("/download")
def download_file(path: str):
    if not os.path.exists(path):
        return {"error": "File not found"}

    return FileResponse(
        path,
        filename=os.path.basename(path),
        media_type="text/csv"
    )

@app.post("/save-memory")
async def save_memory_api(payload: dict):

    memory = load_memory()

    col = payload["column"]
    frm = payload["from"]
    to = payload["to"]

    if col not in memory:
        memory[col] = {}

    memory[col][frm] = to

    save_memory(memory)

    return {"status": "saved"}