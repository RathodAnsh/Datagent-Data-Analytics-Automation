"""
DATAAGENT PHASE 3 — STREAMING SERVICE

Exposes prepare_dashboard_stream() which is called by the FastAPI route
/prepare-dashboard-stream.  Yields JSON-lines so the frontend can update
its progress cards in real time.

Pipeline order:
  1. Load dataset (all-string, same as datagent_v2 loader)
  2. Parse user intent → extract dimension/metric keywords
  3. Score every column for relevance (rule-based + one LLM batch call),
     passing user requirements so intent boosts relevant columns
  4. Filter to relevant columns — irrelevant features dropped,
     primary/foreign key columns (ID, _no, _code) always retained
  5. Derive time / categorical features based on intent
  6. Write dashboard_dataset.csv and stream final metadata

Note: No KPI scalar columns are appended. Feature selection mirrors
      the Power BI workflow — clean + select, ready for the report layer.
"""

import json
import re
import os
import pandas as pd
import polars as pl
from typing import Dict, List, Generator

from measure_creation import (
    DashboardConfig,
    ColumnRelevanceAgent,
)


# ============================================================================
# INTENT PARSING
# ============================================================================

# Keyword → intent flag mapping
_INTENT_MAP = {
    "time_grouping":         {"time", "trend", "monthly", "yearly", "quarterly",
                               "date", "year", "month", "week", "period"},
    "category_grouping":     {"category", "product", "item", "brand", "segment",
                               "type", "class"},
    "location_grouping":     {"location", "region", "city", "state", "store",
                               "branch", "geography", "area", "zone"},
    "customer_grouping":     {"customer", "client", "buyer", "segment",
                               "demographic", "age", "gender"},
    "discount_analysis":     {"discount", "promo", "offer", "coupon", "rebate"},
    "profitability_analysis": {"profit", "margin", "cost", "profitability"},
}


def parse_intent(requirements: str) -> Dict[str, bool]:
    tokens = set(re.sub(r"[^a-z\s]", " ", requirements.lower()).split())
    return {flag: bool(tokens & kws) for flag, kws in _INTENT_MAP.items()}


# ============================================================================
# FEATURE DERIVATION
# ============================================================================

# Date formats the pipeline may produce (most-likely first)
_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]


def _detect_date_format(series: pl.Series) -> str:
    """Try candidate formats on a sample; return the best match."""
    sample = series.drop_nulls().head(200).to_pandas().astype(str)
    best_fmt, best_score = "", 0
    for fmt in _DATE_FORMATS:
        try:
            score = pd.to_datetime(sample, format=fmt, errors="coerce").notna().sum()
            if score > best_score:
                best_score, best_fmt = score, fmt
        except Exception:
            pass
    return best_fmt


def _is_date_col(series: pl.Series) -> bool:
    return bool(_detect_date_format(series))


def derive_features(df: pl.DataFrame, intent: Dict[str, bool]) -> pl.DataFrame:
    """
    Add derived columns based on what the user wants to analyse.
    All date parsing is format-detected, not hardcoded.
    """
    new_cols: List = []

    # ── Time features ──────────────────────────────────────────────────────
    if intent.get("time_grouping"):
        for col in df.columns:
            if not any(kw in col.lower() for kw in ("date", "time", "created", "ordered")):
                continue
            fmt = _detect_date_format(df[col])
            if not fmt:
                continue
            tmp = f"__tmp_{col}"
            try:
                df = df.with_columns(
                    pl.col(col).str.strptime(pl.Date, format=fmt, strict=False).alias(tmp)
                )
                new_cols += [
                    pl.col(tmp).dt.year().alias(f"{col}_Year"),
                    pl.col(tmp).dt.month().alias(f"{col}_Month"),
                    pl.col(tmp).dt.quarter().alias(f"{col}_Quarter"),
                ]
            except Exception as e:
                print(f"  [derive_features] Date parse failed for '{col}': {e}")

    # ── Category normalisation ─────────────────────────────────────────────
    if intent.get("category_grouping"):
        for col in df.columns:
            if any(kw in col.lower() for kw in ("category", "product", "brand", "item", "type")):
                if df[col].dtype == pl.Utf8:
                    new_cols.append(
                        pl.col(col).str.to_lowercase()
                           .str.replace_all(r"\s+", "_")
                           .alias(f"{col}_key")
                    )

    # ── Location normalisation ─────────────────────────────────────────────
    if intent.get("location_grouping"):
        for col in df.columns:
            if any(kw in col.lower() for kw in ("city", "state", "region", "location", "store", "branch")):
                if df[col].dtype == pl.Utf8:
                    new_cols.append(
                        pl.col(col).str.to_lowercase()
                           .str.replace_all(r"\s+", "_")
                           .alias(f"{col}_key")
                    )

    if new_cols:
        df = df.with_columns(new_cols)

    # Drop temp parsed columns
    tmp_cols = [c for c in df.columns if c.startswith("__tmp_")]
    if tmp_cols:
        df = df.drop(tmp_cols)

    return df


# ============================================================================
# DATASET METADATA HELPER
# ============================================================================

def _dataset_info(file_path: str) -> Dict:
    df = pd.read_csv(file_path, nrows=5, dtype=str)
    total = sum(1 for _ in open(file_path)) - 1  # fast line count
    return {
        "filePath":    file_path,
        "fileName":    os.path.basename(file_path),
        "totalRows":   total,
        "totalColumns": len(df.columns),
        "columns":     df.columns.tolist(),
        "previewData": df.to_dict(orient="records"),
    }


# ============================================================================
# MAIN STREAMING PIPELINE
# ============================================================================

def prepare_dashboard_stream(
    cleaned_file: str,
    business_requirements: str,
    pinned_columns: List[str] = None,
) -> Generator[str, None, None]:
    """
    Streaming generator — yields JSON-line strings.
    Each line: {"step": "<id>", "status": "processing"|"completed", ...payload}
    """

    api_key = DashboardConfig.load_api_key()

    # ── STEP 1: Intent ────────────────────────────────────────────────────
    yield json.dumps({"step": "intent", "status": "processing"}) + "\n"

    intent = parse_intent(business_requirements)

    yield json.dumps({
        "step":   "intent",
        "status": "completed",
        "intent": intent,
    }) + "\n"

    # ── STEP 2: Column relevance ──────────────────────────────────────────
    yield json.dumps({"step": "columns", "status": "processing"}) + "\n"

    # Load with all-string (same approach as datagent_v2 loader to preserve IDs/dates)
    df = pl.from_pandas(
        pd.read_csv(cleaned_file, dtype=str, keep_default_na=False, na_values=[""])
    )

    agent     = ColumnRelevanceAgent(api_key)
    # Pass pinned_columns into analyze_columns so the hard-override runs INSIDE
    # the agent — after LLM review — meaning pinned cols are always marked
    # relevant=True before the relevance dict is ever returned or emitted.
    relevance = agent.analyze_columns(df, business_requirements,
                                      pinned_columns=pinned_columns)

    yield json.dumps({
        "step":             "columns",
        "status":           "completed",
        "column_relevance": relevance,   # pinned cols already show relevant=True here
    }) + "\n"

    # ── STEP 3: Apply — filter + derive features (pure feature selection) ──
    yield json.dumps({"step": "apply", "status": "processing"}) + "\n"

    # kept_cols derived from relevance — pinned cols are already relevant=True
    kept_cols    = [c for c, info in relevance.items() if info["relevant"]]
    dropped_cols = [c for c, info in relevance.items() if not info["relevant"]]
    df_kept      = df.select(kept_cols)

    # Derive time/category features based on intent (no KPI columns added)
    df_derived = derive_features(df_kept, intent)
    df_pd      = df_derived.to_pandas()

    output_file = "dashboard_dataset.csv"
    df_pd.to_csv(output_file, index=False)

    yield json.dumps({
        "step":           "apply",
        "status":         "completed",
        "columnsKept":    kept_cols,
        "columnsDropped": dropped_cols,
    }) + "\n"

    # ── STEP 4: Finalize — profile the final dataset and save to disk ────
    final_info = _dataset_info(output_file)

    # Build a full column-level profile of the dashboard-ready dataset
    # Format matches the reference schema:
    #   { rows, num_columns, column_list, column_details: { col: { dtype, unique_values, missing_values, min, max, mean } } }
    df_final_pd = pd.read_csv(output_file)   # read with inferred dtypes for accurate dtype strings

    # Date-keyword probe — columns whose name contains these words are
    # checked for parseable date values and labelled datetime64[us].
    _DATE_KEYWORDS = ("date", "time", "created", "ordered", "shipped",
                      "delivery", "return", "timestamp", "period")

    column_details = {}
    for col in df_final_pd.columns:
        series   = df_final_pd[col]
        non_null = series.dropna()

        dtype_str = str(series.dtype)

        # ── Detect datetime columns ──────────────────────────────────────────
        # CSV has no native datetime type so pandas reads date columns as
        # object/str. We detect them by:
        #   1. Column name contains a date keyword  AND
        #   2. At least 80% of non-null values parse successfully as dates.
        if dtype_str == "object" and any(kw in col.lower() for kw in _DATE_KEYWORDS):
            sample = non_null.astype(str).head(200)
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.80:
                dtype_str = "datetime64[us]"

        # Map remaining object dtype to "str"
        if dtype_str == "object":
            dtype_str = "str"

        cp = {
            "dtype":          dtype_str,
            "unique_values":  int(series.nunique()),
            "missing_values": int(series.isna().sum()),
        }

        # Add numeric stats for numeric columns
        if "int" in dtype_str or "float" in dtype_str:
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(numeric) > 0:
                cp["min"]  = float(numeric.min())
                cp["max"]  = float(numeric.max())
                cp["mean"] = round(float(numeric.mean()), 4)
        elif dtype_str == "str":
            # Also try numeric probe for columns not caught by dtype (e.g. price stored as str)
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.notna().mean() > 0.80 and len(numeric.dropna()) > 0:
                cp["dtype"] = "float64" if numeric.dropna().apply(lambda x: x % 1 != 0).any() else "int64"
                cp["min"]   = float(numeric.min())
                cp["max"]   = float(numeric.max())
                cp["mean"]  = round(float(numeric.mean()), 4)

        column_details[col] = cp

    dashboard_profile = {
        "rows":         int(len(df_final_pd)),
        "num_columns":  int(len(df_final_pd.columns)),
        "column_list":  list(df_final_pd.columns),
        "column_details": column_details,
    }

    profile_file = "dashboard_profile.json"
    with open(profile_file, "w") as pf:
        json.dump(dashboard_profile, pf, indent=2)

    yield json.dumps({
        "step":   "finalize",
        "status": "completed",
        "result": {
            "intent":          intent,
            "finalDataset":    final_info,
            "columnsKept":     kept_cols,
            "columnsDropped":  dropped_cols,
            "profileFile":     profile_file,   # tells frontend where profile JSON lives
        },
    }) + "\n"