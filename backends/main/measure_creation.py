"""
DATAAGENT PHASE 3 — FEATURE SELECTION & DASHBOARD PREPARATION

Implements the full hybrid feature-selection framework:

  Step 1  Domain curation      User requirements force-include/exclude columns
                                before any statistics run.
  Step 2a Variance filter       VarianceThreshold removes near-constant numerics.
  Step 2b Univariate analysis   Pearson correlation, Mutual Information, and
                                Chi-squared rank every feature against the
                                primary numeric target (revenue/total).
  Step 2c Collinearity removal  Correlation matrix drops one from each pair
                                whose |r| > 0.90.
  Step 3  Embedded importance   RandomForest feature importances (using the
                                primary target) give a model-based score.
                                RFE is skipped for dashboard context where
                                the "target" is a business metric, not a label.
  Step 4  Score fusion + LLM    All method scores are fused into one final score.
                                ONE Gemini call reviews the batch for domain
                                adjustments and adds plain-English reasoning.

Public surface (used by measure_service.py):
  DashboardConfig       — thresholds + API key
  ColumnRelevanceAgent  — analyze_columns(df, requirements) -> Dict
  MeasureCreationAgent  — identify_measures(df, kept_cols, requirements) -> List
                          create_measures(df_pd, measures) -> pd.DataFrame
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from google import genai


warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardConfig:
    RELEVANCE_THRESHOLD      = 0.50   # final fused score to keep a column
    MAX_CATEGORICAL_UNIQ     = 100    # above this → high-cardinality
    MIN_CATEGORICAL_UNIQ     = 2
    VARIANCE_THRESHOLD       = 0.01   # below this → near-constant, filter out
    COLLINEARITY_THRESHOLD   = 0.90   # |r| above this → redundant pair
    RF_N_ESTIMATORS          = 60     # RandomForest trees for feature importance
    GEMINI_MODEL             = "gemini-1.5-flash"

    @staticmethod
    def load_api_key() -> str:
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        return key


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _call_gemini(client: genai.Client, prompt: str) -> str:
    try:
        resp = client.models.generate_content(
            model=DashboardConfig.GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048),
        )
        return resp.text or ""
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ═══════════════════════════════════════════════════════════════════════════════
# DTYPE PROBE  (all columns arrive as strings from the all-string CSV loader)
# ═══════════════════════════════════════════════════════════════════════════════

_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y"]
_BOOL_VALUES  = {"true", "false", "1", "0", "yes", "no", "t", "f", "y", "n"}


def _infer_dtype(series: pl.Series) -> str:
    """Infer logical type from a string Series by probing actual values."""
    col_pd = series.drop_nulls().to_pandas()
    if col_pd.empty:
        return "empty"

    cleaned = col_pd.astype(str).str.replace(r"[₹$£€,\s]", "", regex=True)

    # Numeric probe
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().mean() > 0.85:
        # Check for integer: use modulo on the numeric values instead of
        # float.is_integer unbound method, which fails on Python 3.13+
        # when the series contains int (not float) values.
        non_null = numeric.dropna()
        is_int = (non_null % 1 == 0).all()
        return "integer" if is_int else "float"

    # Date probe
    sample = col_pd.astype(str).head(120)
    for fmt in _DATE_FORMATS:
        try:
            if pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() > 0.80:
                return "datetime"
        except Exception:
            pass

    # Boolean probe
    uniq = set(col_pd.astype(str).str.lower().unique())
    if uniq <= _BOOL_VALUES:
        return "boolean"

    return "categorical"


def _to_numeric_series(series: pl.Series) -> Optional[pd.Series]:
    """Return a float pandas Series or None if the column is not numeric."""
    cleaned = series.to_pandas().astype(str).str.replace(r"[₹$£€,\s]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric if numeric.notna().mean() > 0.80 else None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DOMAIN CURATION  (user intent → force-include / force-exclude sets)
# ═══════════════════════════════════════════════════════════════════════════════

# Maps user intent token → column-name keywords that must be force-included.
# Covers sales, HR, healthcare, ecommerce, logistics, and finance domains.
_INTENT_TO_COLUMNS: Dict[str, List[str]] = {
    # ── Time ──────────────────────────────────────────────────────────────────
    "time":          ["date", "time", "created", "ordered", "period", "timestamp"],
    "trend":         ["date", "time", "month", "year"],
    "monthly":       ["date", "month", "time"],
    "yearly":        ["date", "year", "time"],
    "quarterly":     ["date", "quarter", "time"],
    "weekly":        ["date", "week", "time"],
    "daily":         ["date", "day", "time"],

    # ── Sales / Revenue ────────────────────────────────────────────────────────
    "revenue":       ["revenue", "sales", "total", "amount", "income", "turnover"],
    "sales":         ["sales", "revenue", "total", "amount"],
    "profit":        ["profit", "margin", "net", "earnings", "ebitda"],
    "cost":          ["cost", "expense", "cogs", "expenditure", "overhead"],
    "discount":      ["discount", "promo", "coupon", "rebate", "offer", "markdown"],
    "price":         ["price", "rate", "unit price", "tariff", "fee", "charge"],
    "quantity":      ["quantity", "qty", "units", "volume", "orders", "count"],
    "performance":   ["revenue", "profit", "quantity", "total", "sales", "kpi"],

    # ── Product / Inventory ────────────────────────────────────────────────────
    "product":       ["product", "item", "sku", "brand", "model", "article"],
    "category":      ["category", "type", "segment", "class", "group", "department"],
    "brand":         ["brand", "manufacturer", "make", "vendor", "supplier"],
    "inventory":     ["stock", "inventory", "quantity", "units", "available"],

    # ── Customer ───────────────────────────────────────────────────────────────
    "customer":      ["customer", "client", "buyer", "member", "account", "user"],
    "segment":       ["segment", "tier", "group", "cohort", "cluster"],
    "age":           ["age", "dob", "birth", "tenure"],
    "gender":        ["gender", "sex"],
    "behavior":      ["frequency", "recency", "monetary", "clv", "ltv", "churn", "retention"],
    "churn":         ["churn", "attrition", "retention", "lapse", "cancel"],
    "loyalty":       ["loyalty", "points", "rewards", "tier", "vip"],
    "satisfaction":  ["satisfaction", "nps", "csat", "rating", "review", "feedback"],

    # ── Location / Geography ───────────────────────────────────────────────────
    "location":      ["location", "region", "city", "state", "area", "zone", "country"],
    "region":        ["region", "zone", "territory", "district", "area"],
    "city":          ["city", "town", "metro", "municipality"],
    "state":         ["state", "province", "county"],
    "store":         ["store", "branch", "outlet", "shop", "site", "facility"],
    "channel":       ["channel", "source", "medium", "platform", "store", "online"],
    "geography":     ["country", "region", "city", "state", "location", "zone"],

    # ── HR / People Analytics ──────────────────────────────────────────────────
    "employee":      ["employee", "staff", "headcount", "workforce", "personnel"],
    "attrition":     ["attrition", "churn", "turnover", "resignation", "exit"],
    "salary":        ["salary", "wage", "compensation", "pay", "remuneration", "ctc"],
    "department":    ["department", "team", "function", "division", "unit", "group"],
    "tenure":        ["tenure", "experience", "seniority", "years", "duration"],
    "hiring":        ["hire", "recruitment", "onboard", "join", "headcount"],
    "leave":         ["leave", "absence", "sick", "pto", "vacation", "holiday"],

    # ── Healthcare ────────────────────────────────────────────────────────────
    "patient":       ["patient", "member", "beneficiary", "enrollee"],
    "diagnosis":     ["diagnosis", "condition", "icd", "disease", "ailment"],
    "readmission":   ["readmission", "readmit", "return visit", "rehospital"],
    "treatment":     ["treatment", "procedure", "therapy", "intervention", "medication"],
    "ward":          ["ward", "department", "unit", "floor", "bed"],
    "outcome":       ["outcome", "recovery", "mortality", "survival", "discharge"],
    "claim":         ["claim", "insurance", "coverage", "reimbursement", "bill"],

    # ── Ecommerce / Web Analytics ─────────────────────────────────────────────
    "traffic":       ["visit", "session", "pageview", "impression", "click"],
    "clicks":        ["click", "ctr", "clickthrough", "interaction"],
    "session":       ["session", "duration", "bounce", "engagement", "visit"],
    "conversion":    ["conversion", "purchase", "transaction", "checkout", "order"],
    "cart":          ["cart", "basket", "wishlist", "abandon"],
    "campaign":      ["campaign", "ad", "creative", "utm", "source", "medium"],
    "funnel":        ["funnel", "stage", "step", "conversion", "drop"],

    # ── Logistics / Supply Chain ───────────────────────────────────────────────
    "delivery":      ["delivery", "shipment", "dispatch", "fulfilment", "shipping"],
    "delay":         ["delay", "late", "sla", "breach", "on time", "lead time"],
    "route":         ["route", "path", "lane", "corridor", "origin", "destination"],
    "carrier":       ["carrier", "courier", "transport", "vehicle", "fleet"],
    "warehouse":     ["warehouse", "facility", "hub", "dc", "depot", "inventory"],
    "supplier":      ["supplier", "vendor", "partner", "source"],

    # ── Finance / Banking ─────────────────────────────────────────────────────
    "loan":          ["loan", "credit", "mortgage", "advance", "facility"],
    "default":       ["default", "npa", "delinquency", "overdue", "bad debt"],
    "risk":          ["risk", "score", "rating", "exposure", "probability"],
    "account":       ["account", "portfolio", "balance", "deposit", "saving"],
    "transaction":   ["transaction", "payment", "transfer", "debit", "credit"],
    "fraud":         ["fraud", "suspicious", "anomaly", "alert", "flag"],
    "interest":      ["interest", "rate", "yield", "return", "spread"],

    # ── Review / Feedback ──────────────────────────────────────────────────────
    "review":        ["review", "rating", "score", "feedback", "satisfaction", "nps"],
    "complaint":     ["complaint", "issue", "ticket", "escalation", "dispute"],
}

# Column keywords that are almost never useful for dashboards
_ALWAYS_DROP_KEYWORDS = ["phone", "mobile", "email", "address", "url", "uuid",
                          "password", "token", "hash", "secret"]


def _parse_intent(requirements: str) -> Tuple[set, set]:
    """
    Returns (force_include_col_keywords, force_exclude_col_keywords).

    force_include: any column whose name contains one of these keywords MUST
                   be kept regardless of its statistical score.
    force_exclude: any column whose name contains one of these keywords MUST
                   be dropped regardless of its statistical score.
    """
    tokens = set(requirements.lower().replace(",", " ").replace(".", " ").split())

    force_include_kws: set = set()
    for token in tokens:
        if token in _INTENT_TO_COLUMNS:
            force_include_kws.update(_INTENT_TO_COLUMNS[token])

    force_exclude_kws = set(_ALWAYS_DROP_KEYWORDS)

    return force_include_kws, force_exclude_kws


def _domain_label(col_lower: str,
                  force_include: set,
                  force_exclude: set) -> Optional[str]:
    """
    Returns 'force_include', 'force_exclude', 'primary_key', or None.

    Primary-key guard:
      Columns like transaction_id, order_id, customer_id, invoice_id are
      row-level identifiers that must ALWAYS be kept in the output dataset —
      they are essential for joins, drillthrough, and BI relationships.
      We detect them by checking whether the column name ends with a
      recognised ID suffix (e.g. _id, _no, _code).
    """
    _ID_SUFFIXES = ('_id', ' id', '_no', '_num', '_number', '_code',
                    '_key', '_ref', '_uuid', '_guid')

    # Check for primary/foreign key columns first — always keep them
    is_id_col = any(col_lower.endswith(sfx) or col_lower == sfx.lstrip('_')
                    for sfx in _ID_SUFFIXES)
    if is_id_col:
        return "primary_key"  # force-keep regardless of statistical score

    if any(kw in col_lower for kw in force_exclude):
        return "force_exclude"

    if any(kw in col_lower for kw in force_include):
        return "force_include"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2a — VARIANCE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def _variance_scores(df_pd: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """
    Normalised variance score in [0, 1] for each numeric column.
    Near-constant columns (var < threshold) get score 0.
    Columns are coerced to float first — the all-string loader means
    df_pd columns are object dtype even when they contain numeric values.
    """
    if not numeric_cols:
        return {}

    sub = df_pd[numeric_cols].copy()

    # Coerce every column to float — required because the dataset is loaded
    # with dtype=str so all columns arrive as object/string in pandas.
    # Without this, sub[c].max() returns the lexicographic string max
    # ("9" > "100") and the subtraction throws TypeError.
    for c in numeric_cols:
        sub[c] = pd.to_numeric(
            sub[c].astype(str).str.replace(r"[₹$£€,\s]", "", regex=True),
            errors="coerce"
        )

    # Normalise each column to [0,1] before computing variance so columns
    # with different units are comparable.
    for c in numeric_cols:
        col_min = sub[c].min()
        col_max = sub[c].max()
        col_range = col_max - col_min
        if pd.notna(col_range) and col_range > 0:
            sub[c] = (sub[c] - col_min) / col_range

    raw_var = sub.var()
    max_var = raw_var.max() if raw_var.max() > 0 else 1.0

    scores: Dict[str, float] = {}
    for c in numeric_cols:
        v = raw_var.get(c, 0.0)
        scores[c] = 0.0 if v < DashboardConfig.VARIANCE_THRESHOLD else round(v / max_var, 3)

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2b — UNIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _univariate_scores(df_pd: pd.DataFrame,
                       all_cols: List[str],
                       target_col: str) -> Dict[str, float]:
    """
    Compute a normalised [0,1] univariate relevance score for every column
    against the primary numeric target (revenue/total/profit).

    Numeric columns  → Pearson |r| + Mutual Information (average of both).
    Categorical cols → Mutual Information classif (target binned into quartiles).
    Date columns     → skipped (always kept via domain rule).
    """
    if not target_col or target_col not in df_pd.columns:
        return {c: 0.5 for c in all_cols}  # neutral if no target

    y = pd.to_numeric(df_pd[target_col].astype(str).str.replace(r"[₹$£€,]", "", regex=True),
                      errors="coerce").fillna(0)

    # Bin target into 4 classes for MI classification
    y_binned = pd.qcut(y, q=4, labels=False, duplicates="drop")

    scores: Dict[str, float] = {}

    for col in all_cols:
        if col == target_col:
            scores[col] = 1.0
            continue

        col_s = df_pd[col].astype(str).str.replace(r"[₹$£€,]", "", regex=True)
        col_num = pd.to_numeric(col_s, errors="coerce")

        try:
            if col_num.notna().mean() > 0.75:
                # ── Numeric column: Pearson + MI ──────────────────────────────
                x = col_num.fillna(col_num.median())
                # Skip constant columns — pearsonr requires non-zero std
                if x.std() < 1e-10:
                    scores[col] = (0.0, "numeric_raw")  # type: ignore[assignment]
                    continue
                r, _ = stats.pearsonr(x, y)
                mi   = mutual_info_regression(x.values.reshape(-1, 1), y.values,
                                              random_state=42)[0]
                scores[col] = (abs(r) + mi, "numeric_raw")  # type: ignore[assignment]
            else:
                # ── Categorical column: MI classif ────────────────────────────
                le   = LabelEncoder()
                x_enc = le.fit_transform(col_s.fillna("__missing__")).reshape(-1, 1)
                mi_c  = mutual_info_classif(x_enc, y_binned.fillna(0).astype(int),
                                            random_state=42)[0]
                scores[col] = (mi_c, "cat_raw")  # type: ignore[assignment]
        except Exception:
            scores[col] = (0.0, "error")  # type: ignore[assignment]

    # Normalise all raw scores to [0, 1]
    numeric_raw  = [v[0] for v in scores.values() if isinstance(v, tuple) and v[1] == "numeric_raw"]
    cat_raw      = [v[0] for v in scores.values() if isinstance(v, tuple) and v[1] == "cat_raw"]
    max_num  = max(numeric_raw)  if numeric_raw  else 1.0
    max_cat  = max(cat_raw)      if cat_raw      else 1.0

    final: Dict[str, float] = {}
    for col, val in scores.items():
        if isinstance(val, float):
            final[col] = val
        elif val[1] == "numeric_raw":
            final[col] = round(min(val[0] / max_num, 1.0), 3)
        elif val[1] == "cat_raw":
            final[col] = round(min(val[0] / max_cat, 1.0), 3)
        else:
            final[col] = 0.0

    return final


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2c — COLLINEARITY REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════

def _redundant_columns(df_pd: pd.DataFrame, numeric_cols: List[str]) -> set:
    """
    Build a correlation matrix and return the set of columns to DROP
    from each highly-correlated pair (|r| > threshold).
    Keep the one with the higher univariate score or the first alphabetically.
    """
    if len(numeric_cols) < 2:
        return set()

    sub = df_pd[numeric_cols].apply(
        lambda c: pd.to_numeric(c.astype(str).str.replace(r"[₹$£€,]", "", regex=True),
                                errors="coerce")
    )
    corr = sub.corr().abs()
    drop = set()

    checked = set()
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            if (c1, c2) in checked or c2 in drop:
                continue
            checked.add((c1, c2))
            if corr.loc[c1, c2] >= DashboardConfig.COLLINEARITY_THRESHOLD:
                # Drop c2 (keep c1 — the one that appeared first / is the "primary")
                drop.add(c2)

    return drop


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDED: RANDOM FOREST FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def _rf_importance(df_pd: pd.DataFrame,
                   all_cols: List[str],
                   target_col: str) -> Dict[str, float]:
    """
    Train a RandomForestRegressor with the primary numeric target and return
    normalised feature importance scores for all input columns.

    Categorical columns are label-encoded.
    Date columns are excluded (they are always kept by domain rule).
    Returns an empty dict if no suitable target or insufficient data.
    """
    if not target_col or target_col not in df_pd.columns or len(df_pd) < 50:
        return {}

    y = pd.to_numeric(df_pd[target_col].astype(str).str.replace(r"[₹$£€,]", "", regex=True),
                      errors="coerce").fillna(0)

    feature_cols = [c for c in all_cols if c != target_col]
    X_parts: List[pd.Series] = []
    used_cols: List[str] = []

    for col in feature_cols:
        col_s = df_pd[col].astype(str).str.replace(r"[₹$£€,]", "", regex=True)
        num   = pd.to_numeric(col_s, errors="coerce")

        if num.notna().mean() > 0.75:
            X_parts.append(num.fillna(num.median()))
            used_cols.append(col)
        else:
            # Encode categorical
            try:
                le  = LabelEncoder()
                enc = pd.Series(le.fit_transform(df_pd[col].fillna("__missing__").astype(str)),
                                name=col)
                X_parts.append(enc)
                used_cols.append(col)
            except Exception:
                pass

    if len(used_cols) < 2:
        return {}

    X = pd.concat(X_parts, axis=1)
    X.columns = used_cols

    # Drop zero-variance columns — they cause RF instability
    X = X.loc[:, X.std() > 1e-10]
    used_cols = list(X.columns)

    if len(used_cols) < 1:
        return {}

    try:
        rf = RandomForestRegressor(
            n_estimators=DashboardConfig.RF_N_ESTIMATORS,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,  # n_jobs=1 avoids joblib/polars isinstance conflict
        )
        # Pass .values (numpy) not DataFrame — prevents sklearn's internal
        # polars isinstance check from crashing when polars is not installed.
        rf.fit(X.values, y.values)

        importances = dict(zip(used_cols, rf.feature_importances_))
        max_imp     = max(importances.values()) if importances else 1.0

        return {
            col: round(imp / max_imp, 3)
            for col, imp in importances.items()
        }
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE FUSION
# ═══════════════════════════════════════════════════════════════════════════════

# Weights for each signal in the fused score
_W_DOMAIN    = 0.20   # rule-based name/type heuristic
_W_VARIANCE  = 0.15   # variance filter
_W_UNIVAR    = 0.30   # univariate (Pearson + MI)
_W_RF        = 0.35   # Random Forest importance


def _fuse(domain: float, variance: float, univar: float, rf: float,
          has_rf: bool) -> float:
    """
    Weighted combination of all signals.
    If RF importances are unavailable (small dataset or no target), redistribute
    the RF weight equally to variance and univariate.
    """
    if has_rf:
        return domain * _W_DOMAIN + variance * _W_VARIANCE + univar * _W_UNIVAR + rf * _W_RF
    else:
        # Redistribute RF weight
        w_var  = _W_VARIANCE + _W_RF * 0.4
        w_uni  = _W_UNIVAR   + _W_RF * 0.6
        return domain * _W_DOMAIN + variance * w_var + univar * w_uni


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — COLUMN RELEVANCE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class ColumnRelevanceAgent:
    """
    Full hybrid feature-selection pipeline:
      1. Domain curation     — user intent → force include/exclude
      2. Variance filter     — remove near-constant features
      3. Univariate analysis — Pearson + Mutual Information
      4. Collinearity check  — drop redundant pairs
      5. RF importance       — model-based ranking
      6. Score fusion        — weighted combination of all signals
      7. LLM batch review    — one Gemini call to validate and annotate
    """

    def __init__(self, api_key: str):
        self.client = _gemini_client(api_key)

    # ── public ────────────────────────────────────────────────────────────────

    def analyze_columns(self, df: pl.DataFrame, requirements: str,
                        pinned_columns: list = None) -> Dict[str, Dict]:
        """
        Returns {col: {column_name, data_type, unique_count, null_pct,
                        relevance_score, role, relevant, reasoning,
                        method_scores: {domain, variance, univariate, rf}}}

        pinned_columns: list of column names the user has explicitly selected
                        to force-keep regardless of any scoring or LLM review.
        """
        _pinned = set(pinned_columns or [])

        print(f"\n[ColumnRelevanceAgent] Running hybrid feature selection "
              f"on {len(df.columns)} columns…")
        if _pinned:
            print(f"  [Pinned by user] {_pinned}")

        # ── STEP 1: Domain curation ──────────────────────────────────────────
        force_include_kws, force_exclude_kws = _parse_intent(requirements)
        print(f"  [Step 1] Force-include keywords: {force_include_kws}")
        print(f"  [Step 1] Force-exclude keywords: {force_exclude_kws}")

        # Basic per-column metadata
        col_meta: Dict[str, Dict] = {}
        for col in df.columns:
            series      = df[col]
            total       = len(df)
            null_pct    = series.null_count() / total * 100
            unique_cnt  = series.n_unique()
            dtype       = _infer_dtype(series)
            col_lower   = col.lower().replace("_", " ").replace("-", " ")
            domain_flag = _domain_label(col_lower, force_include_kws, force_exclude_kws)

            col_meta[col] = {
                "column_name":  col,
                "data_type":    dtype,
                "unique_count": unique_cnt,
                "null_pct":     round(null_pct, 1),
                "col_lower":    col_lower,
                "domain_flag":  domain_flag,   # 'force_include' | 'force_exclude' | None
            }

        # ── Build pandas df for statistical analysis ─────────────────────────
        df_pd = df.to_pandas()

        numeric_cols = [
            c for c, m in col_meta.items()
            if m["data_type"] in ("float", "integer")
        ]
        all_cols = list(col_meta.keys())

        # Primary target: prefer total/revenue/profit column, else first numeric
        target_col = self._find_primary_target(numeric_cols)
        print(f"  [Step 2] Primary target for statistics: '{target_col}'")

        # ── STEP 2a: Variance filter ─────────────────────────────────────────
        var_scores = _variance_scores(df_pd, numeric_cols)
        print(f"  [Step 2a] Variance scores computed for {len(var_scores)} numeric cols")

        # ── STEP 2b: Univariate (Pearson + MI) ──────────────────────────────
        uni_scores = _univariate_scores(df_pd, all_cols, target_col)
        print(f"  [Step 2b] Univariate scores computed for {len(uni_scores)} cols")

        # ── STEP 2c: Collinearity ────────────────────────────────────────────
        redundant = _redundant_columns(df_pd, numeric_cols)
        if redundant:
            print(f"  [Step 2c] Collinear (will be flagged): {redundant}")

        # ── STEP 3: RF importance ────────────────────────────────────────────
        rf_scores = _rf_importance(df_pd, all_cols, target_col)
        has_rf    = bool(rf_scores)
        print(f"  [Step 3] RF importance: {'computed' if has_rf else 'skipped (insufficient data)'}")

        # ── STEP 4: Score fusion ─────────────────────────────────────────────
        results: Dict[str, Dict] = {}
        for col, meta in col_meta.items():
            flag = meta["domain_flag"]

            # Hard overrides from domain curation
            if flag == "force_exclude":
                final_score = 0.0
                role        = "Force-Excluded"
                reason      = "Excluded by domain rule (contact/system column)"
            elif flag == "primary_key":
                final_score = 1.0
                role        = "Primary / Foreign Key"
                reason      = "ID/key column — always retained for BI relationships and joins"
            elif flag == "force_include":
                final_score = 1.0
                role        = "Force-Included"
                reason      = "Required by user requirements"
            elif meta["unique_count"] == 1:
                final_score = 0.0
                role        = "Constant"
                reason      = "Single value — no analytical use"
            elif meta["null_pct"] > 70:
                final_score = 0.05
                role        = "Mostly Null"
                reason      = f"{meta['null_pct']:.0f}% missing — unreliable"
            elif col in redundant:
                final_score = 0.20
                role        = "Redundant (Collinear)"
                reason      = f"|r| ≥ {DashboardConfig.COLLINEARITY_THRESHOLD} with another column"
            else:
                # Domain heuristic score
                domain_score = self._domain_score(meta)
                variance_s   = var_scores.get(col, 0.5)  # 0.5 neutral for non-numerics
                univar_s     = uni_scores.get(col, 0.0)
                rf_s         = rf_scores.get(col, 0.0)

                final_score  = _fuse(domain_score, variance_s, univar_s, rf_s, has_rf)

                # ── DOMAIN FLOOR (Step 1 guarantee) ─────────────────────────
                # If the column is clearly a core business metric (domain ≥ 0.85)
                # OR its name matches user intent, the domain curation step guarantees
                # it reaches at least the relevance threshold.  Statistical methods
                # may show low scores on a specific dataset (e.g. randomly generated
                # test data) but the business meaning is unambiguous.
                if domain_score >= 0.85:
                    final_score = max(final_score, DashboardConfig.RELEVANCE_THRESHOLD)

                role         = self._role_label(meta, domain_score)
                reason       = (
                    f"domain={domain_score:.2f} | var={variance_s:.2f} | "
                    f"univar={univar_s:.2f} | rf={rf_s:.2f}"
                )

            results[col] = {
                "column_name":    col,
                "data_type":      meta["data_type"],
                "unique_count":   int(meta["unique_count"]),
                "null_pct":       float(meta["null_pct"]),
                "relevance_score": float(round(final_score, 2)),
                "role":            role,
                # bool() converts numpy.bool_ → Python bool (json-serializable)
                "relevant":        bool(final_score >= DashboardConfig.RELEVANCE_THRESHOLD),
                "reasoning":       reason,
                "method_scores": {
                    # float() converts numpy.float64 → Python float
                    "domain":     float(round(self._domain_score(meta), 3)),
                    "variance":   float(round(var_scores.get(col, -1), 3)),
                    "univariate": float(round(uni_scores.get(col, -1), 3)),
                    "rf":         float(round(rf_scores.get(col, -1), 3)),
                },
            }

        # ── STEP 4 (LLM): Batch review & plain-English reasoning ─────────────
        results = self._llm_batch_review(results, requirements)

        # ── PINNED COLUMNS: hard-override AFTER LLM review ───────────────────
        # This runs last so no scoring step or LLM call can ever drop a column
        # the user has explicitly chosen to keep.
        for col in _pinned:
            if col in results:
                results[col]["relevant"]        = True
                results[col]["relevance_score"] = 1.0
                results[col]["role"]            = "Pinned by User"
                results[col]["reasoning"]       = "Force-kept by user — overrides all scoring"

        # Print summary
        kept    = sum(1 for r in results.values() if r["relevant"])
        dropped = len(results) - kept
        print(f"  [Done] {kept} kept, {dropped} dropped (threshold={DashboardConfig.RELEVANCE_THRESHOLD})")
        if _pinned:
            print(f"  [Pinned] {_pinned} — forced to kept regardless of score")
        for col, r in results.items():
            sym = "✓" if r["relevant"] else "✗"
            print(f"    {sym} {col:<30} score={r['relevance_score']:.2f}  {r['role']}")

        return results

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _find_primary_target(numeric_cols: List[str]) -> str:
        """Choose the best column to use as the regression target."""
        priority = ["total_price", "total", "revenue", "sales", "amount",
                    "profit", "income", "earnings"]
        for kw in priority:
            for c in numeric_cols:
                if kw in c.lower().replace("_", " "):
                    return c
        return numeric_cols[0] if numeric_cols else ""

    @staticmethod
    def _domain_score(meta: Dict) -> float:
        """
        Pure name/type heuristic score in [0, 1].
        Covers sales, HR, healthcare, ecommerce, logistics, and finance columns.
        Used as one input to the fusion — never the sole decision.
        """
        col_lower = meta["col_lower"]
        dtype     = meta["data_type"]
        unique    = meta["unique_count"]

        # ── ID / key columns — always kept for BI joins and drillthrough ────────
        # transaction_id, customer_id, order_id, invoice_id etc. are essential
        # for relationships in any BI tool (Power BI, Tableau, Looker).
        _ID_SUFFIXES = ('_id', ' id', '_no', '_num', '_number', '_code',
                        '_key', '_ref', '_uuid', '_guid')
        if any(col_lower.endswith(sfx) or col_lower == sfx.lstrip('_')
               for sfx in _ID_SUFFIXES):
            return 1.0   # always keep — primary/foreign key for BI relationships

        # ── Datetime is always high value ─────────────────────────────────────
        if dtype == "datetime":
            return 1.0

        if dtype in ("float", "integer"):
            # ── Financial / monetary KPIs (any domain) ────────────────────────
            if any(kw in col_lower for kw in (
                "revenue","sales","total","amount","profit","margin",
                "cost","price","spend","income","earning","salary","wage",
                "compensation","pay","fee","charge","tax","tariff",
                "loan","balance","claim","reimbursement","interest","yield",
                "budget","forecast","target","quota","penalty","fine",
            )):
                return 1.0

            # ── Volume / count KPIs (any domain) ─────────────────────────────
            if any(kw in col_lower for kw in (
                "quantity","qty","units","count","volume","orders","headcount",
                "clicks","impressions","visits","sessions","pageviews","views",
                "transactions","bookings","enrollments","admissions","patients",
                "shipments","deliveries","tickets","calls","interactions",
            )):
                return 0.90

            # ── Rate / ratio / score KPIs ─────────────────────────────────────
            if any(kw in col_lower for kw in (
                "rate","ratio","score","pct","percent","index",
                "nps","csat","satisfaction","rating","churn","attrition",
                "conversion","bounce","engagement","accuracy","sla",
                "delay","duration","tenure","age","lead time",
            )):
                return 0.85

            # ── Other numeric — moderate value ────────────────────────────────
            return 0.60

        # ── Dimension columns (categorical) ───────────────────────────────────

        # Customer-related
        if any(kw in col_lower for kw in ("customer","client","buyer","member","user","patient","employee","staff")):
            return 0.85

        # Product / inventory
        if any(kw in col_lower for kw in ("product","item","category","brand","sku","model","article","drug","medication")):
            return 0.85 if unique <= DashboardConfig.MAX_CATEGORICAL_UNIQ else 0.45

        # Location / geography
        if any(kw in col_lower for kw in ("location","region","city","state","store","branch","zone","ward","department","route","country")):
            return 0.85

        # Channel / source
        if any(kw in col_lower for kw in ("channel","source","medium","platform","campaign","carrier","vendor","supplier")):
            return 0.80

        # Status / flag / segment (binary or low-cardinality)
        if dtype == "boolean" or (dtype == "categorical" and unique <= 2):
            return 0.75
        if any(kw in col_lower for kw in ("status","flag","type","segment","tier","stage","class","group","label","outcome","diagnosis","condition")):
            return 0.80 if unique <= DashboardConfig.MAX_CATEGORICAL_UNIQ else 0.40

        # General low-cardinality categorical
        if dtype == "categorical":
            if unique <= DashboardConfig.MAX_CATEGORICAL_UNIQ:
                return 0.65
            return 0.20

        return 0.40

    @staticmethod
    def _role_label(meta: Dict, domain_score: float) -> str:
        col_lower = meta["col_lower"]
        dtype     = meta["data_type"]
        unique    = meta["unique_count"]

        if dtype == "datetime":
            return "Time Dimension"

        # ID / key columns — label them properly (kept but not aggregated)
        _ID_SUFFIXES = ('_id', ' id', '_no', '_num', '_number', '_code',
                        '_key', '_ref', '_uuid', '_guid')
        if any(col_lower.endswith(sfx) or col_lower == sfx.lstrip('_')
               for sfx in _ID_SUFFIXES):
            return "Primary / Foreign Key"

        if dtype in ("float", "integer"):
            if any(kw in col_lower for kw in ("revenue","sales","total","amount","profit","margin","cost","price","salary","wage","fee","loan","balance","claim")):
                return "Financial KPI"
            if any(kw in col_lower for kw in ("quantity","qty","units","count","volume","clicks","sessions","visits","headcount","admissions","deliveries")):
                return "Volume Metric"
            if any(kw in col_lower for kw in ("rate","ratio","score","pct","percent","nps","csat","rating","churn","attrition","sla","duration","tenure","age")):
                return "Rate / Score KPI"
            return "Numeric Measure"
        if any(kw in col_lower for kw in ("customer","client","buyer","member","user","patient","employee","staff")):
            return "People Dimension"
        if any(kw in col_lower for kw in ("product","category","brand","item","sku","drug","medication")):
            return "Product / Category Dimension"
        if any(kw in col_lower for kw in ("location","city","state","region","store","branch","ward","zone","route","country","department")):
            return "Location / Org Dimension"
        if any(kw in col_lower for kw in ("channel","source","medium","platform","campaign","carrier","vendor")):
            return "Channel Dimension"
        if any(kw in col_lower for kw in ("status","flag","type","stage","outcome","diagnosis","condition","segment","tier")):
            return "Status / Segment Flag"
        if dtype == "boolean" or unique <= 2:
            return "Boolean Flag"
        if dtype == "categorical":
            return "Dimension / Slicer" if unique <= DashboardConfig.MAX_CATEGORICAL_UNIQ else "High-Cardinality"
        return "Other"

    def _llm_batch_review(self, results: Dict[str, Dict], requirements: str) -> Dict[str, Dict]:
        """
        One Gemini call: reviews the full batch.
        Domain-agnostic prompt — works for any industry dataset.
        Only columns the LLM disagrees with are returned.
        """
        batch = "\n".join(
            f"  {col}: score={r['relevance_score']}, role={r['role']}, "
            f"dtype={r['data_type']}, unique={r['unique_count']}, "
            f"method_scores={r['method_scores']}"
            for col, r in results.items()
        )

        prompt = f"""You are a senior data analyst reviewing automated feature-selection scores for a BI dashboard.
The dataset may be from ANY industry (sales, HR, healthcare, ecommerce, logistics, finance, etc.).

User requirements: "{requirements}"

Hybrid scores per column (domain heuristic + variance filter + Pearson/MI + RandomForest importances):
{batch}

Columns with score >= {DashboardConfig.RELEVANCE_THRESHOLD} will be KEPT in the dashboard dataset.

Review these scores. Return a JSON array ONLY for columns where the score is CLEARLY WRONG given what the user wants to analyse.
Ask yourself: if this column were dropped, would the user's stated analysis become impossible or misleading?
Do NOT adjust columns that are already correctly scored.

Return ONLY valid JSON (no markdown fences):
[
  {{"column": "<n>", "new_score": <0.0-1.0>, "role": "<short role>", "reasoning": "<one sentence>"}}
]

If all scores look correct for the user's requirements, return: []
"""
        raw = _call_gemini(self.client, prompt)

        try:
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            for adj in json.loads(raw):
                col = adj.get("column")
                if col and col in results:
                    ns = float(adj.get("new_score", results[col]["relevance_score"]))
                    results[col]["relevance_score"] = float(round(ns, 2))
                    results[col]["role"]      = adj.get("role", results[col]["role"])
                    results[col]["reasoning"] = adj.get("reasoning", "") + " [LLM]"
                    results[col]["relevant"]  = bool(ns >= DashboardConfig.RELEVANCE_THRESHOLD)
        except Exception:
            pass  # Keep statistical scores if LLM response is malformed

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — MEASURE CREATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class MeasureCreationAgent:
    """
    Creates computable KPI measures from the kept columns.
    Measures are stored as scalar columns in the output CSV (same value per row)
    so any BI tool can aggregate / pivot them directly.
    """

    def __init__(self, api_key: str):
        self.client = _gemini_client(api_key)

    def identify_measures(self, df: pl.DataFrame, kept_cols: List[str],
                          requirements: str) -> List[Dict]:
        df_pd   = df.select(kept_cols).to_pandas()
        num_c   = self._numeric_cols(df_pd)
        bool_c  = self._bool_cols(df_pd)

        total_c    = self._find(num_c, ["total_price","total","revenue","amount","sales"])
        profit_c   = self._find(num_c, ["profit"])
        cost_c     = self._find(num_c, ["cost"])
        price_c    = self._find(num_c, ["unit_price","price","rate"])
        qty_c      = self._find(num_c, ["quantity","qty","units"])
        customer_c = self._find(kept_cols, ["customer_id","customer"])
        review_c   = self._find(num_c, ["review","rating","score"])

        m: List[Dict] = []

        if total_c:
            m += [self._def("Total_Revenue",    f"SUM({total_c})",    [total_c],  "sum",    "Total sales revenue"),
                  self._def("Avg_Order_Value",   f"AVG({total_c})",    [total_c],  "mean",   "Average revenue per transaction")]
        if profit_c:
            m.append(self._def("Total_Profit",  f"SUM({profit_c})",   [profit_c], "sum",    "Total profit"))
        if total_c and profit_c:
            m.append(self._def("Profit_Margin_%",
                               f"SUM({profit_c})/SUM({total_c})*100",
                               [profit_c, total_c], "profit_margin",  "Profit margin %"))
        if cost_c and total_c:
            m.append(self._def("Cost_to_Revenue",
                               f"SUM({cost_c})/SUM({total_c})",
                               [cost_c, total_c], "ratio",            "Cost-to-revenue ratio"))
        if qty_c:
            m += [self._def("Total_Units_Sold",  f"SUM({qty_c})",      [qty_c],    "sum",    "Total units sold"),
                  self._def("Avg_Units_Per_Order",f"AVG({qty_c})",     [qty_c],    "mean",   "Avg units per order")]
        if price_c:
            m.append(self._def("Avg_Selling_Price", f"AVG({price_c})", [price_c],  "mean",   "Average unit selling price"))
        if customer_c:
            m.append(self._def("Unique_Customers",
                               f"COUNTDISTINCT({customer_c})",
                               [customer_c], "nunique",               "Number of unique customers"))
        if total_c and customer_c:
            m.append(self._def("Revenue_Per_Customer",
                               f"SUM({total_c})/COUNTDISTINCT({customer_c})",
                               [total_c, customer_c], "rev_per_cust", "Average revenue per customer"))
        if review_c:
            m.append(self._def("Avg_Review_Score", f"AVG({review_c})", [review_c], "mean",   "Average customer satisfaction score"))
        for bc in bool_c:
            if "discount" in bc.lower():
                m.append(self._def("Discount_Rate_%", f"AVG({bc})*100", [bc], "discount_rate",
                                   "% of transactions with discount applied"))

        return m

    def create_measures(self, df_pd: pd.DataFrame, measures: List[Dict]) -> pd.DataFrame:
        for m in measures:
            op, deps, name = m["op"], m["column_deps"], m["name"]
            try:
                for c in deps:
                    if c in df_pd.columns:
                        df_pd[c] = pd.to_numeric(
                            df_pd[c].astype(str).str.replace(r"[₹$£€,]","",regex=True),
                            errors="coerce")
                c0 = deps[0] if deps else None

                if   op == "sum"          and c0:              df_pd[name] = df_pd[c0].sum()
                elif op == "mean"         and c0:              df_pd[name] = round(df_pd[c0].mean(), 2)
                elif op == "nunique"      and c0:              df_pd[name] = df_pd[c0].nunique()
                elif op == "profit_margin"and len(deps) >= 2:
                    t = df_pd[deps[1]].sum()
                    df_pd[name] = round(df_pd[deps[0]].sum() / t * 100, 2) if t else 0
                elif op == "ratio"        and len(deps) >= 2:
                    t = df_pd[deps[1]].sum()
                    df_pd[name] = round(df_pd[deps[0]].sum() / t, 4) if t else 0
                elif op == "rev_per_cust" and len(deps) >= 2:
                    n = df_pd[deps[1]].nunique()
                    df_pd[name] = round(df_pd[deps[0]].sum() / n, 2) if n else 0
                elif op == "discount_rate"and c0:
                    df_pd[name] = round(pd.to_numeric(df_pd[c0], errors="coerce").fillna(0).mean() * 100, 2)
            except Exception as e:
                print(f"  [Measure] Skipped '{name}': {e}")
        return df_pd

    # helpers
    @staticmethod
    def _def(name, formula, deps, op, bv) -> Dict:
        return {"name": name, "formula_desc": formula,
                "column_deps": deps, "op": op, "business_value": bv}

    @staticmethod
    def _numeric_cols(df_pd: pd.DataFrame) -> List[str]:
        out = []
        for c in df_pd.columns:
            p = pd.to_numeric(df_pd[c].astype(str).str.replace(r"[₹$£€,]","",regex=True), errors="coerce")
            if p.notna().mean() > 0.80:
                out.append(c)
        return out

    @staticmethod
    def _bool_cols(df_pd: pd.DataFrame) -> List[str]:
        return [c for c in df_pd.columns
                if set(df_pd[c].dropna().astype(str).str.lower().unique()) <= {"true","false","1","0","yes","no"}]

    @staticmethod
    def _find(cols: List[str], keywords: List[str]) -> str:
        for kw in keywords:
            for c in cols:
                if kw in c.lower().replace("_"," ").replace("-"," "):
                    return c
        return ""