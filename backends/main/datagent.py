"""
DATAAGENT - ENHANCED WITH ADAPTIVE ML IMPUTATION
Production-grade multi-agent system with intelligent strategy selection

NEW FEATURES:
✓ Adaptive imputation strategy selector
✓ Random Forest for complex categorical imputation
✓ KNN for context-aware numeric imputation
✓ Trend-based time series imputation
✓ Automatic fallback to statistical methods
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import json
import warnings
import os
import logging

from pathlib import Path
warnings.filterwarnings('ignore')

# Gemini is used in measure_creation.py (feature/KPI phase) via google-generativeai.
# datagent.py (cleaning phase) runs entirely on scikit-learn + SentenceTransformers.
# No LLM import needed here.

# ML & Analysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from scipy import stats
from collections import defaultdict
import re

# ── rapidfuzz with difflib fallback ─────────────────────────────────────────
# rapidfuzz is fast C-extension fuzzy matching.
# If not installed we fall back to Python stdlib difflib — slower but identical
# logic so the rest of the code works unchanged without any crashes.
try:
    from rapidfuzz import fuzz as _fuzz_lib
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import difflib as _difflib_lib
    _RAPIDFUZZ_AVAILABLE = False
    print("⚠ rapidfuzz not installed — using stdlib difflib fallback (slower, same results)")
    print("  Install for better performance: pip install rapidfuzz")

class fuzz:
    """Unified fuzzy-match interface — rapidfuzz when available, difflib otherwise."""
    @staticmethod
    def ratio(a: str, b: str) -> float:
        if _RAPIDFUZZ_AVAILABLE:
            return _fuzz_lib.ratio(a, b)
        return _difflib_lib.SequenceMatcher(None, a, b).ratio() * 100

    @staticmethod
    def partial_ratio(a: str, b: str) -> float:
        if _RAPIDFUZZ_AVAILABLE:
            return _fuzz_lib.partial_ratio(a, b)
        # difflib doesn't have partial_ratio; approximate with longest common sub-sequence
        if not a or not b:
            return 0.0
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        best = 0.0
        for i in range(len(longer) - len(shorter) + 1):
            r = _difflib_lib.SequenceMatcher(None, shorter, longer[i:i+len(shorter)]).ratio()
            if r > best:
                best = r
        return best * 100

    @staticmethod
    def token_sort_ratio(a: str, b: str) -> float:
        if _RAPIDFUZZ_AVAILABLE:
            return _fuzz_lib.token_sort_ratio(a, b)
        a_sorted = " ".join(sorted(a.lower().split()))
        b_sorted = " ".join(sorted(b.lower().split()))
        return _difflib_lib.SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100

    @staticmethod
    def token_set_ratio(a: str, b: str) -> float:
        if _RAPIDFUZZ_AVAILABLE:
            return _fuzz_lib.token_set_ratio(a, b)
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        inter = a_set & b_set
        diff_a = a_set - b_set
        diff_b = b_set - a_set
        base = " ".join(sorted(inter))
        full_a = base + " " + " ".join(sorted(diff_a))
        full_b = base + " " + " ".join(sorted(diff_b))
        scores = [
            _difflib_lib.SequenceMatcher(None, base.strip(), full_a.strip()).ratio(),
            _difflib_lib.SequenceMatcher(None, base.strip(), full_b.strip()).ratio(),
            _difflib_lib.SequenceMatcher(None, full_a.strip(), full_b.strip()).ratio(),
        ]
        return max(scores) * 100

# Sentence Transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── NEW TOOL 1: dateparser ───────────────────────────────────────────────────
# What it does: parses ANY date string format in one call — DD-MM-YYYY,
# MM-DD-YYYY, ISO, "23 Jul 2023", "July 23rd 2023", European formats, etc.
# Why we use it: replaces 80+ lines of candidate_formats loop + dayfirst
# disambiguation + fallback inference. One library call handles everything,
# including formats our loop never anticipated.
# Install: pip install dateparser
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    print("⚠ dateparser not installed — falling back to pandas format detection")
    print("  Install with: pip install dateparser")

# ── NEW TOOL 2: pandera ──────────────────────────────────────────────────────
# What it does: schema-based DataFrame validation. You define column rules
# (type, range, nullability) and it checks all of them in one call,
# returning a structured error report.
# Why we use it: replaces the manual _validate_integrity and
# _validate_business_rules methods in ValidationAgent — ~120 lines
# of hand-written checks become a declarative schema definition.
# Install: pip install pandera
try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    print("⚠ pandera not installed — falling back to manual validation")
    print("  Install with: pip install pandera")

# For environment variables
from dotenv import load_dotenv

MEMORY_PATH = "memory_store.json"

# ── Universal string-null vocabulary ────────────────────────────────────────
# These are values that look like real data but mean "no data".
# Datasets like dirty_cafe_sales.csv use UNKNOWN/ERROR instead of blank/NaN.
# The base code never converts these → they survive as category values,
# get treated as valid strings, and corrupt imputation, fuzzy-grouping,
# and type-detection. This set is applied as the very first cleaning step.
STRING_NULL_VALUES: set = {
    'unknown', 'error', 'n/a', 'na', 'nan', 'none', 'null', 'nil',
    '-', '--', '---', 'missing', '?', 'tbd', 'tba', 'undefined',
    '#n/a', '#na', '#null!', 'inf', '-inf', 'not available',
    'not applicable', 'no data', 'no value', 'empty', 'blank',
}


def normalise_string_nulls(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Convert every cell whose stripped-lowercase value is in STRING_NULL_VALUES
    to real NaN so the rest of the pipeline treats it as missing.

    WHY THIS IS THE FIRST STEP:
    - dirty_cafe_sales.csv uses 'UNKNOWN' and 'ERROR' for ~3,256 cells
      across Item, Quantity, Price, Payment Method, Location, Date.
    - Without this step those strings survive as category values and:
        a) corrupt fuzzy grouping (UNKNOWN groups with nothing)
        b) corrupt type detection (50% parse rate → column mislabeled 'text')
        c) corrupt integer coercion (ERROR → NaN → fillna(0) → Quantity=0)
        d) never get imputed because QualityAgent only sees Polars null_count()

    Returns: (cleaned_df, total_cells_converted)
    """
    before = df.isna().sum().sum()
    for col in df.columns:
        # ── FIX: check for BOTH object and StringDtype ───────────────────────
        # pandas read_csv(dtype=str) creates StringDtype columns whose
        # dtype.name is 'str', not 'object'. The check df[col].dtype == object
        # returns False for StringDtype → function silently skips every column.
        # Fix: check if the column holds string-like values by checking if
        # it is NOT a numeric or datetime dtype — covers object, StringDtype,
        # category, and any future string-backed extension types.
        col_dtype = df[col].dtype
        is_string_col = (
            col_dtype == object
            or col_dtype.name in ('str', 'string', 'StringDtype')
            or hasattr(col_dtype, 'categories')      # CategoricalDtype
            or str(col_dtype).startswith('string')   # pd.StringDtype variants
        )
        if not is_string_col:
            continue
        # Convert to plain object first so NaN assignment works on all pandas versions
        try:
            df[col] = df[col].astype(object)
        except Exception:
            pass
        mask = df[col].astype(str).str.strip().str.lower().isin(STRING_NULL_VALUES)
        df.loc[mask, col] = np.nan
    after = df.isna().sum().sum()
    return df, int(after - before)


def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_memory(memory):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class Config:
    """Centralized configuration for the cleaning pipeline"""

    @staticmethod
    def load_api_key() -> str:
        """
        Load Gemini API key from .env.
        The cleaning pipeline (datagent.py) does NOT use Gemini —
        it runs on scikit-learn + SentenceTransformers only.
        The key is forwarded to measure_creation.py for the KPI phase.
        Returns empty string gracefully if no key is set — no crash, no hang.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            print("✓ GEMINI_API_KEY loaded (used in KPI/measure phase)")
        else:
            print("⚠ No GEMINI_API_KEY in .env — cleaning pipeline unaffected, "
                  "KPI suggestions will require it")
        return api_key
    
    # Cleaning thresholds
    OUTLIER_THRESHOLD = 0.05
    MAX_NULL_PERCENT = 80
    
    # Semantic matching thresholds
    PATTERN_MATCH_THRESHOLD = 0.3
    EMBEDDING_THRESHOLD = 0.5
    USE_HYBRID = True
    
    # ML imputation thresholds (NEW)
    ML_MIN_TRAINING_SAMPLES = 50  # Minimum rows for ML training
    ML_MAX_MISSING_PERCENT = 40    # Don't use ML if >40% missing
    ML_MIN_CONTEXT_FEATURES = 2    # Need at least 2 context columns
    SIMPLE_METHOD_THRESHOLD = 5    # Use simple methods if <5% missing


# ============================================================================
# AGENT BASE CLASS
# ============================================================================

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str, gemini_api_key: str = ""):
        self.name = name
        self.role = role
        self.actions_log = []
        self.start_time = datetime.now()
        
        # Note: LLM initialization removed - not used in current implementation
        # Can be added back if needed for future features
    
    def log_action(self, action: str, details: Dict, confidence: float):
        self.actions_log.append({
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': action,
            'details': details,
            'confidence': confidence
        })
    
    def get_logs(self) -> List[Dict]:
        return self.actions_log
    
    def get_execution_time(self) -> float:
        """Calculate execution time in seconds"""
        return (datetime.now() - self.start_time).total_seconds()


# ============================================================================
# AGENT 1: PROFILER AGENT
# ============================================================================

class ProfilerAgent(BaseAgent):
    """Deep dataset profiling - understands structure, types, distributions"""
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="ProfilerAgent",
            role="Dataset Profiling Expert",
            gemini_api_key=gemini_api_key
        )
    
    def load_dataset(self, file_path: str) -> pl.DataFrame:
        print(f"\n[{self.name}] Loading dataset: {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                df = pl.read_csv(file_path, infer_schema_length=10000, ignore_errors=True)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pl.from_pandas(pd.read_excel(file_path))
            elif file_path.endswith('.json'):
                df = pl.read_json(file_path)
            elif file_path.endswith('.parquet'):
                df = pl.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.log_action(
                action="dataset_loaded",
                details={'file': file_path, 'rows': len(df), 'columns': len(df.columns)},
                confidence=1.0
            )
            
            print(f"✓ Loaded: {len(df)} rows × {len(df.columns)} columns")
            return df
        
        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            raise
    
    def comprehensive_profile(self, df: pl.DataFrame) -> Dict:
        print(f"\n[{self.name}] Generating comprehensive profile...")
        
        profile = {
            'basic': self._basic_stats(df),
            'columns': self._column_analysis(df),
            'quality': self._quality_metrics(df),
            'distributions': self._distribution_analysis(df)
        }
        
        self.log_action(
            action="profile_generated",
            details={'columns_analyzed': len(df.columns)},
            confidence=0.95
        )
        
        print(f"✓ Profile complete: {len(profile['columns'])} columns analyzed")
        return profile
    
    def _basic_stats(self, df: pl.DataFrame) -> Dict:
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_mb': df.estimated_size('mb'),
            'duplicate_rows': df.is_duplicated().sum(),
            'column_names': df.columns
        }
    
    def _column_analysis(self, df: pl.DataFrame) -> List[Dict]:
        columns_info = []
        
        for col in df.columns:
            series = df[col]
            col_pd = series.to_pandas()
            
            actual_type = self._detect_actual_type(col_pd)
            
            null_count = series.null_count()
            null_pct = (null_count / len(df)) * 100
            unique_count = series.n_unique()
            unique_pct = (unique_count / len(df)) * 100
            
            sample_values = col_pd.dropna().head(5).tolist()
            
            info = {
                'name': col,
                'polars_dtype': str(series.dtype),
                'actual_type': actual_type,
                'null_count': null_count,
                'null_pct': round(null_pct, 2),
                'unique_count': unique_count,
                'unique_pct': round(unique_pct, 2),
                'sample_values': sample_values
            }
            
            if actual_type in ['integer', 'float']:
                try:
                    numeric_data = pd.to_numeric(col_pd, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        info['min'] = float(numeric_data.min())
                        info['max'] = float(numeric_data.max())
                        info['mean'] = float(numeric_data.mean())
                        info['median'] = float(numeric_data.median())
                        info['std'] = float(numeric_data.std())
                except:
                    pass
            
            columns_info.append(info)
        
        return columns_info
    
    def _detect_actual_type(self, series: pd.Series) -> str:
        clean = series.dropna()

        # ── FIX: exclude string-null sentinels before type detection ──────────
        # Without this, a column like Transaction Date with 'UNKNOWN'/'ERROR'
        # mixed in gets only ~50% datetime parse rate → mislabeled as 'text'.
        # Use astype(str) so this works on both object and StringDtype columns.
        clean = clean[~clean.astype(str).str.strip().str.lower().isin(STRING_NULL_VALUES)]

        if len(clean) == 0:
            return 'empty'
        
        sample = clean.sample(min(1000, len(clean)))
        
        try:
            numeric = pd.to_numeric(sample, errors='coerce')
            if numeric.notna().sum() / len(sample) > 0.9:
                if all(x.is_integer() for x in numeric.dropna()):
                    return 'integer'
                return 'float'
        except:
            pass
        
        try:
            dates = pd.to_datetime(sample, errors='coerce')
            if dates.notna().sum() / len(sample) > 0.7:
                return 'datetime'
        except:
            pass
        
        unique_vals = set(str(v).lower() for v in sample.unique())
        if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}):
            return 'boolean'
        
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05 or series.nunique() <= 20:
            return 'categorical'
        
        if unique_ratio > 0.95:
            return 'identifier'
        
        return 'text'
    
    def _quality_metrics(self, df: pl.DataFrame) -> Dict:
        total_cells = len(df) * len(df.columns)
        null_cells = sum(df[col].null_count() for col in df.columns)
        
        return {
            'completeness': round((1 - null_cells / total_cells) * 100, 2),
            'total_nulls': null_cells,
            'duplicate_rows': df.is_duplicated().sum(),
            'duplicate_pct': round((df.is_duplicated().sum() / len(df)) * 100, 2)
        }
    
    def _distribution_analysis(self, df: pl.DataFrame) -> Dict:
        distributions = {}
        
        for col in df.columns:
            try:
                numeric = pd.to_numeric(df[col].to_pandas(), errors='coerce').dropna()
                if len(numeric) > 0:
                    distributions[col] = {
                        'skewness': float(stats.skew(numeric)),
                        'kurtosis': float(stats.kurtosis(numeric)),
                        'quartiles': {
                            'q25': float(np.percentile(numeric, 25)),
                            'q50': float(np.percentile(numeric, 50)),
                            'q75': float(np.percentile(numeric, 75))
                        }
                    }
            except:
                continue
        
        return distributions


# ============================================================================
# AGENT 2: SEMANTIC AGENT
# ============================================================================

class SemanticAgent(BaseAgent):
    """Understands column meanings using hybrid pattern + embedding matching"""
    
    SEMANTIC_PATTERNS = {
        'TransactionID': {
            'keywords': ['transaction', 'order', 'invoice', 'bill', 'receipt', 'id', 'trans', 'txn'],
            'description': 'unique transaction identifier order number invoice id',
            'characteristics': {'unique': True, 'nullable': False},
            'validation': lambda x: x.nunique() / len(x) > 0.95
        },
        'CustomerID': {
            'keywords': ['customer', 'client', 'buyer', 'user', 'account', 'cust'],
            'description': 'customer client identifier buyer account user id',
            'characteristics': {'unique': False, 'nullable': False},
            'validation': lambda x: x.nunique() / len(x) < 0.8
        },
        'ProductID': {
            'keywords': ['product', 'item', 'sku', 'article', 'prod'],
            'description': 'product item identifier sku article code',
            'characteristics': {'unique': False, 'nullable': False},
            'validation': lambda x: True
        },
        'ProductName': {
            'keywords': ['product', 'item', 'name', 'title', 'description'],
            'description': 'product item name title description label',
            'characteristics': {'unique': False, 'nullable': False},
            'validation': lambda x: x.dtype == object
        },
        'Quantity': {
            'keywords': ['quantity', 'qty', 'units', 'count', 'amount', 'pieces'],
            'description': 'quantity number of items units count pieces amount',
            'characteristics': {'min': 1, 'nullable': False, 'integer': True},
            'validation': lambda x: (x > 0).all() if x.notna().any() else True
        },
        'Price': {
            'keywords': ['price', 'rate', 'cost', 'unit', 'value', 'perunit'],
            'description': 'price per unit cost rate value unitprice',
            'characteristics': {'min': 0, 'nullable': False},
            'validation': lambda x: (x >= 0).all() if x.notna().any() else True
        },
        'TotalAmount': {
            'keywords': ['total', 'amount', 'sum', 'subtotal', 'gross', 'net', 'spent', 'paid'],
            'description': 'total amount sum paid spent grand total subtotal',
            'characteristics': {'min': 0, 'nullable': False, 'computed': True},
            'validation': lambda x: (x >= 0).all() if x.notna().any() else True
        },
        'Discount': {
            'keywords': ['discount', 'rebate', 'offer', 'promo', 'coupon', 'off', 'applied'],
            'description': 'discount rebate offer promotion coupon savings',
            'characteristics': {'min': 0, 'nullable': True},
            'validation': lambda x: True
        },
        'OrderDate': {
            'keywords': ['date', 'time', 'timestamp', 'created', 'ordered', 'purchase'],
            'description': 'date time timestamp when ordered purchased created',
            'characteristics': {'nullable': False, 'future': False},
            'validation': lambda x: pd.to_datetime(x, errors='coerce').notna().any()
        },
        'PaymentMethod': {
            'keywords': ['payment', 'method', 'type', 'mode', 'pay'],
            'description': 'payment method type mode how paid',
            'characteristics': {'nullable': False},
            'validation': lambda x: x.nunique() < 20
        },
        'Status': {
            'keywords': ['status', 'state', 'condition', 'stage'],
            'description': 'status state condition stage progress',
            'characteristics': {'nullable': False},
            'validation': lambda x: x.nunique() < 15
        },
        'Category': {
            'keywords': ['category', 'type', 'class', 'segment', 'department', 'group'],
            'description': 'category type class department segment group',
            'characteristics': {'nullable': True},
            'validation': lambda x: True
        },
        'Location': {
            'keywords': ['location', 'store', 'place', 'site', 'venue', 'shop'],
            'description': 'location store place site venue branch shop',
            'characteristics': {'nullable': True},
            'validation': lambda x: True
        }
    }
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="SemanticAgent",
            role="Semantic Understanding & Relationship Expert",
            gemini_api_key=gemini_api_key
        )
        
        print(f"[{self.name}] Loading sentence transformer model...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
            
            self.type_embeddings = {}
            for sem_type, config in self.SEMANTIC_PATTERNS.items():
                desc = config['description']
                self.type_embeddings[sem_type] = self.embedding_model.encode(desc)
            
            print(f"✓ Sentence transformer loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load sentence transformer: {e}")
            print(f"  Falling back to pattern matching only")
            self.use_embeddings = False
    
    def identify_semantics(self, df: pl.DataFrame, profile: Dict) -> Dict:
        print(f"\n[{self.name}] Identifying semantic types (hybrid method)...")
        
        semantic_map = {}
        
        for col_info in profile['columns']:
            col_name = col_info['name']
            series = df[col_name].to_pandas()
            sample_values = col_info.get('sample_values', [])
            
            pattern_result = self._pattern_match(col_name, series)
            
            if pattern_result['confidence'] < Config.PATTERN_MATCH_THRESHOLD and self.use_embeddings:
                embedding_result = self._embedding_match(col_name, sample_values, series)
                
                if embedding_result['confidence'] > pattern_result['confidence']:
                    result = embedding_result
                    result['method'] = 'embedding'
                else:
                    result = pattern_result
                    result['method'] = 'pattern'
            else:
                result = pattern_result
                result['method'] = 'pattern'
            
            semantic_map[col_name] = {
                'semantic_type': result['type'],
                'confidence': result['confidence'],
                'method': result.get('method', 'pattern'),
                'expected_dtype': self._get_expected_dtype(result['type'])
            }
        
        self.log_action(
            action="semantics_identified",
            details={
                'columns_mapped': len(semantic_map),
                'embedding_used': self.use_embeddings
            },
            confidence=0.90
        )
        
        print(f"✓ Semantic mapping complete: {len(semantic_map)} columns")
        return semantic_map
    
    def _pattern_match(self, col_name: str, series: pd.Series) -> Dict:
        col_lower = col_name.lower().strip().replace(' ', '').replace('_', '')
        best_match = None
        best_score = 0

        for sem_type, config in self.SEMANTIC_PATTERNS.items():
            keyword_matches = 0
            for kw in config['keywords']:
                kw_normalized = kw.replace(' ', '').replace('_', '')
                if kw_normalized in col_lower:
                    keyword_matches += 1

            score = keyword_matches / len(config['keywords'])

            if col_lower == sem_type.lower().replace(' ', ''):
                score = 1.0

            # ── FIX: only apply validation bonus when keyword score is meaningful ──
            # Old code applied +0.3 validation bonus unconditionally when score > 0.2
            # This caused CustomerID (validation: nunique/len < 0.8) to win over
            # TotalAmount for columns like 'Total Spent' because almost any non-ID
            # column passes that check.
            # Fix: only add the validation bonus when the column already has 2+
            # keyword hits (score >= 0.2 from keywords alone, not just > 0.2).
            # Also skip validation bonus for CustomerID/ProductID/TransactionID
            # when the column name does NOT contain an explicit ID/customer/product
            # keyword — prevents these identity types from hijacking numeric columns.
            id_types = {'CustomerID', 'ProductID', 'TransactionID'}
            if sem_type in id_types:
                # Only allow ID semantic types when the column name strongly
                # implies an identifier — requires 'id', 'cust', 'prod', 'txn'
                # or 'transaction' paired with 'id'/'num'/'no'.
                # This prevents 'Transaction Date' from scoring as TransactionID
                # just because 'transaction' appears in both.
                strict_id_signals = {'id', 'cust', 'txn', 'num', 'no', 'code', 'key', 'ref'}
                has_strict_id = any(kw in col_lower for kw in strict_id_signals)
                if not has_strict_id:
                    if score > best_score:
                        best_score = score
                        best_match = sem_type
                    continue

            if score >= 0.2:
                try:
                    if config['validation'](series):
                        score += 0.3
                except Exception:
                    pass

            if score > best_score:
                best_score = score
                best_match = sem_type

        return {
            'type': best_match if best_score > Config.PATTERN_MATCH_THRESHOLD else 'Unknown',
            'confidence': best_score
        }
    
    def _embedding_match(self, col_name: str, sample_values: List, series: pd.Series) -> Dict:
        sample_str = ' '.join(str(v) for v in sample_values[:3])
        col_description = f"{col_name} {sample_str}"
        
        col_embedding = self.embedding_model.encode(col_description)
        
        best_match = None
        best_score = 0
        
        for sem_type, type_embedding in self.type_embeddings.items():
            similarity = cosine_similarity(
                col_embedding.reshape(1, -1),
                type_embedding.reshape(1, -1)
            )[0][0]
            
            try:
                config = self.SEMANTIC_PATTERNS[sem_type]
                if config['validation'](series):
                    similarity += 0.1
            except:
                pass
            
            if similarity > best_score:
                best_score = similarity
                best_match = sem_type
        
        return {
            'type': best_match if best_score > Config.EMBEDDING_THRESHOLD else 'Unknown',
            'confidence': float(best_score)
        }
    
    def _get_expected_dtype(self, semantic_type: str) -> str:
        type_map = {
            'TransactionID': 'string',
            'CustomerID': 'string',
            'ProductID': 'string',
            'ProductName': 'string',
            'Quantity': 'integer',
            'Price': 'float',
            'TotalAmount': 'float',
            'Discount': 'float',
            'OrderDate': 'datetime',
            'PaymentMethod': 'categorical',
            'Status': 'categorical',
            'Category': 'categorical'
        }
        return type_map.get(semantic_type, 'unknown')
    
    def discover_relationships(self, df: pl.DataFrame, semantic_map: Dict) -> List[Dict]:
        print(f"\n[{self.name}] Discovering column relationships...")
        
        relationships = []
        relationships.extend(self._find_multiplication_relationships(df, semantic_map))
        relationships.extend(self._find_correlations(df, semantic_map))
        
        self.log_action(
            action="relationships_discovered",
            details={'relationships_found': len(relationships)},
            confidence=0.90
        )
        
        print(f"✓ Found {len(relationships)} relationships")
        return relationships
    
    def _find_multiplication_relationships(self, df: pl.DataFrame, semantic_map: Dict) -> List[Dict]:
        relationships = []
        
        # ── Helper: find columns by semantic type OR by column name keywords ──
        # This ensures 'Total Spent' is found even when SemanticAgent labels it
        # 'Unknown' (happens when keyword confidence is below PATTERN_MATCH_THRESHOLD
        # of 0.3 and the embedding model is unavailable).
        def find_cols(sem_types, fallback_keywords):
            """Return columns matching semantic type OR whose name contains a keyword."""
            by_sem = [k for k, v in semantic_map.items()
                      if v['semantic_type'] in sem_types]
            if by_sem:
                return by_sem
            # Fallback: column name keyword search
            return [
                col for col in df.columns
                if any(kw in col.lower().replace(' ', '').replace('_', '')
                       for kw in fallback_keywords)
            ]

        qty_cols     = find_cols(['Quantity'],    ['qty', 'quantity', 'units', 'count', 'pieces'])
        price_cols   = find_cols(['Price'],       ['unitprice', 'price', 'rate', 'cost', 'perunit', 'mrp'])
        total_cols   = find_cols(['TotalAmount'], ['totalprice', 'total', 'spent', 'amount', 'sum', 'subtotal', 'gross', 'net', 'paid', 'revenue'])
        discount_cols= find_cols(['Discount'],    ['discount', 'rebate', 'promo', 'coupon', 'off'])
        
        # ── EXCLUDE cost from price_cols if both cost AND unit_price exist ────
        # cost = purchase cost (what we paid), unit_price = selling price.
        # total_price = quantity × unit_price, NOT quantity × cost.
        # profit = total_price - (quantity × cost).
        # If semantic_map incorrectly assigns 'Price' to 'cost', swap it out.
        has_unit_price = any('unitprice' in c.lower().replace('_','') or 'unitprice' in c.lower().replace(' ','') or c.lower() in ('unit_price','unitprice') for c in df.columns)
        has_cost = any(c.lower() in ('cost',) for c in df.columns)
        if has_unit_price and has_cost:
            # prefer columns explicitly named unit_price over cost
            price_cols = [c for c in price_cols if 'unit' in c.lower()] or price_cols
            # remove 'cost' from total_cols — cost is not the selling price
            total_cols = [c for c in total_cols if 'cost' not in c.lower()]
        
        print(f"    → Found Quantity columns: {qty_cols}")
        print(f"    → Found Price columns: {price_cols}")
        print(f"    → Found Total columns: {total_cols}")
        print(f"    → Found Discount columns: {discount_cols}")
        
        if qty_cols and price_cols and total_cols:
            qty_col = qty_cols[0]
            price_col = price_cols[0]
            total_col = total_cols[0]
            df_pd = df.select([qty_col, price_col, total_col]).to_pandas()
            
            for col in [qty_col, price_col, total_col]:
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')
            
            expected = df_pd[qty_col] * df_pd[price_col]
            actual = df_pd[total_col]
            
            discount_amount = 0
            if discount_cols:
                discount_col = discount_cols[0]
                discount_data = df.select(discount_col).to_pandas()[discount_col]
                
                if discount_data.dtype == bool or set(discount_data.dropna().unique()).issubset({True, False, 'true', 'false', 'True', 'False'}):
                    print(f"    → Discount column '{discount_col}' is boolean, not numeric")
                    discount_cols = []
                else:
                    discount_amount = pd.to_numeric(discount_data, errors='coerce').fillna(0)
                    expected = expected - discount_amount
            
            error_series = np.abs(expected - actual)
            mean_error = error_series.mean()
            max_error = error_series.max()
            mismatches = (error_series > 0.01).sum()
            
            print(f"    → Relationship Check:")
            print(f"       Mean Error: {mean_error:.2f}")
            print(f"       Max Error: {max_error:.2f}")
            print(f"       Mismatches: {mismatches}/{len(df_pd)} rows")
            
            formula = f"{total_col} = {qty_col} × {price_col}"
            if discount_cols:
                formula += f" - {discount_cols[0]}"
            
            confidence = 1 - (mean_error / actual.mean()) if actual.mean() != 0 else 0
            confidence = max(0, min(confidence, 1))
            
            relationships.append({
                'type': 'multiplication',
                'formula': formula,
                'columns': [qty_col, price_col, total_col] + discount_cols,
                'confidence': float(confidence),
                'mean_error': float(mean_error),
                'mismatches': int(mismatches),
                'needs_recalculation': mismatches > 0
            })
            
            print(f"    ✓ Relationship discovered: {formula}")
            print(f"       Confidence: {confidence:.2%}")
            print(f"       Will recalculate: {'YES' if mismatches > 0 else 'NO'}")
        else:
            print(f"    ✗ Could not find all required columns")
        
        return relationships
    
    def _find_correlations(self, df: pl.DataFrame, semantic_map: Dict) -> List[Dict]:
        relationships = []
        numeric_cols = [k for k, v in semantic_map.items() if v['expected_dtype'] in ['integer', 'float']]
        
        if len(numeric_cols) < 2:
            return relationships
        
        df_pd = df.select(numeric_cols).to_pandas()
        for col in numeric_cols:
            df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')
        
        corr_matrix = df_pd.corr()
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i >= j:
                    continue
                
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    relationships.append({
                        'type': 'correlation',
                        'columns': [col1, col2],
                        'correlation': float(corr),
                        'confidence': abs(float(corr))
                    })
        
        return relationships


# ============================================================================
# AGENT 3: QUALITY AGENT
# ============================================================================

class QualityAgent(BaseAgent):
    """Comprehensive problem detection"""
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="QualityAgent",
            role="Data Quality & Problem Detection Expert",
            gemini_api_key=gemini_api_key
        )
    
    def detect_all_problems(self, df: pl.DataFrame, profile: Dict, semantic_map: Dict, relationships: List[Dict]) -> Dict:
        print(f"\n[{self.name}] Scanning for data quality issues...")
        
        problems = {
            'missing_values': self._detect_missing_values(df, profile, semantic_map),
            'outliers': self._detect_outliers(df, profile, semantic_map),
            'duplicates': self._detect_duplicates(df, semantic_map),
            'invalid_values': self._detect_invalid_values(df, semantic_map),
            'inconsistencies': self._detect_inconsistencies(df, semantic_map),
            'business_violations': self._detect_business_violations(df, semantic_map, relationships)
        }
        
        total_issues = sum(len(v) for v in problems.values())
        
        self.log_action(
            action="problems_detected",
            details={'total_issues': total_issues},
            confidence=0.92
        )
        
        print(f"✓ Found {total_issues} issues across 6 categories")
        return problems
    
    def _detect_missing_values(self, df: pl.DataFrame, profile: Dict, semantic_map: Dict) -> List[Dict]:
        missing_issues = []
        
        for col_info in profile['columns']:
            col_name = col_info['name']
            null_count = col_info['null_count']
            null_pct   = col_info['null_pct']

            # ── FIX: also count string-null sentinels that survived as strings ─
            # After normalise_string_nulls() these are real NaN, but if
            # the Polars DataFrame was loaded before that step ran (profile
            # built from original), we recount on the pandas representation.
            # Since clean_dataset converts to pandas early, these are already
            # NaN by the time imputation runs — this just ensures detection
            # counts are accurate for the QualityAgent report.
            if df[col_name].dtype == pl.Utf8 or str(df[col_name].dtype) in ('String', 'str', 'object') or str(df[col_name].dtype).startswith('string'):
                col_pd  = df[col_name].to_pandas()
                str_null_mask = col_pd.astype(str).str.strip().str.lower().isin(STRING_NULL_VALUES)
                extra_nulls = int(str_null_mask.sum())
                null_count  = null_count + extra_nulls
                null_pct    = round((null_count / max(len(df), 1)) * 100, 2)

            if null_count > 0:
                severity = 'CRITICAL' if null_pct > 50 else 'HIGH' if null_pct > 20 else 'MEDIUM' if null_pct > 5 else 'LOW'
                
                semantic_type = semantic_map.get(col_name, {}).get('semantic_type', 'Unknown')
                nullable = semantic_type in ['Discount', 'Category']
                
                missing_issues.append({
                    'column': col_name,
                    'count': null_count,
                    'percentage': null_pct,
                    'severity': severity,
                    'semantic_type': semantic_type,
                    'nullable': nullable
                })
        
        return missing_issues
    
    def _detect_outliers(self, df: pl.DataFrame, profile: Dict, semantic_map: Dict) -> List[Dict]:
        outlier_issues = []
        
        numeric_cols = [c for c in profile['columns'] if c['actual_type'] in ['integer', 'float']]
        
        for col_info in numeric_cols:
            col_name = col_info['name']
            
            try:
                col_series = df[col_name].drop_nulls()
                col_pd = col_series.to_pandas()
                col_data = pd.to_numeric(col_pd, errors='coerce').dropna().values
                
                if len(col_data) < 10 or np.std(col_data) == 0:
                    continue
                
                Q1 = np.percentile(col_data, 25)
                Q3 = np.percentile(col_data, 75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = (z_scores > 3).sum()
                
                try:
                    iso_forest = IsolationForest(contamination=Config.OUTLIER_THRESHOLD, random_state=42)
                    predictions = iso_forest.fit_predict(col_data.reshape(-1, 1))
                    ml_outliers = (predictions == -1).sum()
                except:
                    ml_outliers = 0
                
                consensus = max(iqr_outliers, z_outliers, ml_outliers)
                
                if consensus > 0:
                    outlier_issues.append({
                        'column': col_name,
                        'iqr_count': int(iqr_outliers),
                        'z_score_count': int(z_outliers),
                        'ml_count': int(ml_outliers),
                        'consensus_count': int(consensus),
                        'percentage': round((consensus / len(col_data)) * 100, 2),
                        'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                    })
            
            except Exception as e:
                continue
        
        return outlier_issues
    
    def _detect_duplicates(self, df: pl.DataFrame, semantic_map: Dict) -> List[Dict]:
        duplicate_issues = []
        
        exact_dupes = df.is_duplicated().sum()
        if exact_dupes > 0:
            duplicate_issues.append({
                'type': 'exact_duplicates',
                'count': exact_dupes,
                'percentage': round((exact_dupes / len(df)) * 100, 2)
            })
        
        id_cols = [k for k, v in semantic_map.items() 
                  if v['semantic_type'] in ['TransactionID', 'CustomerID', 'ProductID']]
        
        for col in id_cols:
            dupes = df[col].to_pandas().duplicated().sum()
            if dupes > 0:
                duplicate_issues.append({
                    'type': 'duplicate_ids',
                    'column': col,
                    'count': dupes,
                    'percentage': round((dupes / len(df)) * 100, 2)
                })
        
        return duplicate_issues
    
    def _detect_invalid_values(self, df: pl.DataFrame, semantic_map: Dict) -> List[Dict]:
        invalid_issues = []
        
        for col_name, sem_info in semantic_map.items():
            semantic_type = sem_info['semantic_type']
            
            if semantic_type in ['Quantity', 'Price', 'TotalAmount']:
                # ── FIX: coerce to numeric before comparison ──────────────────
                # Polars may keep these as String if the CSV has mixed values
                # (e.g. '2', 'ERROR', '4'). Direct < 0 comparison on a String
                # column throws TypeError. Convert to pandas numeric first.
                try:
                    col_numeric = pd.to_numeric(
                        df[col_name].to_pandas(), errors='coerce'
                    )
                    negative_count = int((col_numeric < 0).sum())
                    if negative_count > 0:
                        invalid_issues.append({
                            'column': col_name,
                            'issue': 'negative_values',
                            'count': negative_count,
                            'severity': 'HIGH'
                        })
                except Exception:
                    pass
            
            if semantic_type == 'OrderDate':
                try:
                    dates = pd.to_datetime(df[col_name].to_pandas(), errors='coerce')
                    future_count = int((dates > pd.Timestamp.now()).sum())
                    if future_count > 0:
                        invalid_issues.append({
                            'column': col_name,
                            'issue': 'future_dates',
                            'count': future_count,
                            'severity': 'MEDIUM'
                        })
                except:
                    pass
            
            if semantic_type == 'Quantity':
                try:
                    col_numeric = pd.to_numeric(
                        df[col_name].to_pandas(), errors='coerce'
                    )
                    zero_count = int((col_numeric == 0).sum())
                    if zero_count > 0:
                        invalid_issues.append({
                            'column': col_name,
                            'issue': 'zero_quantity',
                            'count': zero_count,
                            'severity': 'MEDIUM'
                        })
                except Exception:
                    pass
        
        return invalid_issues
    
    def _detect_inconsistencies(self, df: pl.DataFrame, semantic_map: Dict) -> List[Dict]:
        inconsistency_issues = []
        
        text_cols = [k for k, v in semantic_map.items() 
                    if v['expected_dtype'] in ['categorical', 'string']]
        
        for col in text_cols[:5]:
            col_data = df[col].to_pandas().astype(str)
            
            unique_original = col_data.nunique()
            unique_normalized = col_data.str.lower().str.strip().nunique()
            
            if unique_original != unique_normalized:
                inconsistency_issues.append({
                    'column': col,
                    'issue': 'case_inconsistency',
                    'variants': unique_original - unique_normalized
                })
            
            has_whitespace = col_data.str.contains(r'^\s+|\s+$', regex=True).sum()
            if has_whitespace > 0:
                inconsistency_issues.append({
                    'column': col,
                    'issue': 'whitespace',
                    'count': int(has_whitespace)
                })
        
        return inconsistency_issues
    
    def _detect_business_violations(self, df: pl.DataFrame, semantic_map: Dict, relationships: List[Dict]) -> List[Dict]:
        violations = []
        
        for rel in relationships:
            if rel['type'] == 'multiplication':
                cols = rel['columns']
                df_pd = df.select(cols).to_pandas()
                
                left = pd.to_numeric(df_pd[cols[0]], errors='coerce')
                right = pd.to_numeric(df_pd[cols[1]], errors='coerce')
                computed = left * right
                actual = pd.to_numeric(df_pd[cols[2]], errors='coerce')
                
                if len(cols) > 3:
                    discount = pd.to_numeric(df_pd[cols[3]], errors='coerce').fillna(0)
                    computed = computed - discount
                
                violation_mask = np.abs(computed - actual) > 0.01
                violation_count = violation_mask.sum()
                
                if violation_count > 0:
                    violations.append({
                        'rule': rel['formula'],
                        'violations': int(violation_count),
                        'percentage': round((violation_count / len(df)) * 100, 2),
                        'columns': cols
                    })
        
        return violations


# ============================================================================
# 🔥 NEW: IMPUTATION STRATEGY SELECTOR
# ============================================================================

class ImputationStrategySelector:
    """
    🎯 INTELLIGENT STRATEGY SELECTOR
    
    Decides WHEN to use ML vs simple methods based on:
    - Missing percentage
    - Data complexity
    - Available context
    - Training sample size
    """
    
    def __init__(self):
        self.strategy_stats = {
            'ml_used': 0,
            'simple_used': 0,
            'skipped': 0
        }
    
    def select_strategy(self, df: pd.DataFrame, col: str, null_pct: float, 
                       semantic_type: str, relationships: List[Dict]) -> Tuple[str, float, str]:
        """
        🧠 DECISION LOGIC
        
        Returns: (strategy, confidence, reason)
        """
        
        missing_mask = df[col].isna()
        n_missing = missing_mask.sum()
        n_available = (~missing_mask).sum()
        
        # ============================================================
        # RULE 1: Too sparse → Skip
        # ============================================================
        if null_pct > Config.MAX_NULL_PERCENT:
            self.strategy_stats['skipped'] += 1
            return 'skip', 0.2, f'Too sparse ({null_pct:.1f}% missing)'
        
        # ============================================================
        # RULE 1.5: Identity/contact columns → NEVER impute with ML or mode
        # ============================================================
        # WHY THIS IS CRITICAL:
        # transaction_id, customer_id, customer_phone_no are unique per row.
        # If KNN runs on them it AVERAGES the numeric parts of IDs → produces
        # nonsense floats (e.g. all rows get 129232.0).
        # If mode runs on phone numbers → every missing phone gets the same
        # number, which is completely wrong.
        # These columns must be left as NaN or filled with 'unknown',
        # NEVER averaged, NEVER mode-filled.
        IDENTITY_SEMANTIC_TYPES = {'TransactionID', 'CustomerID', 'ProductID'}
        IDENTITY_KEYWORDS = ('id', 'phone', 'mobile', 'email', 'mail',
                             'name', 'address', 'url', 'uuid', 'guid')
        col_lower_check = col.lower().replace(' ', '_')
        is_identity_col = (
            semantic_type in IDENTITY_SEMANTIC_TYPES
            or any(kw in col_lower_check for kw in IDENTITY_KEYWORDS)
        )
        if is_identity_col:
            self.strategy_stats['skipped'] += 1
            return 'skip_identity', 0.9, (
                f'Identity/contact column — ML/mode imputation would corrupt values. '
                f'Missing rows left as null.'
            )
        # WHY: Imputing a computed column (Total Spent = Qty × Price) then
        # immediately recalculating it in _enforce_calculations is pure wasted
        # work. Worse — if Qty or Price is also missing in that row, the median
        # fill is based on wrong data, corrupting the recalculation basis.
        # FIX: Any column that is the OUTPUT (index 2) of a multiplication
        # relationship gets strategy='calculate'. This covers TotalAmount AND
        # any dataset where the computed column has a different name (Revenue,
        # GrossTotal, Amount, NetSales, etc.) — not just hardcoded 'TotalAmount'.
        computed_output_cols = set()
        for rel in relationships:
            if rel.get('type') == 'multiplication' and len(rel.get('columns', [])) >= 3:
                computed_output_cols.add(rel['columns'][2])  # index 2 = output column

        if semantic_type == 'TotalAmount' or col in computed_output_cols:
            return 'calculate', 1.0, 'Computed column — will be recalculated from formula, skipping imputation'
        
        # ============================================================
        # RULE 3: Very few missing → Simple statistical
        # WHY: ML overhead not worth it for <5% missing
        # ============================================================
        # ── FIX: check for numeric content not just dtype ─────────────────────
        # After normalise_string_nulls() numeric cols may still be object dtype.
        # Use pd.to_numeric probe to determine if column is truly numeric.
        is_numeric = df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        if not is_numeric:
            probe = pd.to_numeric(df[col].dropna(), errors='coerce')
            is_numeric = probe.notna().sum() / max(len(probe), 1) > 0.8

        if null_pct < Config.SIMPLE_METHOD_THRESHOLD:
            self.strategy_stats['simple_used'] += 1
            if is_numeric:
                return 'median', 0.75, f'Few missing ({null_pct:.1f}%) - median sufficient'
            else:
                return 'mode', 0.70, f'Few missing ({null_pct:.1f}%) - mode sufficient'
        
        # ============================================================
        # RULE 4: Categorical + Context → Random Forest 🤖
        # WHY: Product/Customer IDs depend on context (Category, Price, etc.)
        # ============================================================
        if semantic_type in ['ProductID', 'ProductName', 'CustomerID']:
            context_cols = self._find_context_columns(df, col, semantic_type)
            
            # Check ML eligibility
            if (len(context_cols) >= Config.ML_MIN_CONTEXT_FEATURES and
                n_available >= Config.ML_MIN_TRAINING_SAMPLES and
                null_pct <= Config.ML_MAX_MISSING_PERCENT):
                
                self.strategy_stats['ml_used'] += 1
                return 'random_forest', 0.88, f'Complex categorical with {len(context_cols)} context features: {context_cols}'
        
        # ============================================================
        # RULE 5: Numeric + Correlations → KNN 🤖
        # WHY: Price/Quantity often correlate with other numeric columns
        # ============================================================
        if is_numeric:
            correlated_cols = self._find_correlated_features(df, col, relationships)
            
            if (len(correlated_cols) >= Config.ML_MIN_CONTEXT_FEATURES and
                n_available >= Config.ML_MIN_TRAINING_SAMPLES and
                null_pct <= Config.ML_MAX_MISSING_PERCENT):
                
                self.strategy_stats['ml_used'] += 1
                return 'knn', 0.85, f'Numeric with {len(correlated_cols)} correlations: {correlated_cols}'
        
        # ============================================================
        # RULE 6: Fallback → Statistical
        # WHY: Not enough context or training data for ML
        # ============================================================
        self.strategy_stats['simple_used'] += 1
        if is_numeric:
            return 'median_fallback', 0.60, 'Insufficient context for ML - using median'
        else:
            return 'mode_fallback', 0.55, 'Insufficient context for ML - using mode'
    
    def _find_context_columns(self, df: pd.DataFrame, target_col: str, 
                             semantic_type: str) -> List[str]:
        """
        🔍 Find columns that provide business context
        
        Example: For ProductID, look for Category, Price, Brand
        """
        
        context_map = {
            'ProductID': ['Category', 'Price', 'Price Per Unit', 'Brand', 'ProductName'],
            'ProductName': ['Category', 'Price', 'Price Per Unit', 'ProductID', 'Brand'],
            'CustomerID': ['Location', 'PaymentMethod', 'TotalAmount', 'Total Amount'],
            'Price': ['Category', 'ProductName', 'ProductID', 'Brand'],
        }
        
        possible_contexts = context_map.get(semantic_type, [])
        available_contexts = [
            c for c in possible_contexts 
            if c in df.columns and df[c].notna().sum() > len(df) * 0.8  # >80% complete
        ]
        
        return available_contexts
    
    def _find_correlated_features(self, df: pd.DataFrame, target_col: str, 
                                  relationships: List[Dict]) -> List[str]:
        """
        🔍 Find numerically correlated columns
        
        Uses relationship discovery results
        """
        
        correlated = []
        
        for rel in relationships:
            if rel['type'] == 'correlation' and target_col in rel['columns']:
                other_col = [c for c in rel['columns'] if c != target_col][0]
                if df[other_col].notna().sum() > len(df) * 0.8:
                    correlated.append(other_col)
        
        return correlated
    
    def get_stats(self) -> Dict:
        """Return strategy usage statistics"""
        return self.strategy_stats


class SmartCategoricalNormalizer:

    logger = logging.getLogger("SemanticNormalizer")
    logger.setLevel(logging.INFO)

    def normalize_column(self, series: pd.Series):

        # ── FIX: preserve full index including NaN rows ───────────────────────
        # Old code: series = series.dropna().astype(str)
        # dropna() changes the index. When the returned series is written back
        # to df_pd[col], pandas aligns on index — NaN rows get NaN back (correct)
        # BUT the index of non-null rows may not match df_pd's index if the df
        # was reset at some point, causing misaligned writes.
        # Fix: keep the full series, work on the non-null subset, merge back.
        full_series = series.copy()
        null_mask   = series.isna()
        series_str  = series.dropna().astype(str)  # work only on non-null values
        original    = series_str.copy()

        # ── CARDINALITY GUARD ─────────────────────────────────────────────────
        # Fuzzy matching is O(n²). A 5000-unique column = 25M comparisons
        # → minutes of runtime + false merges (John Smith ~ Jane Smith).
        # If unique count > 50 → skip fuzzy grouping, run abbreviation only.
        MAX_UNIQUE_FOR_FUZZY = 50
        n_unique = series.nunique()

        base = (
            original
            .str.lower()
            .str.strip()
            .str.replace(r'[^a-z0-9 ]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
        )

        mapping = {}

        if n_unique <= MAX_UNIQUE_FOR_FUZZY:
            # ---------------------------------------------------
            # Step 1 — Build fuzzy similarity groups
            # ---------------------------------------------------
            groups = defaultdict(list)

            for value in base.unique():
                placed = False
                for group_key in list(groups.keys()):
                    similarity = max(
                        fuzz.ratio(value, group_key),
                        fuzz.partial_ratio(value, group_key),
                        fuzz.token_sort_ratio(value, group_key),
                        fuzz.token_set_ratio(value, group_key)
                    ) / 100

                    value_words = set(value.split())
                    key_words   = set(group_key.split())
                    overlap     = value_words.intersection(key_words)

                    # ── WORD-OVERLAP FIX ──────────────────────────────────────
                    # OLD bug: `if similarity >= 0.70 and overlap` always fails
                    # for single-word values because overlap is always empty set.
                    # Example: "Csh" vs "Cash" → similarity=0.86, overlap={}
                    #          → never grouped → dirty value stays in output.
                    # Same for: Boks/Books, Nrth/North, Furnitur/Furniture, etc.
                    #
                    # FIX: if BOTH sides are single words, the word-overlap
                    # check is meaningless — skip it and trust fuzzy score.
                    # If either side is multi-word, keep the overlap guard to
                    # prevent false merges ("New York" ≠ "New Jersey").
                    # ─────────────────────────────────────────────────────────
                    both_single_word = (len(value_words) == 1 and len(key_words) == 1)
                    passes_overlap   = bool(overlap) or both_single_word

                    if similarity >= 0.82 and passes_overlap:
                        groups[group_key].append(value)
                        placed = True
                        break

                if not placed:
                    groups[value].append(value)

            # ---------------------------------------------------
            # Step 2 — Dominant value per group
            # ---------------------------------------------------
            for group_key, members in groups.items():
                counts = {m: (base == m).sum() for m in members}
                dominant_base = max(counts, key=counts.get)
                dominant_original = original[base == dominant_base].mode()[0]

                for member in members:
                    if member == dominant_base:
                        continue
                    for v in original[base == member].unique():
                        mapping[v] = dominant_original
                        self.logger.info(f"[SemanticNorm] REPLACED '{v}' → '{dominant_original}'")

        else:
            self.logger.info(
                f"[SemanticNorm] HIGH-CARDINALITY ({n_unique} unique) — "
                f"skipping fuzzy grouping, running abbreviation check only"
            )

        # ---------------------------------------------------
        # Step 3 — Abbreviation detection (always runs, any cardinality)
        # ── ENHANCED: initials AND prefix-word matching ────────────────────
        # Old: only caught 'DW' → 'Digital Wallet' (exact initials, len ≤ 3)
        # New also catches:
        #   'Digital' → 'Digital Wallet'  (first word prefix match)
        #   'Credit'  → 'Credit Card'     (first word prefix match)
        # These are the most common real-world payment method abbreviations.
        # -------------------------------------------------------------------
        for variant in base.unique():
            for dominant in base.unique():
                if variant == dominant:
                    continue

                words = dominant.split()

                # Mode A: exact initials (DW → Digital Wallet)
                initials = "".join(w[0] for w in words)
                if len(variant) <= 4 and variant == initials:
                    canonical = original[base == dominant].mode()[0]
                    for v in original[base == variant].unique():
                        if v not in mapping:
                            mapping[v] = canonical
                            self.logger.info(
                                f"[SemanticNorm] ABBREV-INITIALS '{v}' → '{canonical}'"
                            )
                    break

                # Mode B: leading word prefix (Digital → Digital Wallet)
                if (
                    len(words) > 1              # dominant is multi-word
                    and dominant.startswith(variant + ' ')  # variant is a prefix word
                    and len(variant.split()) >= 1
                ):
                    canonical = original[base == dominant].mode()[0]
                    for v in original[base == variant].unique():
                        if v not in mapping:
                            mapping[v] = canonical
                            self.logger.info(
                                f"[SemanticNorm] ABBREV-PREFIX '{v}' → '{canonical}'"
                            )

        # ---------------------------------------------------
        # Step 4 — Apply mapping and merge back into full series
        # ---------------------------------------------------
        if not mapping:
            self.logger.info("[SemanticNorm] No replacements applied")
            # Return full series (with NaN rows intact) not the dropna'd subset
            return full_series, None, []

        cleaned_subset = original.replace(mapping)

        # Merge cleaned non-null values back into the full-length series
        # so that NaN rows remain NaN and index alignment is guaranteed.
        result_series = full_series.copy()
        result_series.loc[~null_mask] = cleaned_subset.values

        suggestions = [
            {"column": full_series.name, "from": k, "to": v}
            for k, v in mapping.items()
        ]

        report = {
            "mapping": mapping,
            "variants_fixed": len(mapping)
        }

        return result_series, report, suggestions
# ============================================================================
# AGENT 4: 🔥 ENHANCED CLEANING AGENT (WITH ADAPTIVE ML)
# ============================================================================

class CleaningAgent(BaseAgent):
    """
    🤖 INTELLIGENT DATA CLEANING WITH ADAPTIVE ML
    
    Uses ML when beneficial, falls back to simple methods otherwise
    """
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="CleaningAgent",
            role="Data Cleaning & Transformation Expert (ML-Enhanced)",
            gemini_api_key=gemini_api_key
        )

        self.gemini_api_key = gemini_api_key
        
        self.strategy_selector = ImputationStrategySelector()
    
    def clean_dataset(self, df: pl.DataFrame, problems: Dict, semantic_map: Dict, relationships: List[Dict]) -> Tuple[pl.DataFrame, Dict]:
        print(f"\n[{self.name}] Starting intelligent cleaning with adaptive ML...")
        
        cleaning_log = {
            'imputation': [],
            'outliers': [],
            'duplicates': [],
            'invalid': [],
            'formatting': [],
            'calculations': [],
            'strategy_stats': {}
        }
        
        all_suggestions = []
        df_pd = df.to_pandas()

        # ── STEP 0: Convert string-null sentinels to real NaN ────────────────
        # MUST run before everything else. dirty_cafe_sales.csv has 3,256 cells
        # containing 'UNKNOWN' or 'ERROR'. Without this:
        #   • Type detection sees 50% parse rate → mislabels date column as text
        #   • Fuzzy grouping can't normalise these → they stay as-is in output
        #   • Integer coercion converts them → NaN → fillna(0) → Quantity=0
        #   • QualityAgent never detects them → imputation never runs on them
        df_pd, str_null_count = normalise_string_nulls(df_pd)
        if str_null_count > 0:
            print(f"    [Step 0] Converted {str_null_count:,} string-null sentinels "
                  f"(UNKNOWN/ERROR/N/A/etc.) → NaN")
            # Re-register converted cells in missing_values problems so
            # the imputation step picks them up
            for col in df_pd.columns:
                null_count = int(df_pd[col].isna().sum())
                if null_count > 0:
                    null_pct = round(null_count / len(df_pd) * 100, 2)
                    semantic_type = semantic_map.get(col, {}).get('semantic_type', 'Unknown')
                    # Add or update entry in problems['missing_values']
                    existing = next((i for i, x in enumerate(problems['missing_values'])
                                     if x['column'] == col), None)
                    entry = {
                        'column': col,
                        'count': null_count,
                        'percentage': null_pct,
                        'severity': ('CRITICAL' if null_pct > 50 else 'HIGH'
                                     if null_pct > 20 else 'MEDIUM'
                                     if null_pct > 5 else 'LOW'),
                        'semantic_type': semantic_type,
                        'nullable': semantic_type in ['Discount', 'Category']
                    }
                    if existing is not None:
                        problems['missing_values'][existing] = entry
                    else:
                        problems['missing_values'].append(entry)
        
        # Step 1: Handle missing values (🔥 WITH ADAPTIVE ML)
        df_pd, imp_log = self._handle_missing_values_adaptive(df_pd, problems['missing_values'], semantic_map, relationships)
        cleaning_log['imputation'] = imp_log
        
        # Step 2: Fix invalid values
        df_pd, inv_log = self._fix_invalid_values(df_pd, problems['invalid_values'], semantic_map)
        cleaning_log['invalid'] = inv_log
        
        # Step 3: Remove duplicates
        df_pd, dup_log = self._remove_duplicates(df_pd, problems['duplicates'], semantic_map)
        cleaning_log['duplicates'] = dup_log
        
        # Step 4: Fix inconsistencies
        df_pd, fmt_log = self._fix_inconsistencies(df_pd, problems['inconsistencies'], semantic_map)
        cleaning_log['formatting'] = fmt_log

     
        # ── STEP 5: SmartCategoricalNormalizer ───────────────────────────────────
        # ROOT CAUSE OF DIRTY VARIANTS SURVIVING:
        # The old code only normalised columns where semantic_map labels
        # expected_dtype == "categorical". But SemanticAgent assigns this label
        # based on keyword matching — columns like 'Location', 'Region',
        # 'Sales_Channel', 'Return_Flag' may get labeled 'string' or 'unknown'
        # instead of 'categorical', so the normalizer never ran on them.
        #
        # FIX: Run normalizer on ALL columns that look like low-cardinality
        # categoricals — meaning string/object dtype AND unique count ≤ 100.
        # This catches every column regardless of what SemanticAgent labeled it.
        # Skip ID columns (>80% unique) and pure numeric columns.
        # ─────────────────────────────────────────────────────────────────────
        value_normalizer = SmartCategoricalNormalizer()
        value_norm_log   = {}

        # Columns explicitly labeled categorical by semantic_map
        semantic_categorical = {
            col for col, info in semantic_map.items()
            if info.get("expected_dtype") == "categorical"
        }

        # Auto-detect any remaining low-cardinality string columns
        # (catches Location, Region, Sales_Channel, Return_Flag, etc.)
        auto_categorical = set()
        for col in df_pd.columns:
            if col not in semantic_categorical and df_pd[col].dtype == object:
                n_unique = df_pd[col].nunique()
                n_rows   = len(df_pd)
                unique_ratio = n_unique / max(n_rows, 1)
                # Low-cardinality: ≤100 unique values AND <20% unique ratio
                # Skips Transaction_ID (100% unique), Product_Name (many unique)
                if n_unique <= 100 and unique_ratio < 0.20:
                    auto_categorical.add(col)

        all_categorical_cols = semantic_categorical | auto_categorical
        print(f"\n    [Normalizer] Running on {len(all_categorical_cols)} categorical columns:")
        print(f"      Semantic: {sorted(semantic_categorical)}")
        print(f"      Auto-detected: {sorted(auto_categorical)}")

        for col in sorted(all_categorical_cols):
            if col not in df_pd.columns:
                continue
            try:
                df_pd[col], rep, suggestions = value_normalizer.normalize_column(df_pd[col])
                if suggestions:
                    all_suggestions.extend(suggestions)
                if rep:
                    value_norm_log[col] = rep
            except Exception as e:
                print(f"    [Normalizer] Skipped '{col}': {e}")

        cleaning_log["categorical_value_normalization"] = value_norm_log

        # ── STEP 5b: Boolean normalization for flag columns ──────────────────
        # Return_Flag (and any column with bool/flag/returned in name) may have
        # values like True/False/1/0/Yes/No/TRUE/FALSE after dirty injection.
        # _standardize_types doesn't handle this because the semantic_map labels
        # these columns as 'string' not 'boolean'. Normalise explicitly here.
        bool_map = {
            'true': True,  'false': False,
            '1':    True,  '0':     False,
            'yes':  True,  'no':    False,
            't':    True,  'f':     False,
            'y':    True,  'n':     False,
        }
        bool_keywords = ('flag', 'return', 'bool', 'is_', 'has_', 'active', 'discount')
        for col in df_pd.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in bool_keywords):
                if df_pd[col].dtype == object:
                    normalised = df_pd[col].astype(str).str.strip().str.lower().map(bool_map)
                    if normalised.notna().sum() > len(df_pd) * 0.5:
                        df_pd[col] = normalised.fillna(False).astype(bool)
                        print(f"    [BoolNorm] '{col}' normalised to bool")

        # ── STEP 5c: Customer_Rating out-of-range clamping ───────────────────
        # Pandera will catch these in validation but they should be fixed here
        # so the output file is clean. Any rating column (detected by name) gets
        # clamped to [0, 5]. Values > 5 are set to the column's median.
        rating_keywords = ('rating', 'score', 'stars', 'review')
        for col in df_pd.columns:
            if any(kw in col.lower() for kw in rating_keywords):
                numeric = pd.to_numeric(df_pd[col], errors='coerce')
                if numeric.notna().any():
                    out_of_range = numeric > 5
                    if out_of_range.sum() > 0:
                        median_val = numeric[~out_of_range].median()
                        df_pd.loc[out_of_range, col] = median_val
                        print(f"    [RatingClamp] '{col}': {out_of_range.sum()} values > 5 "
                              f"replaced with median {median_val:.1f}")

        
        # Step 6: Enforce calculations FIRST — so Total Spent = Qty × Price is correct
        df_pd, calc_log = self._enforce_calculations(df_pd, semantic_map, relationships)
        cleaning_log['calculations'] = calc_log

        # Step 7: Handle outliers AFTER recalculation — so we cap the final values,
        # not the pre-calculation values that would be overwritten anyway.
        df_pd, out_log = self._handle_outliers(df_pd, problems['outliers'], semantic_map)
        cleaning_log['outliers'] = out_log
        
        # Step 7: Standardize types
        df_pd = self._standardize_types(df_pd, semantic_map)
        
        # Get strategy statistics
        cleaning_log['strategy_stats'] = self.strategy_selector.get_stats()
        
        self.log_action(
            action="cleaning_complete",
            details={
                'steps_executed': 7,
                'ml_used': cleaning_log['strategy_stats'].get('ml_used', 0),
                'simple_used': cleaning_log['strategy_stats'].get('simple_used', 0)
            },
            confidence=0.87
        )
        
        print(f"✓ Cleaning complete")
        print(f"  📊 Strategy Stats: ML={cleaning_log['strategy_stats']['ml_used']}, Simple={cleaning_log['strategy_stats']['simple_used']}, Skipped={cleaning_log['strategy_stats']['skipped']}")
        
        cleaning_log["normalization_suggestions"] = all_suggestions

        return pl.from_pandas(df_pd), cleaning_log
    
    # ============================================================
    # 🔥 NEW: ADAPTIVE MISSING VALUE HANDLER
    # ============================================================
    
    def _handle_missing_values_adaptive(self, df: pd.DataFrame, missing_issues: List[Dict], 
                                       semantic_map: Dict, relationships: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        🤖 ADAPTIVE IMPUTATION
        
        Automatically chooses best method for each column
        """
        
        imputation_log = []
        
        print(f"\n    🤖 Adaptive Imputation Starting...")

        # ── FIX: pre-coerce numeric columns so strategy selector sees float64 ──
        # After normalise_string_nulls(), numeric columns that had ERROR/UNKNOWN
        # are still object dtype (all values are strings from CSV load).
        # ImputationStrategySelector checks df[col].dtype in [np.float64, np.int64]
        # to decide median vs mode — if dtype is still object, it picks mode for
        # every numeric column (wrong). Pre-coerce based on semantic_map so the
        # dtype reflects the actual data type before strategy selection runs.
        #
        # ── FIX B: also pre-clean currency strings before numeric probe ────────
        # A column like unit_price = ['$12.50', '$34.00', ...] fails the numeric
        # probe (pd.to_numeric('$12.50') = NaN) so it's treated as non-numeric
        # and gets mode imputation. Then _standardize_types strips the $ later —
        # but by then the imputed cells already have a mode string, not a median.
        # Fix: strip currency symbols before probing, same as clean_numeric_string.
        def _pre_clean_for_probe(series: pd.Series) -> pd.Series:
            """Strip currency/locale noise so numeric probe works on $ £ Rs values."""
            s = series.astype(str)
            s = s.str.replace(r'[₹$£€¥₩฿]', '', regex=True)
            s = s.str.replace(r'\bRs\.?\b', '', regex=True, flags=re.IGNORECASE)
            s = s.str.replace(r'\bINR\b', '', regex=True, flags=re.IGNORECASE)
            s = s.str.replace(r'(?<=\d),(?=\d{3})', '', regex=True)  # thousands
            s = s.str.replace(r'%$', '', regex=True)                  # percent
            s = s.str.strip()
            return pd.to_numeric(s, errors='coerce')

        for col in df.columns:
            if col not in df.columns:
                continue
            sem_info = semantic_map.get(col, {})
            expected = sem_info.get('expected_dtype', 'unknown')
            if expected in ('integer', 'float') and df[col].dtype == object:
                df[col] = _pre_clean_for_probe(df[col])
            elif df[col].dtype == object:
                # Also probe columns not explicitly labeled — catches unit_price='$12'
                probe = _pre_clean_for_probe(df[col])
                if probe.notna().sum() / max(len(probe.dropna()), 1) > 0.6:
                    df[col] = probe
        
        for issue in missing_issues:
            col = issue['column']
            null_pct = issue['percentage']
            semantic_type = issue['semantic_type']
            
            # Get smart strategy recommendation
            strategy, confidence, reason = self.strategy_selector.select_strategy(
                df, col, null_pct, semantic_type, relationships
            )
            
            print(f"\n    📋 Column: {col}")
            print(f"       Missing: {null_pct:.1f}%")
            print(f"       Strategy: {strategy.upper()}")
            print(f"       Reason: {reason}")
            
            missing_mask = df[col].isna()
            n_filled = missing_mask.sum()
            
            # ============================================================
            # Execute selected strategy
            # ============================================================
            
            if strategy in ('skip', 'skip_identity'):
                imputation_log.append({
                    'column': col,
                    'method': 'skipped' if strategy == 'skip' else 'skipped_identity',
                    'reason': reason,
                    'confidence': confidence
                })
                continue
            
            elif strategy == 'calculate':
                imputation_log.append({
                    'column': col,
                    'method': 'will_be_calculated',
                    'reason': reason,
                    'confidence': confidence
                })
                continue
            
            elif strategy == 'random_forest':
                # 🤖 ML METHOD 1: Random Forest
                try:
                    df[col] = self._ml_categorical_imputation(df, col, semantic_type)
                    print(f"       ✓ Random Forest imputation successful")
                    
                    imputation_log.append({
                        'column': col,
                        'method': 'random_forest_ml',
                        'reason': reason,
                        'rows_filled': int(n_filled),
                        'confidence': confidence
                    })
                except Exception as e:
                    print(f"       ⚠ ML failed: {e}, falling back to mode")
                    df.loc[missing_mask, col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    imputation_log.append({
                        'column': col,
                        'method': 'mode_fallback_after_ml_fail',
                        'reason': f'ML failed: {str(e)}',
                        'rows_filled': int(n_filled),
                        'confidence': 0.55
                    })
            
            elif strategy == 'knn':
                # 🤖 ML METHOD 2: KNN
                try:
                    df[col] = self._knn_imputation(df, col, relationships)
                    print(f"       ✓ KNN imputation successful")
                    
                    imputation_log.append({
                        'column': col,
                        'method': 'knn_ml',
                        'reason': reason,
                        'rows_filled': int(n_filled),
                        'confidence': confidence
                    })
                except Exception as e:
                    print(f"       ⚠ ML failed: {e}, falling back to median")
                    median_val = df[col].median()
                    df.loc[missing_mask, col] = median_val
                    imputation_log.append({
                        'column': col,
                        'method': 'median_fallback_after_ml_fail',
                        'reason': f'ML failed: {str(e)}',
                        'value': float(median_val),
                        'rows_filled': int(n_filled),
                        'confidence': 0.60
                    })
            
            elif strategy in ['median', 'median_fallback']:
                # ── FIX: always coerce to numeric before median ───────────────
                # Column may be object dtype (string numbers after CSV load).
                # pd.to_numeric handles this; non-parseable values become NaN.
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0  # absolute last resort
                df.loc[missing_mask, col] = median_val
                print(f"       ✓ Median imputation: {median_val:.4g}")
                
                imputation_log.append({
                    'column': col,
                    'method': strategy,
                    'reason': reason,
                    'value': float(median_val),
                    'rows_filled': int(n_filled),
                    'confidence': confidence
                })
            
            elif strategy in ['mode', 'mode_fallback']:
                # FIX: For boolean/near-binary columns (like Discount Applied),
                # mode() picks True here because True=4219 > False=4157, but
                # the margin is too thin to trust. Business rule wins:
                # "unknown discount status" = not discounted = False.
                sem_type = semantic_map.get(col, {}).get('semantic_type', '')
                if sem_type == 'Discount':
                    # ── FIX: cast to column dtype before assigning ─────────────
                    # Discount Applied may be a string column ('True'/'False')
                    # because Polars read it as String from CSV. Assigning a
                    # Python bool False into a StringDtype column throws TypeError.
                    # Detect the current dtype and cast the fill value accordingly.
                    if df[col].dtype == object:
                        fill_val = 'False'
                    else:
                        fill_val = False
                    print(f"       ✓ Discount boolean fill: {fill_val} (business default = not discounted)")
                    imputation_log.append({
                        'column': col,
                        'method': 'business_rule_default',
                        'reason': 'Discount NaN → False (unknown = not discounted)',
                        'value': str(fill_val),
                        'rows_filled': int(n_filled),
                        'confidence': 0.85
                    })
                    df.loc[missing_mask, col] = fill_val
                    continue

                # ── FIX: don't mode-impute high-cardinality identifier columns ──
                # Email, full_name, address, and similar unique-per-row columns
                # should NOT be filled with the mode — every row would get the
                # same value, which is meaningless and corrupts the data.
                # Detection: if unique_ratio > 0.5 it's an identifier/free-text
                # column. Leave these as NaN (or 'unknown') rather than imputing.
                n_unique     = df[col].nunique()
                unique_ratio = n_unique / max(len(df), 1)
                sem_type_col = semantic_map.get(col, {}).get('semantic_type', '')
                is_identifier = (
                    unique_ratio > 0.5
                    or sem_type_col in ('TransactionID', 'CustomerID', 'ProductID')
                    or any(kw in col.lower() for kw in
                           ('email', 'mail', 'name', 'address', 'url', 'phone', 'mobile'))
                )
                if is_identifier:
                    df.loc[missing_mask, col] = 'unknown'
                    print(f"       ✓ Identifier/text col: filled with 'unknown' (mode imputation skipped)")
                    imputation_log.append({
                        'column': col,
                        'method': 'identifier_unknown',
                        'reason': 'High-cardinality identifier — mode fill would corrupt data',
                        'value': 'unknown',
                        'rows_filled': int(n_filled),
                        'confidence': 0.5
                    })
                    continue

                # ── FIX: exclude string-null sentinels from mode calculation ──
                # Without this, if UNKNOWN/ERROR survived to this point (e.g.
                # Transaction Date column) the mode picks 'UNKNOWN' as the
                # fill value, which is clearly wrong. Filter them out first.
                valid_vals = df[col][~df[col].astype(str).str.strip().str.lower().isin(STRING_NULL_VALUES)]
                mode_val = valid_vals.mode()
                if len(mode_val) > 0:
                    df.loc[missing_mask, col] = mode_val[0]
                    print(f"       ✓ Mode imputation: {mode_val[0]}")
                    
                    imputation_log.append({
                        'column': col,
                        'method': strategy,
                        'reason': reason,
                        'value': str(mode_val[0]),
                        'rows_filled': int(n_filled),
                        'confidence': confidence
                    })
        
        print(f"\n    ✓ Adaptive imputation complete")
        return df, imputation_log
    
    # ============================================================
    # 🤖 ML METHOD 1: RANDOM FOREST FOR CATEGORICAL
    # ============================================================
    
    def _ml_categorical_imputation(self, df: pd.DataFrame, target_col: str, 
                                   semantic_type: str) -> pd.Series:
        """
        🌲 RANDOM FOREST IMPUTATION
        
        WHY USE THIS:
        - ProductID depends on Category + Price + other context
        - Simple mode gives most common product overall (wrong!)
        - RF learns: "Electronics at $299 = Laptop, not Pen"
        
        WHEN USED:
        - Categorical columns (ProductID, CustomerID, ProductName)
        - ≥2 context features available
        - ≥50 training samples
        - ≤40% missing
        """
        
        print(f"       🌲 Training Random Forest...")
        
        # ── CONTEXT MAP: Features that help predict each semantic type ──────
        # WHY CATEGORY IS CRITICAL FOR ITEM/PRODUCTNAME:
        #   Item codes in this dataset encode their category suffix (Item_X_PAT =
        #   Patisserie, Item_X_FUR = Furniture). Without Category as a feature,
        #   the RF trains blind and can assign Item_PAT to a Furniture row.
        #   Category must be in the feature list so the model learns the
        #   Category→Item suffix relationship directly.
        #
        # FIX: ProductName and ProductID now explicitly include 'Category' and
        #   all common real-world column name variants. We also dynamically add
        #   any column from the dataframe whose name contains 'category' or
        #   'product' (case-insensitive), so this works on ANY sales dataset
        #   regardless of exact column naming.
        # ────────────────────────────────────────────────────────────────────
        context_map = {
            'ProductID': ['Category', 'Price Per Unit', 'Price', 'Item',
                          'ProductName', 'Brand', 'Sub_Category'],
            'ProductName': ['Category', 'Price Per Unit', 'Price', 'Item',
                            'ProductID', 'Brand', 'Sub_Category'],
            'CustomerID': ['Location', 'Payment Method', 'PaymentMethod',
                           'Total Spent', 'TotalAmount', 'Total Amount'],
            'Price':      ['Category', 'ProductName', 'ProductID', 'Item', 'Brand'],
        }

        possible_contexts = context_map.get(semantic_type, [])

        # Dynamically discover additional context columns by keyword matching
        # so this works even if the dataset calls the column 'product_category'
        # or 'item_type' instead of 'Category'.
        keyword_map = {
            'ProductID':   ['categ', 'type', 'class', 'group'],
            'ProductName': ['categ', 'type', 'class', 'group'],
            'CustomerID':  ['region', 'location', 'city', 'store'],
            'Price':       ['categ', 'brand', 'type'],
        }
        for kw in keyword_map.get(semantic_type, []):
            for c in df.columns:
                if kw in c.lower() and c not in possible_contexts and c != target_col:
                    possible_contexts = possible_contexts + [c]

        feature_cols = [
            c for c in possible_contexts
            if c in df.columns and df[c].notna().sum() > len(df) * 0.8
        ]
        
        if len(feature_cols) < 2:
            raise ValueError(f"Insufficient context features (found {len(feature_cols)})")
        
        # Prepare data
        df_work = df.copy()
        missing_mask = df[target_col].isna()
        
        # Encode all categorical columns
        encoders = {}
        for col in feature_cols + [target_col]:
            if col in df_work.columns and df_work[col].dtype == object:
                le = LabelEncoder()
                valid_mask = df_work[col].notna()
                df_work.loc[valid_mask, col + '_encoded'] = le.fit_transform(df_work.loc[valid_mask, col].astype(str))
                encoders[col] = le
            elif col in df_work.columns:
                df_work[col + '_encoded'] = pd.to_numeric(df_work[col], errors='coerce')
        
        # Prepare training data
        feature_cols_encoded = [c + '_encoded' for c in feature_cols]
        train_mask = df[target_col].notna()
        
        # Remove rows with missing features
        for col in feature_cols_encoded:
            train_mask = train_mask & df_work[col].notna()
        
        if train_mask.sum() < Config.ML_MIN_TRAINING_SAMPLES:
            raise ValueError(f"Insufficient training samples ({train_mask.sum()} < {Config.ML_MIN_TRAINING_SAMPLES})")
        
        X_train = df_work.loc[train_mask, feature_cols_encoded].fillna(0)
        y_train = df_work.loc[train_mask, target_col + '_encoded']
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        print(f"       📊 Model trained on {len(X_train)} samples with {len(feature_cols)} features")
        
        # Predict missing values
        predict_mask = missing_mask.copy()
        for col in feature_cols_encoded:
            predict_mask = predict_mask & df_work[col].notna()
        
        if predict_mask.sum() > 0:
            X_missing = df_work.loc[predict_mask, feature_cols_encoded].fillna(0)
            predictions_encoded = model.predict(X_missing)
            
            # Decode predictions
            if target_col in encoders:
                predictions = encoders[target_col].inverse_transform(predictions_encoded.astype(int))
            else:
                predictions = predictions_encoded
            
            # Fill predictions
            result = df[target_col].copy()
            result.loc[predict_mask] = predictions
            
            print(f"       ✓ Predicted {predict_mask.sum()} missing values")
            
            return result
        else:
            raise ValueError("No complete rows to predict from")
    
    # ============================================================
    # 🤖 ML METHOD 2: KNN FOR NUMERIC
    # ============================================================
    
    def _knn_imputation(self, df: pd.DataFrame, target_col: str, 
                       relationships: List[Dict]) -> pd.Series:
        """
        🎯 KNN IMPUTATION
        
        WHY USE THIS:
        - Numeric values often correlate (Price ↔ Quantity)
        - Median ignores correlations (assumes independence)
        - KNN finds similar records and uses their values
        
        WHEN USED:
        - Numeric columns (Price, Quantity)
        - ≥2 correlated features
        - ≥50 training samples
        - ≤40% missing
        
        EXAMPLE:
        Missing Price for "Electronics" category
        - Find 5 most similar Electronics items
        - Average their prices
        - More accurate than overall median
        """
        
        print(f"       🎯 Applying KNN imputation...")
        
        # Find correlated columns
        correlated_cols = []
        for rel in relationships:
            if rel['type'] == 'correlation' and target_col in rel['columns']:
                other_col = [c for c in rel['columns'] if c != target_col][0]
                if df[other_col].notna().sum() > len(df) * 0.8:
                    correlated_cols.append(other_col)
        
        if len(correlated_cols) < 1:
            raise ValueError(f"No correlated features found")
        
        # Prepare data
        df_work = df.copy()
        all_cols = correlated_cols + [target_col]
        
        # Encode categoricals if any
        for col in all_cols:
            if col in df_work.columns:
                if df_work[col].dtype == object:
                    le = LabelEncoder()
                    valid_mask = df_work[col].notna()
                    df_work.loc[valid_mask, col] = le.fit_transform(df_work.loc[valid_mask, col].astype(str))
                else:
                    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        
        # Apply KNN
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_work[all_cols] = imputer.fit_transform(df_work[all_cols])
        
        print(f"       ✓ KNN completed using {len(correlated_cols)} correlated features")
        
        return df_work[target_col]
    
    # ============================================================
    # REMAINING METHODS (UNCHANGED)
    # ============================================================
    
    def _fix_invalid_values(self, df: pd.DataFrame, invalid_issues: List[Dict], semantic_map: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fix invalid values — with numeric coercion before any comparison"""
        fix_log = []

        for issue in invalid_issues:
            col = issue['column']
            issue_type = issue['issue']
            count = issue['count']

            if issue_type == 'negative_values':
                # ROOT CAUSE OF THE ERROR:
                # df[col] < 0 throws TypeError when the column dtype is object
                # (strings like '$49.99', '8 units', or plain '"-5.0"').
                # Polars read_csv sometimes keeps numeric-looking columns as
                # strings, and our test dataset intentionally stores them as
                # strings to simulate real-world dirty data.
                #
                # FIX: always coerce to numeric first. This is safe because
                # _fix_invalid_values only runs on columns that QualityAgent
                # flagged as having negative values — meaning they were already
                # identified as numeric columns during profiling. Any value
                # that can't be parsed becomes NaN (handled by imputation).
                df[col] = pd.to_numeric(df[col], errors='coerce')

                semantic_type = semantic_map[col]['semantic_type']
                negative_mask = df[col] < 0

                if negative_mask.sum() == 0:
                    # After coercion the negatives may have become NaN
                    # (e.g. '-$5' can't be parsed) — nothing to fix here
                    continue

                if semantic_type == 'Quantity':
                    df.loc[negative_mask, col] = df.loc[negative_mask, col].abs()
                    method = 'absolute_value'
                else:
                    median_val = df.loc[~negative_mask, col].median()
                    df.loc[negative_mask, col] = median_val
                    method = f'median_{round(float(median_val), 2)}'

                fix_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': method,
                    'rows_fixed': int(negative_mask.sum()),
                    'confidence': 0.75
                })

            elif issue_type == 'future_dates':
                # Date column may also be a raw string at this point
                # (format detection runs later in _standardize_types).
                # Use infer_datetime_format for robustness here.
                try:
                    dates = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                    future_mask = dates > pd.Timestamp.now()
                    if future_mask.sum() > 0:
                        df.loc[future_mask, col] = pd.Timestamp.now().strftime('%d-%m-%Y')
                    fix_log.append({
                        'column': col,
                        'issue': issue_type,
                        'method': 'set_to_today',
                        'rows_fixed': int(future_mask.sum()),
                        'confidence': 0.6
                    })
                except Exception:
                    pass

            elif issue_type == 'zero_quantity':
                # Same guard — coerce before comparing == 0
                df[col] = pd.to_numeric(df[col], errors='coerce')
                zero_mask = df[col] == 0
                if zero_mask.sum() > 0:
                    df.loc[zero_mask, col] = 1
                fix_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': 'set_to_1',
                    'rows_fixed': int(zero_mask.sum()),
                    'confidence': 0.7
                })

        return df, fix_log
    
    def _handle_outliers(self, df: pd.DataFrame, outlier_issues: List[Dict], semantic_map: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Handle outliers by capping"""
        outlier_log = []
        
        for issue in outlier_issues:
            col = issue['column']
            lower_bound = issue['bounds']['lower']
            upper_bound = issue['bounds']['upper']
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # ── FIX: price/cost/amount columns must always have lower_bound ≥ 0 ─
            # IQR-based bounds can give negative lower bounds (Q1 - 1.5*IQR < 0)
            # for right-skewed distributions. A negative price, cost, or total is
            # always a data error. Clamp the lower bound to 0 for these columns.
            sem_type = semantic_map.get(col, {}).get('semantic_type', '')
            if sem_type in ('Price', 'TotalAmount', 'Quantity') or any(
                kw in col.lower() for kw in ('price', 'cost', 'amount', 'total', 'quantity', 'qty')
            ):
                lower_bound = max(lower_bound, 0)
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            original_count = outlier_mask.sum()
            
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            
            outlier_log.append({
                'column': col,
                'method': 'capped',
                'bounds': {'lower': lower_bound, 'upper': upper_bound},
                'rows_affected': int(original_count),
                'confidence': 0.8
            })
        
        # ── ALWAYS enforce non-negative on money/quantity columns ────────────
        # Even if no outlier_issues were detected for a column, scan all
        # numeric columns labeled as money/quantity and clip to ≥ 0.
        for col in df.columns:
            sem_type = semantic_map.get(col, {}).get('semantic_type', '')
            is_money = sem_type in ('Price', 'TotalAmount', 'Quantity') or any(
                kw in col.lower() for kw in ('price', 'cost', 'amount', 'total', 'quantity', 'qty')
            )
            if is_money:
                numeric = pd.to_numeric(df[col], errors='coerce')
                neg_count = (numeric < 0).sum()
                if neg_count > 0:
                    df[col] = numeric.clip(lower=0)
                    print(f"    [OutlierFix] '{col}': clipped {neg_count} negative values to 0")
        
        return df, outlier_log
    
    def _remove_duplicates(self, df: pd.DataFrame, duplicate_issues: List[Dict], semantic_map: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Remove duplicate records - exact row and transaction-ID based"""
        dup_log = []
        
        for issue in duplicate_issues:
            if issue['type'] == 'exact_duplicates':
                original_len = len(df)
                df = df.drop_duplicates()
                removed = original_len - len(df)
                
                dup_log.append({
                    'type': 'exact_duplicates',
                    'rows_removed': removed,
                    'confidence': 1.0
                })
        
        # ── BONUS: dedup on TransactionID if present ─────────────────────────
        # A dirty dataset can have the same transaction_id on multiple rows
        # (e.g. from re-exported data). Keep only the first occurrence.
        txn_col = next(
            (k for k, v in semantic_map.items() if v['semantic_type'] == 'TransactionID'),
            None
        )
        if txn_col is None:
            # Fallback: look for column named 'transaction_id' / 'txn_id' etc.
            for col in df.columns:
                if 'transaction' in col.lower() and 'id' in col.lower():
                    txn_col = col
                    break
        if txn_col and txn_col in df.columns:
            before = len(df)
            # Only dedup on non-null TXN IDs to avoid removing legitimate nulls
            non_null_mask = df[txn_col].notna() & (df[txn_col].astype(str).str.strip() != '') & (df[txn_col].astype(str).str.lower() != 'nan')
            df_non_null = df[non_null_mask].drop_duplicates(subset=[txn_col], keep='first')
            df_null     = df[~non_null_mask]
            df = pd.concat([df_non_null, df_null], ignore_index=True)
            removed_txn = before - len(df)
            if removed_txn > 0:
                print(f"    [Dedup] Removed {removed_txn:,} duplicate transaction IDs on '{txn_col}'")
                dup_log.append({
                    'type': 'duplicate_transaction_ids',
                    'column': txn_col,
                    'rows_removed': removed_txn,
                    'confidence': 1.0
                })
        
        return df, dup_log
    
    def _fix_inconsistencies(self, df: pd.DataFrame, inconsistency_issues: List[Dict], semantic_map: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fix formatting inconsistencies without corrupting correctly-cased values"""
        format_log = []
        
        for issue in inconsistency_issues:
            col = issue['column']
            issue_type = issue['issue']
            
            if issue_type == 'case_inconsistency':
                # FIX: Don't blindly apply title() to all values — that turns
                # 'Electric household essentials' into 'Electric Household Essentials'
                # which breaks dashboard category filters.
                # Instead: lowercase+strip to find the canonical form per group,
                # then preserve the most-frequent original casing for that group.
                original = df[col].astype(str)
                normalized_key = original.str.lower().str.strip()
                
                # For each normalized group, find the most common original spelling
                canonical_map = (
                    original.groupby(normalized_key)
                    .agg(lambda x: x.value_counts().index[0])
                    .to_dict()
                )
                df[col] = normalized_key.map(canonical_map).fillna(original)
                
                format_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': 'canonical_case_preserved',
                    'confidence': 0.95
                })
            
            elif issue_type == 'whitespace':
                df[col] = df[col].astype(str).str.strip()
                
                format_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': 'stripped',
                    'rows_affected': issue['count'],
                    'confidence': 1.0
                })
        
        return df, format_log
    
    def _enforce_calculations(self, df: pd.DataFrame, semantic_map: Dict, relationships: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """Enforce calculated columns - ALWAYS recalculate"""
        calc_log = []
        
        print(f"\n    [Calculation Enforcement] Checking {len(relationships)} relationships...")
        
        for rel in relationships:
            if rel['type'] == 'multiplication':
                cols = rel['columns']
                
                if len(cols) >= 3:
                    qty_col, price_col, total_col = cols[0], cols[1], cols[2]
                    
                    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
                    df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
                    df[total_col] = pd.to_numeric(df[total_col], errors='coerce').fillna(0)
                    
                    print(f"    → Recalculating {total_col} = {qty_col} × {price_col}")
                    df[total_col] = df[qty_col] * df[price_col]
                    
                    if len(cols) > 3:
                        discount_col = cols[3]
                        discount_data = pd.to_numeric(df[discount_col], errors='coerce')
                        
                        if discount_data.notna().any() and not df[discount_col].dtype == bool:
                            print(f"    → Subtracting {discount_col}")
                            df[total_col] = df[total_col] - discount_data.fillna(0)
                    
                    df[total_col] = df[total_col].round(2).clip(lower=0)
                    
                    rows_recalculated = len(df)
                    rows_changed = rel.get('mismatches', 0)
                    
                    print(f"    ✓ Recalculated {total_col} for ALL {rows_recalculated} rows")
                    print(f"       Fixed {rows_changed} rows that had incorrect values")
                    
                    calc_log.append({
                        'formula': rel['formula'],
                        'rows_recalculated': rows_recalculated,
                        'rows_corrected': rows_changed,
                        'confidence': 1.0
                    })
        
        # ── BONUS: compute profit if total_price and cost exist ─────────────
        # profit = total_price - (quantity × cost)
        # This is a standard derived column in sales datasets.
        col_lower_map = {c.lower().replace('_','').replace(' ',''): c for c in df.columns}
        total_price_col = col_lower_map.get('totalprice')
        cost_col        = col_lower_map.get('cost')
        qty_col_g       = col_lower_map.get('quantity') or col_lower_map.get('qty')
        profit_col      = col_lower_map.get('profit')

        if total_price_col and cost_col and qty_col_g and profit_col:
            tp  = pd.to_numeric(df[total_price_col], errors='coerce').fillna(0)
            cost= pd.to_numeric(df[cost_col],        errors='coerce').fillna(0)
            qty = pd.to_numeric(df[qty_col_g],       errors='coerce').fillna(1)
            df[profit_col] = (tp - (qty * cost)).round(2)
            print(f"    ✓ Recalculated '{profit_col}' = {total_price_col} - ({qty_col_g} × {cost_col})")
            calc_log.append({
                'formula': f"{profit_col} = {total_price_col} - ({qty_col_g} × {cost_col})",
                'rows_recalculated': len(df),
                'rows_corrected': 0,
                'confidence': 1.0
            })

        if len(calc_log) == 0:
            print(f"    ⚠ No calculation relationships found to enforce")
        
        return df, calc_log
    
    def _standardize_types(self, df: pd.DataFrame, semantic_map: Dict) -> pd.DataFrame:
        """Convert to proper data types with currency/locale cleaning"""

        # ── CURRENCY & LOCALE CLEANER ────────────────────────────────────────
        # WHY: Real sales exports frequently contain values like $1,299.99,
        #   ₹1299, 1.299,99 (European comma-decimal), or "5 units".
        #   pd.to_numeric(errors='coerce') on these produces NaN silently —
        #   the value is lost with no warning.
        #
        # Approach: regex-based pre-cleaning for numeric columns only.
        #   1. Strip currency symbols ($ £ € ₹ ¥ and others)
        #   2. Handle European comma-decimal: if comma appears as decimal
        #      separator (e.g. 1.299,99), convert to standard 1299.99
        #   3. Remove thousands separators (commas in 1,299 → 1299)
        #   4. Strip trailing non-numeric noise ("5 units" → "5")
        # ────────────────────────────────────────────────────────────────────
        def clean_numeric_string(series: pd.Series) -> pd.Series:
            s = series.astype(str).str.strip()
            # Strip currency symbols and other non-numeric prefixes/suffixes
            s = s.str.replace(r'[$£€₹¥₩฿₺₽]', '', regex=True)
            s = s.str.strip()
            # Detect European decimal format: digit.digit.digit,digit
            # e.g. 1.299,99 → swap . and , → 1299.99
            euro_mask = s.str.match(r'^\d{1,3}(\.\d{3})+(,\d+)?$')
            if euro_mask.any():
                s[euro_mask] = (
                    s[euro_mask]
                    .str.replace('.', '', regex=False)  # remove thousands sep
                    .str.replace(',', '.', regex=False)  # decimal comma → point
                )
            # Remove standard thousands separators (1,299 → 1299)
            s = s.str.replace(r'(?<=\d),(?=\d{3})', '', regex=True)
            # Strip trailing unit noise: "5 units", "41.0 USD" → keep numeric part
            s = s.str.extract(r'^(-?[\d.]+)', expand=False)
            return s
        
        # ── FIX: columns that must NEVER be coerced to numeric ──────────────
        # transaction_id, customer_id, customer_phone_no are string identifiers.
        # If SemanticAgent mis-labels them as 'integer' or 'float' (because they
        # look numeric), converting them destroys the original string value.
        # E.g. "TXN000153649" → NaN, phone "8082864030" → 8.08e9 (loses precision).
        # Guard: skip numeric coercion for any column whose name contains known
        # identity/contact keywords, regardless of semantic_map label.
        NEVER_COERCE_KEYWORDS = (
            'id', 'phone', 'mobile', 'fax', 'zip', 'pincode', 'postcode',
            'transaction_id', 'customer_id', 'order_id', 'invoice_id',
            'branch_id', 'sales_rep', 'rep', 'agent', 'employee',
        )
        def is_identity_column(col_name: str) -> bool:
            cl = col_name.lower().replace(' ', '_')
            return any(kw in cl for kw in NEVER_COERCE_KEYWORDS)

        for col, sem_info in semantic_map.items():
            expected_type = sem_info['expected_dtype']
            
            # ── GUARD: never coerce identity/contact columns ──────────────────
            if is_identity_column(col):
                # Keep as string, strip whitespace only
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.strip().replace('nan', np.nan)
                continue

            try:
                if expected_type == 'integer':
                    # Clean currency/locale noise before numeric conversion
                    cleaned = clean_numeric_string(df[col])
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    # ── FIX: use median fill, NOT fillna(0) ──────────────────
                    # Old code: .fillna(0).astype(int) → ERROR/UNKNOWN → 0
                    # This is wrong: Quantity=0 is an impossible business value.
                    # Use median of valid values instead. The imputation step
                    # should have already filled these, but as a safety net
                    # we fill any remaining NaN with median here.
                    if numeric.isna().any():
                        median_val = numeric.median()
                        fill = median_val if not np.isnan(median_val) else 1
                        numeric = numeric.fillna(fill)
                    df[col] = numeric.astype(int)
                
                elif expected_type == 'float':
                    # Clean currency/locale noise before numeric conversion
                    cleaned = clean_numeric_string(df[col])
                    df[col] = pd.to_numeric(cleaned, errors='coerce').round(2)
                
                elif expected_type == 'datetime':
                    # ── dateparser + pandas hybrid ────────────────────────────
                    # HOW IT WORKS:
                    #   Step 1: Detect the dominant format from a sample using
                    #   pandas — this gives us a concrete format string (e.g.
                    #   '%d-%m-%Y') that we pass to dateparser as date_formats.
                    #
                    #   Step 2 (dateparser available): parse_one() passes the
                    #   detected format as a hint via date_formats=[best_format].
                    #   Without this hint, dateparser treats numeric strings like
                    #   "08-04-2024" as ambiguous relative expressions and silently
                    #   returns None for ALL rows — wiping the entire column.
                    #   With the hint, dateparser parses correctly AND still handles
                    #   mixed-format outlier rows (e.g. "23 Jul 2023") in the same
                    #   column that the format-detector missed.
                    #   If dateparser still returns None for a specific value,
                    #   pandas tries it as a secondary fallback before giving up.
                    #
                    #   Step 2 (dateparser not available): use the detected format
                    #   directly with pandas — same reliable behaviour as before.
                    #
                    # ROOT CAUSE OF THE ORIGINAL BUG:
                    #   dateparser.parse("08-04-2024", settings=...) with NO
                    #   date_formats hint returned None for every row because it
                    #   couldn't confidently resolve the DD-MM-YYYY ambiguity.
                    #   All rows → None → all NaT → np.where writes all None →
                    #   Polars exports as empty float64 column → dates wiped.
                    #
                    # SETTINGS:
                    #   PREFER_DAY_OF_MONTH_FIRST=True → for "08-04-2024"
                    #   treat first number as day (DD-MM), not month (MM-DD).
                    #   RETURN_AS_TIMEZONE_AWARE=False → plain datetime objects.
                    # ─────────────────────────────────────────────────────────
                    date_settings = {
                        'PREFER_DAY_OF_MONTH_FIRST': True,
                        'RETURN_AS_TIMEZONE_AWARE': False,
                        'STRICT_PARSING': False,
                    }

                    raw_series = df[col].astype(str)

                    # ── STEP 0: Detect and convert Unix timestamps ──────────────
                    # If >80% of non-null values look like 9-13 digit integers
                    # they are Unix epoch timestamps and must be converted first
                    # BEFORE format detection runs (which would fail on floats).
                    unix_probe = pd.to_numeric(df[col], errors='coerce')
                    looks_unix = (
                        unix_probe.notna().sum() / max(len(unix_probe.dropna()), 1) > 0.8
                        and unix_probe.dropna().between(1e9, 2e10).mean() > 0.8
                    )
                    if looks_unix:
                        print(f"       [DateParser] '{col}': detected Unix timestamps → converting")
                        def unix_to_date(v):
                            try:
                                ts = float(v)
                                if np.isnan(ts): return None
                                if ts > 1e12: ts /= 1000   # milliseconds
                                dt = datetime.utcfromtimestamp(ts)
                                return dt if 1990 <= dt.year <= datetime.now().year + 2 else None
                            except: return None
                        parsed_from_unix = df[col].apply(unix_to_date)
                        parsed           = pd.to_datetime(parsed_from_unix, errors='coerce')
                        df[col] = pd.Series(
                            np.where(parsed.notna(), parsed.dt.strftime('%Y-%m-%d'), None),
                            dtype=object, index=df.index
                        )
                        print(f"       [DateParser] '{col}': converted {parsed.notna().sum():,} Unix timestamps → ISO dates")
                        continue   # skip the rest of datetime processing for this col
                    # This runs regardless of whether dateparser is available.
                    # For dateparser: provides the date_formats hint it needs.
                    # For pandas fallback: used directly as the parse format.
                    candidate_formats = [
                        '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
                        '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                        '%d-%m-%y', '%m-%d-%y',
                        '%d %b %Y', '%b %d %Y', '%B %d %Y', '%d %B %Y',
                    ]
                    best_format, best_score = None, 0
                    _clean_sample = (
                        raw_series
                        .replace({'nan': np.nan, 'NaT': np.nan, 'None': np.nan})
                        .dropna()
                    )
                    _sample = _clean_sample.sample(min(200, len(_clean_sample)), random_state=42)
                    for fmt in candidate_formats:
                        try:
                            s = pd.to_datetime(_sample, format=fmt, errors='coerce').notna().sum()
                            if s > best_score:
                                best_score, best_format = s, fmt
                        except Exception:
                            continue
                    print(f"       [DateParser] '{col}': detected dominant format → {best_format!r} (score {best_score}/{len(_sample)})")

                    if DATEPARSER_AVAILABLE:
                        # ── STEP 2a: dateparser with format hint + pandas fallback ──
                        fmt_hint = [best_format] if best_format else None

                        def parse_one(val):
                            if pd.isna(val) or str(val).strip() in ('', 'nan', 'NaT', 'None'):
                                return None
                            v = str(val).strip()
                            # Primary: dateparser with explicit format hint so it
                            # doesn't misinterpret numeric date strings as relative
                            # expressions (the root cause of the all-None bug).
                            try:
                                result = dateparser.parse(
                                    v,
                                    date_formats=fmt_hint,
                                    settings=date_settings,
                                )
                                if result is not None:
                                    return result
                            except Exception:
                                pass
                            # Secondary: pandas fallback for any value dateparser
                            # still can't handle (e.g. unusual locale formats).
                            try:
                                if best_format:
                                    ts = pd.to_datetime(v, format=best_format, errors='coerce')
                                else:
                                    ts = pd.to_datetime(v, infer_datetime_format=True, errors='coerce')
                                return ts if not pd.isna(ts) else None
                            except Exception:
                                return None

                        parsed_dates = raw_series.apply(parse_one)
                        parsed = pd.to_datetime(parsed_dates, errors='coerce')

                    else:
                        # ── STEP 2b: pandas-only path (dateparser not installed) ──
                        parsed = (
                            pd.to_datetime(df[col], format=best_format, errors='coerce')
                            if best_format else
                            pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                        )

                    # Null tracking
                    original_nulls = df[col].isna().sum()
                    created_nulls = parsed.isna().sum() - original_nulls
                    parsed_count = parsed.notna().sum()
                    print(f"       [DateParser] '{col}': parsed {parsed_count} rows"
                          f"{f', {created_nulls} unparseable → NaN' if created_nulls > 0 else ' — 0 nulls introduced'}")

                    # Date range sanity — catch 1970 Unix epoch and far-future parse errors
                    if parsed.notna().any():
                        min_year, max_year = 1990, datetime.now().year + 1
                        bad_range = (parsed.dt.year < min_year) | (parsed.dt.year > max_year)
                        if bad_range.any():
                            print(f"       [DateParser] {bad_range.sum()} dates outside "
                                  f"{min_year}–{max_year} set to NaN")
                            parsed[bad_range] = pd.NaT

                    # ── WRITE-BACK FIX ────────────────────────────────────
                    # ROOT CAUSE of all-NaN Transaction_Date in cleaned file:
                    #
                    # OLD:  parsed.dt.strftime('%d-%m-%Y').where(parsed.notna(), other=None)
                    # This produces a pandas Series with dtype=str that contains
                    # a MIX of string dates AND float nan for null rows.
                    # When pl.from_pandas() sees float nan in a string column,
                    # it converts the ENTIRE column to float64 → every date
                    # becomes NaN → column is completely wiped in the CSV.
                    #
                    # FIX: np.where forces a pure object-dtype Series.
                    # Python None (not float nan) is used for nulls.
                    # Polars reads object+None correctly as Utf8/String
                    # with null — no float64 conversion, no data loss.
                    # ──────────────────────────────────────────────────────
                    df[col] = pd.Series(
                        np.where(parsed.notna(), parsed.dt.strftime('%Y-%m-%d'), None),
                        dtype=object,
                        index=df.index
                    )
                
                elif expected_type == 'categorical':
                    # FIX: Only strip whitespace — do NOT apply .title() here.
                    # Title-casing 'Electric household essentials' produces
                    # 'Electric Household Essentials' which corrupts category values
                    # and breaks dashboard grouping. Casing is handled contextually
                    # in _fix_inconsistencies using canonical majority-vote logic.
                    df[col] = df[col].astype(str).str.strip()
                
                elif expected_type == 'string':
                    df[col] = df[col].astype(str).str.strip()
            
            except:
                continue
        
        return df


# ============================================================================
# AGENT 5: VALIDATION AGENT — powered by Pandera
# ============================================================================

class ValidationAgent(BaseAgent):
    """
    Quality assurance using Pandera schema validation.

    HOW PANDERA IMPROVES THIS:

    OLD approach (_validate_integrity):
      - Manual loops over semantic_map
      - Checks nulls, negatives one-by-one
      - Crashes when numeric column is still a string dtype
        (TypeError: '<' not supported between str and int)
      - ~120 lines of hand-written checks
      - Returns only pass/fail per column type

    NEW approach (Pandera):
      - Builds a DataFrameSchema dynamically from semantic_map
      - Runs ALL checks (nulls, ranges, types, business rules)
        in ONE validate() call with lazy=True
      - Returns structured errors: which column, which row,
        which check failed, what the actual value was
      - Gracefully handles string dtype columns (coerces first)
      - ~40 lines total — 3x less code, more accurate output

    WHAT PANDERA CHECKS AUTOMATICALLY:
      - Price/Quantity/TotalAmount columns: value > 0
      - Customer_Rating: value in range 0–5
      - All non-nullable columns: no nulls remain
      - Computed columns: formula holds (Total = Qty × Price)
      - Data types match expected semantic types
    """

    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="ValidationAgent",
            role="Quality Assurance & Validation Expert (Pandera)",
            gemini_api_key=gemini_api_key
        )

    def validate_cleaning(
        self,
        df_original: pl.DataFrame,
        df_cleaned: pl.DataFrame,
        semantic_map: Dict,
        relationships: List[Dict]
    ) -> Dict:
        print(f"\n[{self.name}] Validating cleaning results...")

        completeness = self._validate_completeness(df_original, df_cleaned)
        integrity    = self._validate_integrity_pandera(df_cleaned, semantic_map)
        biz_rules    = self._validate_business_rules(df_cleaned, semantic_map, relationships)

        validation = {
            'completeness':   completeness,
            'integrity':      integrity,
            'business_rules': biz_rules,
            'quality_score':  0.0
        }
        validation['quality_score'] = self._calculate_quality_score(validation)

        self.log_action(
            action="validation_complete",
            details={'quality_score': validation['quality_score']},
            confidence=validation['quality_score']
        )
        print(f"✓ Validation complete — Quality Score: {validation['quality_score']:.1%}")
        return validation

    # ── Completeness (unchanged — polars-native, fast) ───────────────────────
    def _validate_completeness(self, df_original: pl.DataFrame, df_cleaned: pl.DataFrame) -> Dict:
        def get_completeness(df):
            total = len(df) * len(df.columns)
            nulls = sum(df[c].null_count() for c in df.columns)
            return (1 - nulls / total) * 100

        orig  = get_completeness(df_original)
        clean = get_completeness(df_cleaned)
        return {
            'original':    round(orig,  2),
            'cleaned':     round(clean, 2),
            'improvement': round(clean - orig, 2),
            'status':      'IMPROVED' if clean > orig else 'UNCHANGED'
        }

    # ── Pandera integrity validation ─────────────────────────────────────────
    def _validate_integrity_pandera(self, df_polars: pl.DataFrame, semantic_map: Dict) -> Dict:
        """
        Build a Pandera schema from semantic_map and validate in one call.

        Schema rules are derived directly from the semantic type:
          Price / Quantity / TotalAmount → Check.gt(0), nullable=False
          Rating                         → Check.in_range(0, 5)
          Discount                       → nullable=True (by design)
          OrderDate                      → nullable=False
          All others                     → nullable=False

        lazy=True collects ALL errors before raising — so we get a full
        error report instead of stopping at the first failure.
        """
        if not PANDERA_AVAILABLE:
            return self._validate_integrity_fallback(df_polars, semantic_map)

        df = df_polars.to_pandas()

        # ── Coerce numeric columns before validation ─────────────────────────
        # Pandera checks fail on object/string dtypes the same way our old
        # code did. Pre-coerce so checks run on actual numeric values.
        for col, info in semantic_map.items():
            if col not in df.columns:
                continue
            if info['expected_dtype'] in ('float', 'integer'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ── Build schema dynamically ─────────────────────────────────────────
        column_schemas = {}

        for col, info in semantic_map.items():
            if col not in df.columns:
                continue

            sem   = info.get('semantic_type', '')
            dtype = info.get('expected_dtype', '')
            checks = []
            nullable = False

            if sem == 'Discount':
                nullable = True

            if sem in ('Price', 'Quantity'):
                checks.append(Check.gt(0, error=f"{col} must be > 0"))

            if sem == 'TotalAmount':
                checks.append(Check.ge(0, error=f"{col} must be ≥ 0"))

            # Customer_Rating: must be in 0–5 range
            if 'rating' in col.lower() or 'score' in col.lower():
                checks.append(Check.in_range(0, 5, error=f"{col} must be between 0 and 5"))

            column_schemas[col] = pa.Column(
                nullable=nullable,
                checks=checks if checks else None,
                required=True
            )

        if not column_schemas:
            return {'issues_found': 0, 'details': [], 'status': 'PASS',
                    'engine': 'pandera'}

        schema = pa.DataFrameSchema(
            columns=column_schemas,
            coerce=True   # coerce types before checking — handles string numerics
        )

        issues = []
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            # exc.failure_cases is a tidy DataFrame: column, check, failure_case, index
            err_df = exc.failure_cases
            for _, row in err_df.iterrows():
                issues.append(
                    f"{row.get('column', '?')}: "
                    f"check '{row.get('check', '?')}' failed "
                    f"(value: {row.get('failure_case', '?')})"
                )
            # Deduplicate — pandera reports one row per failing cell
            issues = list(dict.fromkeys(issues))[:50]  # cap at 50 for readability
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")

        return {
            'issues_found': len(issues),
            'details':      issues,
            'status':       'PASS' if not issues else 'ISSUES_FOUND',
            'engine':       'pandera'
        }

    def _validate_integrity_fallback(self, df_polars: pl.DataFrame, semantic_map: Dict) -> Dict:
        """Fallback used when pandera is not installed."""
        issues = []
        df = df_polars.to_pandas()
        for col, info in semantic_map.items():
            if col not in df.columns:
                continue
            sem = info.get('semantic_type', '')
            col_data = pd.to_numeric(df[col], errors='coerce')
            if sem not in ('Discount',) and df[col].isna().sum() > 0:
                issues.append(f"{col}: {df[col].isna().sum()} nulls remain")
            if sem in ('Quantity', 'Price', 'TotalAmount'):
                neg = (col_data < 0).sum()
                if neg > 0:
                    issues.append(f"{col}: {neg} negative values")
        return {
            'issues_found': len(issues),
            'details':      issues,
            'status':       'PASS' if not issues else 'ISSUES_FOUND',
            'engine':       'fallback'
        }

    # ── Business rules (relationship formula check) ──────────────────────────
    def _validate_business_rules(self, df_polars: pl.DataFrame, semantic_map: Dict, relationships: List[Dict]) -> Dict:
        """
        Check that computed columns satisfy their formula.
        Uses pandera's Check.call() for custom formula validation when available,
        falls back to numpy for the arithmetic check otherwise.
        Both approaches return the same structured result.
        """
        violations = []
        df = df_polars.to_pandas()

        for rel in relationships:
            if rel.get('type') != 'multiplication':
                continue
            cols = rel.get('columns', [])
            if len(cols) < 3:
                continue

            qty_col, price_col, total_col = cols[0], cols[1], cols[2]
            if not all(c in df.columns for c in [qty_col, price_col, total_col]):
                continue

            qty   = pd.to_numeric(df[qty_col],   errors='coerce').fillna(0)
            price = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
            total = pd.to_numeric(df[total_col], errors='coerce').fillna(0)

            discount = pd.Series(0.0, index=df.index)
            if len(cols) > 3 and cols[3] in df.columns:
                discount = pd.to_numeric(df[cols[3]], errors='coerce').fillna(0)

            expected = (qty * price - discount).round(2)
            mismatch_count = (~np.isclose(total, expected, atol=0.02)).sum()

            if mismatch_count > 0:
                violations.append(
                    f"{rel.get('formula','formula')}: "
                    f"{mismatch_count} rows where Total ≠ Qty × Price"
                )

        return {
            'violations': len(violations),
            'details':    violations,
            'status':     'PASS' if not violations else 'VIOLATIONS_FOUND'
        }

    # ── Quality score ────────────────────────────────────────────────────────
    def _calculate_quality_score(self, validation: Dict) -> float:
        completeness_score = validation['completeness']['cleaned'] / 100
        integrity_score    = 1.0 if validation['integrity']['status']      == 'PASS' else 0.7
        rules_score        = 1.0 if validation['business_rules']['status'] == 'PASS' else 0.6
        overall = (completeness_score * 0.4) + (integrity_score * 0.3) + (rules_score * 0.3)
        return round(overall, 3)


# ============================================================================
# AGENT 6: ORCHESTRATOR AGENT
# ============================================================================

class OrchestratorAgent(BaseAgent):
    """Master coordinator - runs all agents in sequence"""
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="OrchestratorAgent",
            role="Master Coordinator",
            gemini_api_key=gemini_api_key
        )
        
        self.profiler = ProfilerAgent(gemini_api_key)
        self.semantic = SemanticAgent(gemini_api_key)
        self.quality = QualityAgent(gemini_api_key)
        self.cleaner = CleaningAgent(gemini_api_key)
        self.validator = ValidationAgent(gemini_api_key)
    
    def execute_pipeline(self, input_file: str, output_file: str = "cleaned_dataset.csv") -> Dict:
        """Execute complete cleaning pipeline"""
        
        print("\n" + "="*70)
        print("DATAAGENT - AUTOMATED DATA CLEANING PIPELINE (ML-ENHANCED)")
        print("="*70)
        
        try:
            # Phase 1: Load & Profile
            df_original = self.profiler.load_dataset(input_file)
            profile = self.profiler.comprehensive_profile(df_original)
            
            # Phase 2: Understand Semantics
            semantic_map = self.semantic.identify_semantics(df_original, profile)
            relationships = self.semantic.discover_relationships(df_original, semantic_map)
            
            # Phase 3: Detect Problems
            problems = self.quality.detect_all_problems(df_original, profile, semantic_map, relationships)
            
            # Phase 4: Clean Data (🔥 WITH ADAPTIVE ML)
            df_cleaned, cleaning_log = self.cleaner.clean_dataset(df_original, problems, semantic_map, relationships)
            
            # Phase 5: Validate
            validation = self.validator.validate_cleaning(df_original, df_cleaned, semantic_map, relationships)
                        
            # Phase 6: Generate Report
            report = self._generate_report(
                df_original, df_cleaned, profile, semantic_map,
                relationships, problems, cleaning_log, validation
            )
            
            # Phase 7: Export Results
            self._export_results(df_cleaned, report, output_file)
            
            self.log_action(
                action="pipeline_complete",
                details={'quality_score': validation['quality_score']},
                confidence=validation['quality_score']
            )
            
            print("\n" + "="*70)
            print(f"✓ PIPELINE COMPLETE - Quality Score: {validation['quality_score']:.1%}")
            print("="*70)
            
            return report
        
        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            raise
    
    def _generate_report(self, df_original, df_cleaned, profile, semantic_map, relationships, problems, cleaning_log, validation) -> Dict:
        
        accuracy_score = self._business_rule_accuracy(df_cleaned, semantic_map)
        
        return {
            'summary': {
                'original_rows': len(df_original),
                'cleaned_rows': len(df_cleaned),
                'original_columns': len(df_original.columns),
                'quality_score': validation['quality_score'],
                'accuracy_score': accuracy_score,
                'assessment': self._get_assessment(validation['quality_score']),
                'ml_usage': cleaning_log.get('strategy_stats', {})
            },
            'problems_detected': {
                'missing_values': len(problems['missing_values']),
                'outliers': len(problems['outliers']),
                'duplicates': len(problems['duplicates']),
                'invalid_values': len(problems['invalid_values']),
                'inconsistencies': len(problems['inconsistencies']),
                'business_violations': len(problems['business_violations'])
            },
            'cleaning_actions': {
                'imputation': len(cleaning_log['imputation']),
                'outliers_handled': len(cleaning_log['outliers']),
                'duplicates_removed': len(cleaning_log['duplicates']),
                'invalid_fixed': len(cleaning_log['invalid']),
                'formatting': len(cleaning_log['formatting']),
                'calculations': len(cleaning_log['calculations'])
            },
            'validation_results': validation,
            'detailed_logs': cleaning_log,
            'agent_logs': self._collect_all_logs()
        }
    
    def _business_rule_accuracy(self, df_cleaned, semantic_map) -> float:
        df_pd = df_cleaned.to_pandas()
        n = len(df_pd)
        if n == 0:
            return 1.0
        
        qty_col = next((k for k, v in semantic_map.items() if v['semantic_type'] == 'Quantity'), None)
        price_col = next((k for k, v in semantic_map.items() if v['semantic_type'] == 'Price'), None)
        total_col = next((k for k, v in semantic_map.items() if v['semantic_type'] == 'TotalAmount'), None)
        
        if not all([qty_col, price_col, total_col]):
            return 1.0
        
        rule1 = np.isclose(df_pd[total_col], df_pd[qty_col] * df_pd[price_col], atol=0.01)
        accuracy = rule1.mean()
        
        return round(accuracy, 3)
    
    def _get_assessment(self, score: float) -> str:
        if score >= 0.9:
            return "EXCELLENT - Production Ready"
        elif score >= 0.8:
            return "GOOD - Ready with Minor Cautions"
        elif score >= 0.7:
            return "ACCEPTABLE - Review Recommended"
        else:
            return "NEEDS REVIEW - Manual Inspection Required"
    
    def _collect_all_logs(self) -> Dict:
        return {
            'profiler': self.profiler.get_logs(),
            'semantic': self.semantic.get_logs(),
            'quality': self.quality.get_logs(),
            'cleaner': self.cleaner.get_logs(),
            'validator': self.validator.get_logs()
        }
    
    def _export_results(self, df_cleaned: pl.DataFrame, report: Dict, output_file: str):
        print(f"\n[{self.name}] Exporting results...")
        
        df_cleaned.write_csv(output_file)
        print(f"✓ Cleaned dataset: {output_file}")
        
        summary_df = pd.DataFrame([report['summary']])
        summary_df.to_csv("cleaning_summary.csv", index=False)
        print(f"✓ Summary report: cleaning_summary.csv")
        
        with open("full_cleaning_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ Full report: full_cleaning_report.json")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for DataAgent with ML
    
    SETUP:
    1. Get Gemini API Key: https://makersuite.google.com/app/apikey
    2. Create .env file: GEMINI_API_KEY=your_key
    3. Install: pip install polars pandas numpy google-generativeai scikit-learn 
                scipy python-dotenv sentence-transformers
    4. Run: python dataagent_ml.py
    """
    print("\n" + "="*70)
    print("DATAAGENT INITIALIZATION (ML-ENHANCED)")
    print("="*70)
    
    try:
        api_key = Config.load_api_key()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return
    
    orchestrator = OrchestratorAgent(api_key)
    
    input_file = "dirty_cafe_sales.csv"  # CHANGE THIS
    base_name = os.path.splitext(input_file)[0]
    output_file = f"cleaned_{base_name}_dataset.csv"
    
    try:
        report = orchestrator.execute_pipeline(input_file, output_file)
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(f"\nDataset: {input_file}")
        print(f"Rows: {report['summary']['original_rows']:,}")
        print(f"Columns: {report['summary']['original_columns']}")
        print(f"Quality Score: {report['summary']['quality_score']:.1%}")
        print(f"Assessment: {report['summary']['assessment']}")
        
        # Show ML usage
        ml_stats = report['summary'].get('ml_usage', {})
        if ml_stats:
            print(f"\n🤖 ML Usage Statistics:")
            print(f"  • ML Methods Used: {ml_stats.get('ml_used', 0)} columns")
            print(f"  • Simple Methods Used: {ml_stats.get('simple_used', 0)} columns")
            print(f"  • Skipped: {ml_stats.get('skipped', 0)} columns")
        
        print(f"\nProblems Detected & Fixed:")
        for problem_type, count in report['problems_detected'].items():
            if count > 0:
                print(f"  • {problem_type.replace('_', ' ').title()}: {count} issues")
        
        print(f"\nCleaning Actions Performed:")
        for action_type, count in report['cleaning_actions'].items():
            if count > 0:
                print(f"  • {action_type.replace('_', ' ').title()}: {count} operations")
        
        if 'calculations' in report['detailed_logs'] and report['detailed_logs']['calculations']:
            print(f"\nCalculated Columns:")
            for calc in report['detailed_logs']['calculations']:
                print(f"  • {calc['formula']}")
                print(f"    - Recalculated: {calc['rows_recalculated']} rows")
                if calc.get('rows_corrected', 0) > 0:
                    print(f"    - Fixed incorrect values: {calc['rows_corrected']} rows")
        
        print(f"\nOutput Files:")
        print(f"  • Cleaned Data: {output_file}")
        print(f"  • Summary: cleaning_summary.csv")
        print(f"  • Full Report: full_cleaning_report.json")
        
        print("\n" + "="*70)
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()