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
from pathlib import Path
warnings.filterwarnings('ignore')

# LLM
import google.generativeai as genai

# ML & Analysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from scipy import stats
import re

# Sentence Transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For environment variables
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class Config:
    """Centralized configuration with API key management"""
    
    @staticmethod
    def load_api_key() -> str:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            print("✓ API key loaded from .env file")
            return api_key
        
        print("\n⚠ No API key found in .env file")
        print("\nTo set up .env file:")
        print("1. Create a file named '.env' in your project folder")
        print("2. Add this line: GEMINI_API_KEY=your_actual_key_here")
        print("\nOr enter your API key now (not recommended for production):")
        
        api_key = input("Gemini API Key: ").strip()
        
        if not api_key:
            raise ValueError("API key required. Get one from: https://makersuite.google.com/app/apikey")
        
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
    
    def __init__(self, name: str, role: str, gemini_api_key: str):
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
            
            if score > 0.2:
                try:
                    if config['validation'](series):
                        score += 0.3
                except:
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
        
        qty_cols = [k for k, v in semantic_map.items() if v['semantic_type'] == 'Quantity']
        price_cols = [k for k, v in semantic_map.items() if v['semantic_type'] == 'Price']
        total_cols = [k for k, v in semantic_map.items() if v['semantic_type'] == 'TotalAmount']
        discount_cols = [k for k, v in semantic_map.items() if v['semantic_type'] == 'Discount']
        
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
            if col_info['null_count'] > 0:
                col_name = col_info['name']
                null_pct = col_info['null_pct']
                
                severity = 'CRITICAL' if null_pct > 50 else 'HIGH' if null_pct > 20 else 'MEDIUM' if null_pct > 5 else 'LOW'
                
                semantic_type = semantic_map.get(col_name, {}).get('semantic_type', 'Unknown')
                nullable = semantic_type in ['Discount', 'Category']
                
                missing_issues.append({
                    'column': col_name,
                    'count': col_info['null_count'],
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
                negative_count = (df[col_name] < 0).sum()
                if negative_count > 0:
                    invalid_issues.append({
                        'column': col_name,
                        'issue': 'negative_values',
                        'count': negative_count,
                        'severity': 'HIGH'
                    })
            
            if semantic_type == 'OrderDate':
                try:
                    dates = pd.to_datetime(df[col_name].to_pandas(), errors='coerce')
                    future_count = (dates > pd.Timestamp.now()).sum()
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
                zero_count = (df[col_name] == 0).sum()
                if zero_count > 0:
                    invalid_issues.append({
                        'column': col_name,
                        'issue': 'zero_quantity',
                        'count': zero_count,
                        'severity': 'MEDIUM'
                    })
        
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
        # RULE 2: Calculated column → Will be computed
        # ============================================================
        if semantic_type == 'TotalAmount':
            return 'calculate', 1.0, 'Calculated column - will recompute'
        
        # ============================================================
        # RULE 3: Very few missing → Simple statistical
        # WHY: ML overhead not worth it for <5% missing
        # ============================================================
        if null_pct < Config.SIMPLE_METHOD_THRESHOLD:
            self.strategy_stats['simple_used'] += 1
            if df[col].dtype in [np.float64, np.int64]:
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
        if df[col].dtype in [np.float64, np.int64]:
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
        if df[col].dtype in [np.float64, np.int64]:
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
        
        df_pd = df.to_pandas()
        
        # Step 1: Handle missing values (🔥 WITH ADAPTIVE ML)
        df_pd, imp_log = self._handle_missing_values_adaptive(df_pd, problems['missing_values'], semantic_map, relationships)
        cleaning_log['imputation'] = imp_log
        
        # Step 2: Fix invalid values
        df_pd, inv_log = self._fix_invalid_values(df_pd, problems['invalid_values'], semantic_map)
        cleaning_log['invalid'] = inv_log
        
        # Step 3: Handle outliers
        df_pd, out_log = self._handle_outliers(df_pd, problems['outliers'], semantic_map)
        cleaning_log['outliers'] = out_log
        
        # Step 4: Remove duplicates
        df_pd, dup_log = self._remove_duplicates(df_pd, problems['duplicates'], semantic_map)
        cleaning_log['duplicates'] = dup_log
        
        # Step 5: Fix inconsistencies
        df_pd, fmt_log = self._fix_inconsistencies(df_pd, problems['inconsistencies'], semantic_map)
        cleaning_log['formatting'] = fmt_log
        
        # Step 6: Enforce calculations
        df_pd, calc_log = self._enforce_calculations(df_pd, semantic_map, relationships)
        cleaning_log['calculations'] = calc_log
        
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
            
            if strategy == 'skip':
                imputation_log.append({
                    'column': col,
                    'method': 'skipped',
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
                # Simple statistical method
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                df.loc[missing_mask, col] = median_val
                print(f"       ✓ Median imputation: {median_val:.2f}")
                
                imputation_log.append({
                    'column': col,
                    'method': strategy,
                    'reason': reason,
                    'value': float(median_val),
                    'rows_filled': int(n_filled),
                    'confidence': confidence
                })
            
            elif strategy in ['mode', 'mode_fallback']:
                # Simple statistical method
                mode_val = df[col].mode()
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
        
        # Find context columns
        context_map = {
            'ProductID': ['Category', 'Price', 'Price Per Unit', 'ProductName'],
            'ProductName': ['Category', 'Price', 'Price Per Unit', 'ProductID'],
            'CustomerID': ['Location', 'PaymentMethod', 'Total Amount', 'TotalAmount']
        }
        
        possible_contexts = context_map.get(semantic_type, [])
        feature_cols = [c for c in possible_contexts if c in df.columns and df[c].notna().sum() > len(df) * 0.8]
        
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
        """Fix invalid values"""
        fix_log = []
        
        for issue in invalid_issues:
            col = issue['column']
            issue_type = issue['issue']
            count = issue['count']
            
            if issue_type == 'negative_values':
                semantic_type = semantic_map[col]['semantic_type']
                negative_mask = df[col] < 0
                
                if semantic_type == 'Quantity':
                    df.loc[negative_mask, col] = df.loc[negative_mask, col].abs()
                    method = 'absolute_value'
                else:
                    median_val = df.loc[~negative_mask, col].median()
                    df.loc[negative_mask, col] = median_val
                    method = f'median_{median_val}'
                
                fix_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': method,
                    'rows_fixed': count,
                    'confidence': 0.75
                })
            
            elif issue_type == 'future_dates':
                dates = pd.to_datetime(df[col], errors='coerce')
                future_mask = dates > pd.Timestamp.now()
                df.loc[future_mask, col] = pd.Timestamp.now()
                
                fix_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': 'set_to_today',
                    'rows_fixed': count,
                    'confidence': 0.6
                })
            
            elif issue_type == 'zero_quantity':
                zero_mask = df[col] == 0
                df.loc[zero_mask, col] = 1
                
                fix_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': 'set_to_1',
                    'rows_fixed': count,
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
        
        return df, outlier_log
    
    def _remove_duplicates(self, df: pd.DataFrame, duplicate_issues: List[Dict], semantic_map: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Remove duplicate records"""
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
        
        return df, dup_log
    
    def _fix_inconsistencies(self, df: pd.DataFrame, inconsistency_issues: List[Dict], semantic_map: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Fix formatting inconsistencies"""
        format_log = []
        
        for issue in inconsistency_issues:
            col = issue['column']
            issue_type = issue['issue']
            
            if issue_type == 'case_inconsistency':
                df[col] = df[col].astype(str).str.strip().str.title()
                
                format_log.append({
                    'column': col,
                    'issue': issue_type,
                    'method': 'title_case',
                    'confidence': 0.9
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
        
        if len(calc_log) == 0:
            print(f"    ⚠ No calculation relationships found to enforce")
        
        return df, calc_log
    
    def _standardize_types(self, df: pd.DataFrame, semantic_map: Dict) -> pd.DataFrame:
        """Convert to proper data types"""
        
        for col, sem_info in semantic_map.items():
            expected_type = sem_info['expected_dtype']
            
            try:
                if expected_type == 'integer':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                
                elif expected_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
                
                elif expected_type == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].dt.hour.eq(0).all() and df[col].dt.minute.eq(0).all() and df[col].dt.second.eq(0).all():
                        df[col] = df[col].dt.date
                
                elif expected_type == 'categorical':
                    df[col] = df[col].astype(str).str.strip().str.title()
                
                elif expected_type == 'string':
                    df[col] = df[col].astype(str).str.strip()
            
            except:
                continue
        
        return df


# ============================================================================
# AGENT 5: VALIDATION AGENT
# ============================================================================

class ValidationAgent(BaseAgent):
    """Quality assurance - verify cleaning improved data"""
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="ValidationAgent",
            role="Quality Assurance & Validation Expert",
            gemini_api_key=gemini_api_key
        )
    
    def validate_cleaning(self, df_original: pl.DataFrame, df_cleaned: pl.DataFrame, semantic_map: Dict, relationships: List[Dict]) -> Dict:
        print(f"\n[{self.name}] Validating cleaning results...")
        
        validation = {
            'completeness': self._validate_completeness(df_original, df_cleaned),
            'integrity': self._validate_integrity(df_cleaned, semantic_map),
            'business_rules': self._validate_business_rules(df_cleaned, semantic_map, relationships),
            'quality_score': 0.0
        }
        
        validation['quality_score'] = self._calculate_quality_score(validation)
        
        self.log_action(
            action="validation_complete",
            details={'quality_score': validation['quality_score']},
            confidence=validation['quality_score']
        )
        
        print(f"✓ Validation complete - Quality Score: {validation['quality_score']:.1%}")
        return validation
    
    def _validate_completeness(self, df_original: pl.DataFrame, df_cleaned: pl.DataFrame) -> Dict:
        
        def get_completeness(df):
            total_cells = len(df) * len(df.columns)
            null_cells = sum(df[col].null_count() for col in df.columns)
            return (1 - null_cells / total_cells) * 100
        
        original_completeness = get_completeness(df_original)
        cleaned_completeness = get_completeness(df_cleaned)
        improvement = cleaned_completeness - original_completeness
        
        return {
            'original': round(original_completeness, 2),
            'cleaned': round(cleaned_completeness, 2),
            'improvement': round(improvement, 2),
            'status': 'IMPROVED' if improvement > 0 else 'UNCHANGED'
        }
    
    def _validate_integrity(self, df: pl.DataFrame, semantic_map: Dict) -> Dict:
        issues = []
        
        for col, sem_info in semantic_map.items():
            semantic_type = sem_info['semantic_type']
            col_data = df[col].to_pandas()
            
            if semantic_type not in ['Discount', 'Category']:
                null_count = col_data.isna().sum()
                if null_count > 0:
                    issues.append(f"{col}: {null_count} nulls remain")
            
            if semantic_type in ['Quantity', 'Price', 'TotalAmount']:
                negative_count = (col_data < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} negative values")
        
        return {
            'issues_found': len(issues),
            'details': issues,
            'status': 'PASS' if len(issues) == 0 else 'ISSUES_FOUND'
        }
    
    def _validate_business_rules(self, df: pl.DataFrame, semantic_map: Dict, relationships: List[Dict]) -> Dict:
        violations = []
        
        for rel in relationships:
            if rel['type'] == 'multiplication':
                cols = rel['columns']
                df_pd = df.select(cols).to_pandas()
                
                computed = df_pd[cols[0]] * df_pd[cols[1]]
                actual = df_pd[cols[2]]
                
                if len(cols) > 3:
                    computed = computed - df_pd[cols[3]].fillna(0)
                
                error = np.abs(computed - actual).max()
                
                if error > 0.01:
                    violations.append(f"{rel['formula']}: max error {error:.2f}")
        
        return {
            'violations': len(violations),
            'details': violations,
            'status': 'PASS' if len(violations) == 0 else 'VIOLATIONS_FOUND'
        }
    
    def _calculate_quality_score(self, validation: Dict) -> float:
        completeness_score = validation['completeness']['cleaned'] / 100
        integrity_score = 1.0 if validation['integrity']['status'] == 'PASS' else 0.7
        rules_score = 1.0 if validation['business_rules']['status'] == 'PASS' else 0.6
        
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
    
    input_file = "retail_store_sales.csv"  # CHANGE THIS
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