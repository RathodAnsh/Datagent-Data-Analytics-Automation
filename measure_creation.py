"""
DATAAGENT PHASE 3 - FEATURE ENGINEERING & DASHBOARD PREPARATION
Intelligent column relevance analysis and automated measure creation

This runs AFTER data cleaning is complete.

Architecture:
    1. ColumnRelevanceAgent - Analyzes which columns are useful for dashboards
    2. MeasureCreationAgent - Creates calculated measures (like Power BI)
    3. InteractiveCoordinator - Manages user interaction and confirmation

Author: DataAgent Team
Purpose: Transform cleaned data into dashboard-ready format with user control
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# LLM
import google.generativeai as genai

# For environment variables
from dotenv import load_dotenv
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class DashboardConfig:
    """Configuration for dashboard preparation"""
    
    @staticmethod
    def load_api_key() -> str:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY in .env file")
        return api_key
    
    # Dashboard relevance criteria
    RELEVANCE_THRESHOLD = 0.6
    MIN_UNIQUE_FOR_CATEGORICAL = 2
    MAX_UNIQUE_FOR_CATEGORICAL = 50


# ============================================================================
# BASE AGENT
# ============================================================================

class DashboardAgent:
    """Base class for dashboard preparation agents"""
    
    def __init__(self, name: str, role: str, gemini_api_key: str):
        self.name = name
        self.role = role
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
        )
        
        self.actions_log = []
    
    def query_llm(self, prompt: str, context: str = "") -> str:
        """Query LLM with context"""
        full_prompt = f"""
You are {self.name}, a {self.role}.

CONTEXT:
{context}

TASK:
{prompt}

Provide clear, actionable analysis with business reasoning.
"""
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def log_action(self, action: str, details: Dict):
        """Log action"""
        self.actions_log.append({
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'action': action,
            'details': details
        })


# ============================================================================
# AGENT 1: COLUMN RELEVANCE ANALYZER
# ============================================================================

class ColumnRelevanceAgent(DashboardAgent):
    """
    Analyzes which columns are relevant for business dashboards
    
    Evaluates based on:
    - Business value (KPIs, metrics, dimensions)
    - Data quality (uniqueness, distribution)
    - Dashboard utility (filters, slicers, measures)
    """
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="ColumnRelevanceAgent",
            role="Dashboard Column Relevance Analyst",
            gemini_api_key=gemini_api_key
        )
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
        )
    
    def analyze_columns(self, df: pl.DataFrame) -> Dict[str, Dict]:
        """
        Analyze each column for dashboard relevance
        
        Returns:
            Dictionary with column analysis including relevance score and reasoning
        """
        print(f"\n[{self.name}] Analyzing {len(df.columns)} columns for dashboard relevance...")
        
        column_analysis = {}
        
        for col in df.columns:
            analysis = self._analyze_single_column(df, col)
            column_analysis[col] = analysis
            
            # Print summary
            status = "✓ KEEP" if analysis['relevant'] else "✗ DROP"
            print(f"  {status} | {col:30s} | Score: {analysis['relevance_score']:.2f} | {analysis['role']}")
        
        return column_analysis
    
    def _analyze_single_column(self, df: pl.DataFrame, col: str) -> Dict:
        """Analyze a single column"""
        series = df[col]
        col_pd = series.to_pandas()
        
        # Basic statistics
        total_rows = len(df)
        null_count = series.null_count()
        null_pct = (null_count / total_rows) * 100
        unique_count = series.n_unique()
        unique_pct = (unique_count / total_rows) * 100
        
        # Determine data type
        dtype = self._get_column_type(col_pd)
        
        # Calculate relevance score
        relevance_score, role, reasoning = self._calculate_relevance(
            col, dtype, unique_count, unique_pct, null_pct, total_rows
        )
        
        # Get LLM validation
        llm_reasoning = self._get_llm_opinion(col, dtype, role, reasoning, col_pd.head(10).tolist())
        
        return {
            'column_name': col,
            'data_type': dtype,
            'unique_count': unique_count,
            'unique_pct': round(unique_pct, 2),
            'null_pct': round(null_pct, 2),
            'relevance_score': round(relevance_score, 2),
            'role': role,
            'relevant': relevance_score >= DashboardConfig.RELEVANCE_THRESHOLD,
            'reasoning': reasoning,
            'llm_reasoning': llm_reasoning
        }
    
    def _get_column_type(self, series: pd.Series) -> str:
        """Determine column type"""
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        elif pd.api.types.is_bool_dtype(series):
            return 'boolean'
        else:
            return 'categorical'
    
    def _calculate_relevance(self, col_name: str, dtype: str, unique_count: int, 
                            unique_pct: float, null_pct: float, total_rows: int) -> Tuple[float, str, str]:
        """
        Calculate relevance score based on business logic
        
        Returns: (score, role, reasoning)
        """
        score = 0.0
        role = "Unknown"
        reasoning = ""
        
        # HIGH RELEVANCE COLUMNS
        
        # 1. Transaction/Order IDs - NOT needed for dashboards (too granular)
        if 'id' in col_name.lower() and 'transaction' in col_name.lower():
            score = 0.2
            role = "Transaction ID (too granular)"
            reasoning = "Transaction IDs are too granular for dashboard aggregations"
        
        # 2. Customer IDs - KEEP (for customer analysis)
        elif 'customer' in col_name.lower() and 'id' in col_name.lower():
            score = 0.9
            role = "Customer Dimension"
            reasoning = "Essential for customer segmentation and analysis"
        
        # 3. Numeric KPIs (Price, Quantity, Total, Revenue, etc.)
        elif dtype == 'numeric' and any(kw in col_name.lower() for kw in ['price', 'quantity', 'total', 'amount', 'revenue', 'cost', 'profit']):
            score = 1.0
            role = "KPI/Metric"
            reasoning = "Core business metric for calculations and aggregations"
        
        # 4. Date/Time - CRITICAL for time-series analysis
        elif dtype == 'datetime':
            score = 1.0
            role = "Time Dimension"
            reasoning = "Essential for time-based analysis and trends"
        
        # 5. Category/Segment columns - KEEP (for slicers/filters)
        elif dtype == 'categorical' and DashboardConfig.MIN_UNIQUE_FOR_CATEGORICAL <= unique_count <= DashboardConfig.MAX_UNIQUE_FOR_CATEGORICAL:
            score = 0.95
            role = "Dimension/Slicer"
            reasoning = f"Good cardinality ({unique_count} values) for filtering and grouping"
        
        # 6. Boolean flags (discount applied, premium customer, etc.)
        elif dtype == 'boolean':
            score = 0.85
            role = "Boolean Filter"
            reasoning = "Useful for binary filtering and conditional analysis"
        
        # 7. Product/Item IDs - KEEP if reasonable cardinality
        elif 'item' in col_name.lower() or 'product' in col_name.lower():
            if unique_count <= 100:
                score = 0.9
                role = "Product Dimension"
                reasoning = f"Product column with {unique_count} items - good for analysis"
            else:
                score = 0.5
                role = "Product Dimension (high cardinality)"
                reasoning = f"{unique_count} unique products - may need aggregation"
        
        # 8. Location/Region - KEEP
        elif 'location' in col_name.lower() or 'region' in col_name.lower() or 'store' in col_name.lower():
            score = 0.9
            role = "Location Dimension"
            reasoning = "Geographic analysis dimension"
        
        # LOW RELEVANCE COLUMNS
        
        # 9. Too many nulls
        elif null_pct > 50:
            score = 0.1
            role = "High Nulls"
            reasoning = f"{null_pct:.1f}% missing data - unreliable"
        
        # 10. Single value columns (no variance)
        elif unique_count == 1:
            score = 0.0
            role = "Constant Value"
            reasoning = "Only one value - no analytical value"
        
        # 11. Too many unique values for categorical
        elif dtype == 'categorical' and unique_count > DashboardConfig.MAX_UNIQUE_FOR_CATEGORICAL:
            score = 0.3
            role = "High Cardinality Text"
            reasoning = f"{unique_count} unique values - too granular for grouping"
        
        # 12. Unique identifier (like row numbers)
        elif unique_pct > 98:
            score = 0.2
            role = "Unique Identifier"
            reasoning = "Unique per row - not useful for aggregation"
        
        # Default scoring for other columns
        else:
            if dtype == 'numeric':
                score = 0.7
                role = "Numeric Column"
                reasoning = "Numeric data - potential for calculations"
            elif dtype == 'categorical':
                score = 0.6
                role = "Categorical Column"
                reasoning = "May be useful for grouping"
            else:
                score = 0.4
                role = "Other"
                reasoning = "Uncertain utility"
        
        return score, role, reasoning
    
    def _get_llm_opinion(self, col_name: str, dtype: str, role: str, reasoning: str, sample_values: List) -> str:
        """Get LLM's opinion on column relevance"""
        context = f"""
Column: {col_name}
Type: {dtype}
Current Assessment: {role}
Reasoning: {reasoning}
Sample Values: {sample_values}
"""
        
        prompt = """
Is this column relevant for a sales/business dashboard? Consider:
- Will analysts use this for filtering, grouping, or calculations?
- Does it provide business insights?
- Is it actionable?

Respond in ONE sentence with your assessment.
"""
        
        return self.query_llm(prompt, context)


# ============================================================================
# AGENT 2: MEASURE CREATION AGENT
# ============================================================================

class MeasureCreationAgent(DashboardAgent):
    """
    Creates calculated measures like Power BI
    
    Creates:
    - Revenue metrics
    - Growth rates
    - Averages and aggregations
    - Period comparisons
    - Business KPIs
    """
    
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="MeasureCreationAgent",
            role="Automated Measure & KPI Creator",
            gemini_api_key=gemini_api_key
        )
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            }
        )
    
    def identify_potential_measures(self, df: pl.DataFrame, kept_columns: List[str]) -> List[Dict]:
        """
        Identify potential calculated measures that could be created
        
        Returns list of measure definitions
        """
        print(f"\n[{self.name}] Analyzing potential calculated measures...")
        
        df_pd = df.select(kept_columns).to_pandas()
        measures = []
        
        # Detect column types
        numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = df_pd.select_dtypes(include=['datetime64']).columns.tolist()
        categorical_cols = df_pd.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Revenue/Sales Measures
        measures.extend(self._create_revenue_measures(df_pd, numeric_cols))
        
        # 2. Time-based Measures
        if date_cols:
            measures.extend(self._create_time_measures(df_pd, date_cols))
        
        # 3. Category/Segment Measures
        measures.extend(self._create_category_measures(df_pd, categorical_cols, numeric_cols))
        
        # 4. Customer Measures
        measures.extend(self._create_customer_measures(df_pd, categorical_cols, numeric_cols))
        
        # 5. Statistical Measures
        measures.extend(self._create_statistical_measures(df_pd, numeric_cols))
        
        print(f"✓ Identified {len(measures)} potential measures")
        
        return measures
    
    def _create_revenue_measures(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict]:
        """Create revenue-related measures"""
        measures = []
        
        # Find total/amount columns
        total_cols = [c for c in numeric_cols if any(kw in c.lower() for kw in ['total', 'amount', 'revenue', 'sales'])]
        price_cols = [c for c in numeric_cols if 'price' in c.lower()]
        qty_cols = [c for c in numeric_cols if 'quantity' in c.lower() or 'qty' in c.lower()]
        
        if total_cols:
            total_col = total_cols[0]
            
            # Total Revenue
            measures.append({
                'name': 'Total_Revenue',
                'formula': f'SUM({total_col})',
                'description': f'Sum of all {total_col}',
                'type': 'aggregation',
                'business_value': 'Core KPI for revenue tracking',
                'category': 'Revenue'
            })
            
            # Average Transaction Value
            measures.append({
                'name': 'Avg_Transaction_Value',
                'formula': f'AVERAGE({total_col})',
                'description': f'Average {total_col} per transaction',
                'type': 'aggregation',
                'business_value': 'Measures average basket size',
                'category': 'Revenue'
            })
        
        if price_cols and qty_cols:
            # Revenue per Unit
            measures.append({
                'name': 'Revenue_Per_Unit',
                'formula': f'{total_cols[0]} / {qty_cols[0]}',
                'description': 'Revenue divided by quantity sold',
                'type': 'calculated',
                'business_value': 'Shows effective price per item',
                'category': 'Revenue'
            })
        
        return measures
    
    def _create_time_measures(self, df: pd.DataFrame, date_cols: List[str]) -> List[Dict]:
        """Create time-based measures"""
        measures = []
        date_col = date_cols[0]
        
        measures.extend([
            {
                'name': 'Year',
                'formula': f'YEAR({date_col})',
                'description': 'Extract year from date',
                'type': 'datetime',
                'business_value': 'Enables year-over-year analysis',
                'category': 'Time'
            },
            {
                'name': 'Month',
                'formula': f'MONTH({date_col})',
                'description': 'Extract month from date',
                'type': 'datetime',
                'business_value': 'Enables monthly trends',
                'category': 'Time'
            },
            {
                'name': 'Quarter',
                'formula': f'QUARTER({date_col})',
                'description': 'Extract quarter from date',
                'type': 'datetime',
                'business_value': 'Quarterly business reporting',
                'category': 'Time'
            },
            {
                'name': 'Day_of_Week',
                'formula': f'DAYOFWEEK({date_col})',
                'description': 'Extract day of week',
                'type': 'datetime',
                'business_value': 'Analyze weekly patterns',
                'category': 'Time'
            },
            {
                'name': 'Month_Name',
                'formula': f'MONTHNAME({date_col})',
                'description': 'Month name (Jan, Feb, etc.)',
                'type': 'datetime',
                'business_value': 'User-friendly month display',
                'category': 'Time'
            }
        ])
        
        return measures
    
    def _create_category_measures(self, df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> List[Dict]:
        """Create category-based measures"""
        measures = []
        
        # Find category column
        category_cols = [c for c in categorical_cols if 'category' in c.lower()]
        
        if category_cols and numeric_cols:
            cat_col = category_cols[0]
            
            measures.append({
                'name': 'Category_Count',
                'formula': f'COUNT(DISTINCT {cat_col})',
                'description': 'Number of distinct categories',
                'type': 'aggregation',
                'business_value': 'Product portfolio diversity',
                'category': 'Category'
            })
        
        return measures
    
    def _create_customer_measures(self, df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> List[Dict]:
        """Create customer-related measures"""
        measures = []
        
        customer_cols = [c for c in categorical_cols if 'customer' in c.lower()]
        
        if customer_cols:
            cust_col = customer_cols[0]
            
            measures.extend([
                {
                    'name': 'Total_Customers',
                    'formula': f'COUNT(DISTINCT {cust_col})',
                    'description': 'Number of unique customers',
                    'type': 'aggregation',
                    'business_value': 'Customer base size',
                    'category': 'Customer'
                },
                {
                    'name': 'Transactions_Per_Customer',
                    'formula': f'COUNT(*) / COUNT(DISTINCT {cust_col})',
                    'description': 'Average transactions per customer',
                    'type': 'calculated',
                    'business_value': 'Customer engagement frequency',
                    'category': 'Customer'
                }
            ])
        
        return measures
    
    def _create_statistical_measures(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict]:
        """Create statistical measures"""
        measures = []
        
        qty_cols = [c for c in numeric_cols if 'quantity' in c.lower() or 'qty' in c.lower()]
        
        if qty_cols:
            qty_col = qty_cols[0]
            
            measures.extend([
                {
                    'name': 'Total_Units_Sold',
                    'formula': f'SUM({qty_col})',
                    'description': f'Total {qty_col} sold',
                    'type': 'aggregation',
                    'business_value': 'Volume metric',
                    'category': 'Volume'
                },
                {
                    'name': 'Avg_Units_Per_Transaction',
                    'formula': f'AVERAGE({qty_col})',
                    'description': f'Average {qty_col} per transaction',
                    'type': 'aggregation',
                    'business_value': 'Basket size metric',
                    'category': 'Volume'
                }
            ])
        
        return measures
    
    def create_measures(self, df: pd.DataFrame, measures_to_create: List[Dict]) -> pd.DataFrame:
        """
        Actually create the calculated columns in the dataframe
        """
        print(f"\n[{self.name}] Creating {len(measures_to_create)} measures...")
        
        for measure in measures_to_create:
            try:
                measure_name = measure['name']
                formula = measure['formula']
                measure_type = measure['type']
                
                if measure_type == 'datetime':
                    # Extract date parts
                    if 'YEAR' in formula:
                        col = formula.split('(')[1].split(')')[0]
                        df[measure_name] = pd.to_datetime(df[col]).dt.year
                    elif 'MONTH' in formula:
                        col = formula.split('(')[1].split(')')[0]
                        df[measure_name] = pd.to_datetime(df[col]).dt.month
                    elif 'QUARTER' in formula:
                        col = formula.split('(')[1].split(')')[0]
                        df[measure_name] = pd.to_datetime(df[col]).dt.quarter
                    elif 'DAYOFWEEK' in formula:
                        col = formula.split('(')[1].split(')')[0]
                        df[measure_name] = pd.to_datetime(df[col]).dt.dayofweek
                    elif 'MONTHNAME' in formula:
                        col = formula.split('(')[1].split(')')[0]
                        df[measure_name] = pd.to_datetime(df[col]).dt.strftime('%B')
                
                elif measure_type == 'calculated':
                    # Parse and execute calculation
                    if '/' in formula:
                        parts = formula.split('/')
                        col1 = parts[0].strip()
                        col2 = parts[1].strip()
                        df[measure_name] = (df[col1] / df[col2]).round(2)
                
                print(f"  ✓ Created: {measure_name}")
                
            except Exception as e:
                print(f"  ✗ Failed to create {measure['name']}: {str(e)}")
        
        return df


# ============================================================================
# AGENT 3: INTERACTIVE COORDINATOR
# ============================================================================

class InteractiveCoordinator(DashboardAgent):
    """
    Manages user interaction for column selection and measure creation
    """
    def __init__(self, gemini_api_key: str):
        super().__init__(
            name="InteractiveCoordinator",
            role="User Interaction Manager",
            gemini_api_key=gemini_api_key
        )
        self.relevance_agent = ColumnRelevanceAgent(gemini_api_key)
        self.measure_agent = MeasureCreationAgent(gemini_api_key)
    
    def prepare_dashboard_dataset(self, cleaned_file: str, output_file: str = "dashboard_ready_dataset.csv") -> Dict:
        """
        Main orchestration - analyzes, asks user, prepares dashboard dataset
        """
        print("\n" + "="*70)
        print("DATAAGENT PHASE 3 - DASHBOARD PREPARATION")
        print("="*70)
        
        # STEP 0: Get user's business requirements
        business_requirements = self._get_business_requirements()
        
        # Load cleaned dataset
        print(f"\nLoading cleaned dataset: {cleaned_file}")
        df = pl.read_csv(cleaned_file)
        print(f"✓ Loaded: {len(df)} rows × {len(df.columns)} columns")
        
        # Step 1: Analyze column relevance (now considers business requirements)
        column_analysis = self.relevance_agent.analyze_columns(df)
        
        # Apply business requirements to column analysis
        column_analysis = self._apply_business_requirements_to_columns(
            column_analysis, business_requirements
        )
        
        # Step 2: Ask user about columns to keep/drop
        kept_columns = self._ask_user_column_selection(column_analysis, business_requirements)
        
        # Step 3: Identify potential measures (based on business requirements)
        potential_measures = self.measure_agent.identify_potential_measures(df, kept_columns)
        
        # Add custom measures from business requirements
        custom_measures = self._create_custom_measures_from_requirements(
            df, kept_columns, business_requirements
        )
        potential_measures.extend(custom_measures)
        
        # Step 4: Ask user which measures to create
        selected_measures = self._ask_user_measure_selection(potential_measures)
        
        # Step 5: Create final dataset
        print("\n" + "="*70)
        print("CREATING FINAL DASHBOARD DATASET")
        print("="*70)
        
        # Filter columns
        df_dashboard = df.select(kept_columns)
        df_dashboard_pd = df_dashboard.to_pandas()
        
        # Create measures
        if selected_measures:
            df_dashboard_pd = self.measure_agent.create_measures(df_dashboard_pd, selected_measures)
        
        # Convert back to polars and export
        df_final = pl.from_pandas(df_dashboard_pd)
        df_final.write_csv(output_file)
        
        print(f"\n✓ Dashboard-ready dataset saved: {output_file}")
        print(f"  Columns: {len(df_final.columns)}")
        print(f"  Rows: {len(df_final)}")
        print(f"  Original columns: {len(df.columns)}")
        print(f"  Dropped columns: {len(df.columns) - len(kept_columns)}")
        print(f"  Created measures: {len(selected_measures)}")
        
        # Generate report
        report = {
            'input_file': cleaned_file,
            'output_file': output_file,
            'business_requirements': business_requirements,
            'columns_analyzed': len(column_analysis),
            'columns_kept': len(kept_columns),
            'columns_dropped': len(df.columns) - len(kept_columns),
            'measures_created': len(selected_measures),
            'final_columns': len(df_final.columns),
            'kept_columns_list': kept_columns,
            'dropped_columns_list': [col for col in df.columns if col not in kept_columns],
            'measures_list': selected_measures
        }
        
        # Save report
        with open("dashboard_prep_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved: dashboard_prep_report.json")
        
        return report
    
    def _get_business_requirements(self) -> Dict:
        """
        Get business requirements from user
        This guides what columns to keep and what measures to create
        """
        print("\n" + "="*70)
        print("BUSINESS REQUIREMENTS")
        print("="*70)
        print("\nLet's understand what you want to analyze in your dashboard.")
        print("This will help us keep only relevant columns and create useful measures.\n")
        
        requirements = {
            'analysis_focus': [],
            'key_metrics': [],
            'time_analysis': False,
            'customer_analysis': False,
            'product_analysis': False,
            'location_analysis': False,
            'custom_requirements': ""
        }
        
        # 1. Analysis Focus
        print("What is your PRIMARY analysis goal? (select one)")
        print("  1. Sales Performance (revenue, growth, trends)")
        print("  2. Customer Behavior (who buys, frequency, preferences)")
        print("  3. Product Performance (best sellers, categories)")
        print("  4. Operational Efficiency (transactions, volumes)")
        print("  5. Custom (describe your goal)")
        
        focus = input("\nEnter choice (1-5): ").strip()
        focus_map = {
            '1': 'sales_performance',
            '2': 'customer_behavior',
            '3': 'product_performance',
            '4': 'operational_efficiency',
            '5': 'custom'
        }
        requirements['analysis_focus'].append(focus_map.get(focus, 'sales_performance'))
        
        if focus == '5':
            requirements['custom_requirements'] = input("Describe your analysis goal: ").strip()
        
        # 2. Key Metrics
        print("\nWhat KEY METRICS do you need? (comma-separated, or press Enter for all)")
        print("Examples: total revenue, average order value, customer count, units sold")
        metrics_input = input("Key metrics: ").strip()
        
        if metrics_input:
            requirements['key_metrics'] = [m.strip() for m in metrics_input.split(',')]
        else:
            requirements['key_metrics'] = ['all']
        
        # 3. Dimensional Analysis
        print("\nDo you need TIME-BASED analysis? (year, month, quarter trends)")
        requirements['time_analysis'] = input("(y/n): ").strip().lower() == 'y'
        
        print("Do you need CUSTOMER analysis? (customer segments, behavior)")
        requirements['customer_analysis'] = input("(y/n): ").strip().lower() == 'y'
        
        print("Do you need PRODUCT/CATEGORY analysis? (best sellers, categories)")
        requirements['product_analysis'] = input("(y/n): ").strip().lower() == 'y'
        
        print("Do you need LOCATION/GEOGRAPHIC analysis? (store performance, regions)")
        requirements['location_analysis'] = input("(y/n): ").strip().lower() == 'y'
        
        # 4. Additional Requirements
        print("\nAny ADDITIONAL requirements? (optional)")
        additional = input("Additional needs: ").strip()
        if additional:
            requirements['custom_requirements'] += " " + additional
        
        print("\n✓ Business requirements captured")
        print(f"  Focus: {requirements['analysis_focus']}")
        print(f"  Time Analysis: {requirements['time_analysis']}")
        print(f"  Customer Analysis: {requirements['customer_analysis']}")
        print(f"  Product Analysis: {requirements['product_analysis']}")
        print(f"  Location Analysis: {requirements['location_analysis']}")
        
        return requirements
    
    def _apply_business_requirements_to_columns(
        self, column_analysis: Dict, requirements: Dict
    ) -> Dict:
        """
        Adjust column relevance based on business requirements
        """
        for col, info in column_analysis.items():
            # Boost score if column aligns with requirements
            boost = 0.0
            
            # Customer columns
            if requirements['customer_analysis'] and 'customer' in col.lower():
                boost += 0.2
                info['reasoning'] += " | USER NEEDS: Customer analysis required"
            
            # Product columns
            if requirements['product_analysis'] and ('product' in col.lower() or 'item' in col.lower() or 'category' in col.lower()):
                boost += 0.2
                info['reasoning'] += " | USER NEEDS: Product analysis required"
            
            # Location columns
            if requirements['location_analysis'] and 'location' in col.lower():
                boost += 0.2
                info['reasoning'] += " | USER NEEDS: Location analysis required"
            
            # Date columns
            if requirements['time_analysis'] and 'date' in col.lower():
                boost += 0.2
                info['reasoning'] += " | USER NEEDS: Time-based analysis required"
            
            # Apply boost
            if boost > 0:
                info['relevance_score'] = min(1.0, info['relevance_score'] + boost)
                info['relevant'] = info['relevance_score'] >= DashboardConfig.RELEVANCE_THRESHOLD
        
        return column_analysis
    
    def _create_custom_measures_from_requirements(
        self, df: pl.DataFrame, kept_columns: List[str], requirements: Dict
    ) -> List[Dict]:
        """
        Create measures based on user's specific business requirements
        """
        custom_measures = []
        df_pd = df.select(kept_columns).to_pandas()
        
        # If user specified custom metrics
        if requirements['key_metrics'] and requirements['key_metrics'] != ['all']:
            for metric in requirements['key_metrics']:
                metric_lower = metric.lower()
                
                # Try to match to existing columns
                if 'revenue' in metric_lower or 'sales' in metric_lower:
                    total_cols = [c for c in kept_columns if 'total' in c.lower() or 'amount' in c.lower()]
                    if total_cols:
                        custom_measures.append({
                            'name': f'User_Requested_{metric.replace(" ", "_")}',
                            'formula': f'SUM({total_cols[0]})',
                            'description': f'User requested: {metric}',
                            'type': 'aggregation',
                            'business_value': f'User-defined KPI: {metric}',
                            'category': 'Custom'
                        })
        
        return custom_measures
    
    def _ask_user_column_selection(self, column_analysis: Dict, requirements: Dict) -> List[str]:
        """Ask user which columns to keep/drop (with business context)"""
        print("\n" + "="*70)
        print("COLUMN SELECTION (Based on Your Business Requirements)")
        print("="*70)
        
        # Separate relevant and irrelevant
        relevant = {k: v for k, v in column_analysis.items() if v['relevant']}
        irrelevant = {k: v for k, v in column_analysis.items() if not v['relevant']}
        
        print(f"\n✓ RECOMMENDED TO KEEP ({len(relevant)} columns):")
        print("   (These align with your business requirements)")
        for col, info in relevant.items():
            print(f"  • {col:30s} | {info['role']:25s} | Score: {info['relevance_score']:.2f}")
            print(f"    Reason: {info['reasoning']}")
        
        print(f"\n✗ RECOMMENDED TO DROP ({len(irrelevant)} columns):")
        print("   (These don't align with your analysis needs)")
        for col, info in irrelevant.items():
            print(f"  • {col:30s} | {info['role']:25s} | Score: {info['relevance_score']:.2f}")
            print(f"    Reason: {info['reasoning']}")
        
        # Ask user
        print("\n" + "-"*70)
        print("Would you like to:")
        print("  1. Accept all recommendations (based on your requirements)")
        print("  2. Keep all columns (no drops)")
        print("  3. Customize selection")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            kept = list(relevant.keys())
            print(f"\n✓ Accepting recommendations: keeping {len(kept)} columns")
        
        elif choice == '2':
            kept = list(column_analysis.keys())
            print(f"\n✓ Keeping all {len(kept)} columns")
        
        elif choice == '3':
            kept = []
            print("\nFor each column, type 'k' to KEEP or 'd' to DROP:")
            for col, info in column_analysis.items():
                default = "KEEP" if info['relevant'] else "DROP"
                decision = input(f"  {col:30s} (recommended: {default}): ").strip().lower()
                if decision == 'k' or (decision == '' and default == "KEEP"):
                    kept.append(col)
            print(f"\n✓ Selected {len(kept)} columns to keep")
        
        else:
            print("\n⚠ Invalid choice. Accepting recommendations by default.")
            kept = list(relevant.keys())
        
        return kept
    
    def _ask_user_measure_selection(self, potential_measures: List[Dict]) -> List[Dict]:
        """Ask user which measures to create"""
        print("\n" + "="*70)
        print("MEASURE CREATION")
        print("="*70)
        
        print(f"\nFound {len(potential_measures)} potential measures:\n")
        
        for i, measure in enumerate(potential_measures, 1):
            print(f"{i}. {measure['name']}")
            print(f"   Formula: {measure['formula']}")
            print(f"   Description: {measure['description']}")
            print(f"   Business Value: {measure['business_value']}")
            print()
        
        print("-"*70)
        print("Would you like to:")
        print("  1. Create all measures")
        print("  2. Create none (skip measure creation)")
        print("  3. Select specific measures")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            selected = potential_measures
            print(f"\n✓ Creating all {len(selected)} measures")
        
        elif choice == '2':
            selected = []
            print("\n✓ Skipping measure creation")
        
        elif choice == '3':
            print("\nEnter measure numbers to create (comma-separated, e.g., 1,3,5):")
            indices_str = input("Measures to create: ").strip()
            
            try:
                indices = [int(x.strip()) for x in indices_str.split(',')]
                selected = [potential_measures[i-1] for i in indices if 1 <= i <= len(potential_measures)]
                print(f"\n✓ Selected {len(selected)} measures to create")
            except:
                print("\n⚠ Invalid input. Creating all measures by default.")
                selected = potential_measures
        
        else:
            print("\n⚠ Invalid choice. Creating all measures by default.")
            selected = potential_measures
        
        return selected


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for Dashboard Preparation
    
    USAGE:
    1. Ensure you have a cleaned dataset (from Phase 2)
    2. Run: python dashboard_prep.py
    3. Follow interactive prompts
    """
    
    print("\n" + "="*70)
    print("DATAAGENT PHASE 3 - DASHBOARD PREPARATION")
    print("Interactive Column Selection & Measure Creation")
    print("="*70)
    
    # Load API key
    try:
        api_key = DashboardConfig.load_api_key()
        print("✓ API key loaded")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return
    
    # Get input file
    print("\nEnter the path to your CLEANED dataset:")
    input_file = input("File path (or press Enter for 'cleaned_retail_store_sales_dataset.csv'): ").strip()
    
    if not input_file:
        input_file = "cleaned_retail_store_sales_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"\n✗ Error: File not found: {input_file}")
        return
    
    # Get output file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"dashboard_{base_name}.csv"
    
    # Execute dashboard preparation
    coordinator = InteractiveCoordinator(api_key)
    
    try:
        report = coordinator.prepare_dashboard_dataset(input_file, output_file)
        
        # Final summary
        print("\n" + "="*70)
        print("DASHBOARD PREPARATION COMPLETE")
        print("="*70)
        
        print(f"\nSummary:")
        print(f"  Input: {report['input_file']}")
        print(f"  Output: {report['output_file']}")
        print(f"  Original Columns: {report['columns_analyzed']}")
        print(f"  Kept Columns: {report['columns_kept']}")
        print(f"  Dropped Columns: {report['columns_dropped']}")
        print(f"  Created Measures: {report['measures_created']}")
        print(f"  Final Columns: {report['final_columns']}")
        
        if report['dropped_columns_list']:
            print(f"\nDropped Columns:")
            for col in report['dropped_columns_list']:
                print(f"  - {col}")
        
        if report['measures_list']:
            print(f"\nCreated Measures:")
            for measure in report['measures_list']:
                print(f"  - {measure['name']}: {measure['description']}")
        
        print(f"\nYour dataset is now ready for dashboard creation!")
        print(f"Import '{output_file}' into Power BI, Tableau, or any BI tool.")
        
        print("\n" + "="*70)
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()