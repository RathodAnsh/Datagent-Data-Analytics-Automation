import json
import os
from dotenv import load_dotenv
import google.genai as genai 

def generate_analysis_plan(summary, questions):
    """
    Generates a structured analysis plan using Gemini.
    Tuned specifically for Retail/Sales domains to handle binning,
    relationship analysis, and data type coercion.
    """
    # 1. Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in .env file."}

    # 2. Initialize the Client
    client = genai.Client(api_key=api_key)

    # Prepare a serializable version of the dataframe summary for the prompt
    serializable_summary = {
        "rows": summary["rows"],
        "columns": summary["columns"],
        "dtypes": summary["dtypes"].to_dict() if hasattr(summary["dtypes"], "to_dict") else summary["dtypes"],
        "head": summary["head"].to_dict(orient="records") if hasattr(summary["head"], "to_dict") else summary["head"]
    }

    prompt = f"""
You are an expert Senior Python Data Analyst specializing in Retail and Sales datasets.
**Task:** Analyze the provided dataset summary and create a multi-step execution plan for the user's questions.

**DATASET SUMMARY:**
{json.dumps(serializable_summary)}

**USER QUESTIONS:**
{questions}

**INSTRUCTIONS & CONSTRAINTS:**

1. **DATA CLEANING & TYPE SAFETY (CRITICAL):**
   - Always check the 'dtypes'. If a column needed for math is an 'object', include: 
     `df['col'] = pd.to_numeric(df['col'], errors='coerce')` in the `python_logic`.
   - For date-based questions, include: `df['DateCol'] = pd.to_datetime(df['DateCol'])`.

2. **HANDLING DISTRIBUTIONS:**
   - If a question asks for a "distribution", you MUST create a binning column.
   - **Logic Example:** `df['Spending_Bins'] = pd.qcut(df['Total_Price'], q=5, labels=['Low', 'Below Avg', 'Average', 'Above Avg', 'High'], duplicates='drop')`.
   - Set `x_axis` to the name of this new binning column.

3. **RELATIONSHIP QUESTIONS:**
   - If a question asks "is there a relationship" or "correlation" between two metrics, set `chart_type` to "scatter".
   - For scatter plots, set `aggregation` to null as we want to see individual data points or a raw relationship.

4. **AGGREGATION STANDARDS:**
   - Use only: 'sum', 'mean', 'count', 'nunique'.
   - **NEVER use 'count_distinct'**. Use 'nunique' instead.

5. **JSON STRUCTURE (STRICT):**
   - Return ONLY a JSON list. No preamble.
   - Ensure the `new_column` name in `derivation` matches the `x_axis` or `y_axis` exactly.

**JSON SCHEMA:**
[
  {{
    "question": "The user's question",
    "chart_type": "bar | line | pie | scatter",
    "x_axis": "column_name",
    "y_axis": "column_name",
    "aggregation": "sum | mean | count | nunique | null",
    "derivation": {{
        "new_column": "name_of_derived_column",
        "python_logic": "Executable pandas code snippets separated by ;"
    }}
  }}
]

**RETURN ONLY CLEAN JSON.**
"""

    try:
        # 3. Call the model (using 2.0-flash for high speed and reasoning)
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        
        text = response.text.strip()
        
        # 4. Clean markdown formatting if present
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        
        # 5. Parse and Return
        plan = json.loads(text)
        return plan if isinstance(plan, list) else [plan]
    
    except Exception as e:
        return {"error": f"Planner Error: {str(e)}"}