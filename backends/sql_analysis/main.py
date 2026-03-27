from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import mysql.connector
import pandas as pd
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
app = FastAPI(title="Retail SQL Automation API")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Next.js domain (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("⚠️ API Key not found in .env file")

# --- Pydantic Models for Next.js Communication ---
class DBCreds(BaseModel):
    host: str
    port: str
    user: str
    password: str
    database: str

class ProfileRequest(BaseModel):
    creds: DBCreds
    tables: List[str]

class AnalyzeRequest(BaseModel):
    creds: DBCreds
    schema_json: str
    question: str

# --- Helper Functions ---
def get_db_connection(creds: DBCreds):
    try:
        return mysql.connector.connect(
            host=creds.host, 
            port=creds.port, 
            user=creds.user,
            password=creds.password, 
            database=creds.database
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {str(e)}")

# --- API Endpoints ---

@app.post("/api/connect")
async def connect_db(creds: DBCreds):
    """Tests connection and returns available tables from the database."""
    conn = get_db_connection(creds)
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [x[0] for x in cursor.fetchall()]
        return {"status": "success", "tables": tables}
    finally:
        conn.close()

@app.post("/api/profile")
async def profile_db(req: ProfileRequest):
    """Extracts the schema ONLY for the tables the user selected."""
    conn = get_db_connection(req.creds)
    schema_profile = {}
    try:
        cursor = conn.cursor()
        for table in req.tables:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            # Format: { "column_name": "data_type" }
            schema_profile[table] = {col[0]: str(col[1]) for col in columns}
        return {"schema_json": json.dumps(schema_profile, indent=2)}
    finally:
        conn.close()

@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest):
    """
    1. Generates highly optimized SQL queries based on user questions.
    2. Executes them one by one.
    3. Runs Explainable AI on the results of each query.
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API Key is missing on the backend.")

    # 1. Ask AI to generate optimized SQL for all questions
    prompt_sql = f"""
    I am querying a MySQL database for the RETAIL SALES domain.
    Schema: {req.schema_json}
    
    RETAIL BUSINESS CONTEXT:
    - "Revenue" or "Sales" = total amount or (price * quantity).
    - "Best Sellers" = sort by highest revenue or highest quantity.
    - Write HIGHLY EFFICIENT, optimized MySQL queries. Use AS aliases for readable columns.
    - Always use proper aggregations (SUM, COUNT) and GROUP BY clauses where needed.
    
    User Question(s): "{req.question}"
    
    TASK: Break the user's input into individual questions. Generate an optimized SQL query for each.
    Return ONLY a raw JSON array. No markdown, no backticks.
    Format: [{{"question": "text of question", "sql": "SELECT ..."}}]
    """
    
    try:
        resp_sql = model.generate_content(prompt_sql)
        clean_text = resp_sql.text.replace("```json", "").replace("```", "").strip()
        queries = json.loads(clean_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse questions into SQL: {str(e)}\n\nAI Response: {resp_sql.text if 'resp_sql' in locals() else 'None'}")

    results = []
    conn = get_db_connection(req.creds)
    
    # 2. Execute SQL & Generate Explainable AI insight ONE BY ONE
    try:
        for item in queries:
            q_text = item.get("question", req.question)
            sql = item.get("sql", "")
            
            try:
                # Run the optimized SQL query
                df = pd.read_sql(sql, conn)
                
                # Convert DataFrame to a list of dictionaries for the Next.js frontend
                data_records = df.to_dict(orient="records")
                columns = list(df.columns)
                error = None
                
                # 3. Explainable AI: Generate insight based on the table output
                if len(df) == 0:
                    explanation = "No data found for this specific query."
                else:
                    # Send only the first 5 rows to AI so we don't exceed token limits
                    if len(df) <= 5:
                        data_str = df.to_string(index=False)
                    else:
                        data_str = f"First 5 rows:\n{df.head(5).to_string(index=False)}\n(Total rows: {len(df)})"
                        
                    prompt_insight = f"""
                    You are an expert Retail Data Analyst.
                    Question: "{q_text}"
                    SQL Executed: "{sql}"
                    
                    Data Results:
                    {data_str}
                    
                    Provide a VERY SHORT, 1-2 sentence business explanation of these results. 
                    Do NOT explain how the SQL works. Just give the direct business insight.
                    """
                    resp_insight = model.generate_content(prompt_insight)
                    explanation = resp_insight.text.strip()
                
            except Exception as e:
                data_records, columns, explanation = [], [], None
                error = str(e)
                
            # Append the packaged result for this specific question
            results.append({
                "question": q_text,
                "sql": sql,
                "columns": columns,
                "data": data_records,
                "explanation": explanation,
                "error": error
            })
            
        return {"results": results}
    finally:
        conn.close()