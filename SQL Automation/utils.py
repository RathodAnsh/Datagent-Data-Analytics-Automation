import mysql.connector
import pandas as pd
import json
import google.generativeai as genai

def get_db_connection(creds):
    """Establishes a temporary connection to MySQL."""
    try:
        conn = mysql.connector.connect(
            host=creds['host'],
            port=creds['port'],
            user=creds['user'],
            password=creds['password'],
            database=creds['database']
        )
        return conn
    except Exception as e:
        return None

def profile_database(creds, selected_tables):
    """
    Connects to DB, describes selected tables, and returns a JSON structure.
    """
    conn = get_db_connection(creds)
    if not conn:
        return None
    
    schema_profile = {}
    cursor = conn.cursor()
    
    try:
        for table in selected_tables:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            # Format: { "column_name": "data_type" }
            schema_profile[table] = {col[0]: str(col[1]) for col in columns}
    except Exception as e:
        return None
    finally:
        conn.close()
        
    return json.dumps(schema_profile, indent=2)

def check_llm_understanding(chat_session, json_schema):
    """
    Sends the schema to LLM and waits for 'DATA_UNDERSTOOD' confirmation.
    """
    system_instruction = f"""
    I am acting as a bridge between a user and a database.
    Here is the database schema in JSON format:
    {json_schema}
    
    TASK:
    1. Study the table names, column names, and data types carefully.
    2. If you understand the structure, reply with ONLY the text: "DATA_UNDERSTOOD".
    3. Do not generate SQL yet.
    """
    response = chat_session.send_message(system_instruction)
    return "DATA_UNDERSTOOD" in response.text

def generate_queries_from_llm(chat_session, user_input):
    """
    Takes multiple user questions and returns a LIST of SQL queries.
    """
    prompt = f"""
    User Input: "{user_input}"
    
    The user may have asked multiple distinct questions. 
    Your task:
    1. Break the input down into individual questions.
    2. Generate a valid MySQL query for EACH question based on the schema you learned.
    3. Return the output strictly as a JSON ARRAY of objects.
    
    REQUIRED OUTPUT FORMAT:
    [
        {{"question": "text of first question", "sql": "SELECT * FROM..."}},
        {{"question": "text of second question", "sql": "SELECT COUNT(*)..."}}
    ]
    
    Do not add markdown formatting like ```json. Just return the raw JSON array.
    """
    
    response = chat_session.send_message(prompt)
    clean_text = response.text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(clean_text)
    except:
        return []