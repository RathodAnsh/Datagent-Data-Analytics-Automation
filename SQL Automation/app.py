import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
from utils import get_db_connection, profile_database, check_llm_understanding, generate_queries_from_llm

# --- Setup ---
load_dotenv()
st.set_page_config(page_title="SQL Automation Tool", layout="wide")

# Initialize Session State
if 'db_creds' not in st.session_state: st.session_state.db_creds = None
if 'profiling_done' not in st.session_state: st.session_state.profiling_done = False
if 'chat_session' not in st.session_state: st.session_state.chat_session = None
if 'analysis_history' not in st.session_state: st.session_state.analysis_history = []

# Configure LLM
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("⚠️ API Key not found in .env file")
    st.stop()

# --- Sidebar: Secure Connection ---
with st.sidebar:
    st.header("🔒 Database Login")
    host = st.text_input("Host", "localhost")
    port = st.text_input("Port", "3306")
    user = st.text_input("User", "root")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database Name")
    
    if st.button("Connect"):
        creds = {'host': host, 'port': port, 'user': user, 'password': password, 'database': database}
        conn = get_db_connection(creds)
        if conn:
            st.session_state.db_creds = creds
            st.success("Connected!")
            conn.close()
        else:
            st.error("Connection Failed.")

# --- Main Interface ---
st.title("🤖 Intelligent SQL Analysis Automation")

# STEP 1: Select Tables & Profile
if st.session_state.db_creds:
    conn = get_db_connection(st.session_state.db_creds)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    all_tables = [x[0] for x in cursor.fetchall()]
    conn.close()

    if not st.session_state.profiling_done:
        st.subheader("Select Files (Tables) to Analyze")
        selected_tables = st.multiselect("Choose tables:", all_tables)

        if selected_tables and st.button("Start Data Profiling"):
            with st.spinner("Profiling Data..."):
                # 1. Profile Data to JSON
                json_schema = profile_database(st.session_state.db_creds, selected_tables)
                
                # 2. Initialize LLM Chat
                chat = model.start_chat(history=[])
                
                # 3. Send JSON to LLM and wait for "DATA_UNDERSTOOD"
                is_understood = check_llm_understanding(chat, json_schema)
                
                if is_understood:
                    st.session_state.profiling_done = True
                    st.session_state.chat_session = chat
                    st.rerun()
                else:
                    st.error("LLM failed to understand the data structure. Try again.")

# STEP 2: Analysis (Only shown after LLM confirms understanding)
if st.session_state.profiling_done:
    st.success("✅ LLM has successfully understood your data profiling! You can now ask questions.")
    
    # Input Area
    user_questions = st.text_area("Enter your analysis questions (You can ask multiple things at once):", 
                                  placeholder="Ex: Show me total sales. Also show me the top 5 customers.")
    
    if st.button("Analyze"):
        if user_questions:
            with st.spinner("LLM is processing your questions..."):
                # 1. Get List of Queries from LLM
                query_list = generate_queries_from_llm(st.session_state.chat_session, user_questions)
                
                if not query_list:
                    st.error("Could not parse questions.")
                
                # 2. Process One by One
                conn = get_db_connection(st.session_state.db_creds)
                
                for item in query_list:
                    question_text = item.get('question')
                    sql_query = item.get('sql')
                    
                    # Store in history
                    try:
                        df = pd.read_sql(sql_query, conn)
                        result = df
                        error = None
                    except Exception as e:
                        result = None
                        error = str(e)
                    
                    st.session_state.analysis_history.append({
                        "q": question_text,
                        "sql": sql_query,
                        "result": result,
                        "error": error
                    })
                
                conn.close()
                st.rerun()

    # Display History (One by One)
    st.divider()
    if st.session_state.analysis_history:
        st.subheader("Analysis Results")
        
        # Iterate through history in reverse (newest first) or normal order
        for i, entry in enumerate(reversed(st.session_state.analysis_history)):
            with st.container():
                st.markdown(f"### ❓ Question: {entry['q']}")
                st.code(entry['sql'], language="sql")
                
                if entry['error']:
                    st.error(f"Execution Error: {entry['error']}")
                else:
                    st.dataframe(entry['result'], use_container_width=True)
                
                st.markdown("---")