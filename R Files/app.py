import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import warnings
import shutil
import glob
from pathlib import Path
import json
import subprocess
import google.generativeai as genai
import re

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# ==================== INITIALIZE SESSION STATE ====================
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "profile_data" not in st.session_state:
    st.session_state.profile_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = None
if "dashboard_ready" not in st.session_state:
    st.session_state.dashboard_ready = False
if "last_analysis_result" not in st.session_state:
    st.session_state.last_analysis_result = None
if "last_charts_bytes" not in st.session_state:
    st.session_state.last_charts_bytes = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "last_r_code" not in st.session_state:
    st.session_state.last_r_code = None

# ==================== DIRECTORY SETUP ====================
TEMP_CHARTS_DIR = Path("temp_charts")
TEMP_DATA_DIR = Path("temp_data")
PROFILES_DIR = Path("data_profiles")

# Create directories if they don't exist
for d in [TEMP_CHARTS_DIR, TEMP_DATA_DIR, PROFILES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SOURCE_DATA_PATH = (TEMP_DATA_DIR / "source.csv").resolve()

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="AI Data Analyst Agent (R)", layout="wide")
st.title("🤖 AI Data Analyst Agent (R Edition)")
st.markdown("""
**System Workflow:**
1. **Profile:** Auto-cleans data (Smart Detection) & saves JSON profile.
2. **Analysis:** AI writes robust R code to generate Insights & Charts.
3. **Dual Output:** Generates **Formatted Text Report** & **Formula-Based Dashboard**.
""")

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.header("⚙️ Configuration")

default_api_key = st.session_state.api_key_saved or os.getenv("GOOGLE_API_KEY", "")
api_key = st.sidebar.text_input(
    "Google API Key",
    value=default_api_key,
    type="password",
    help="Enter your Google Gemini API key"
)
st.session_state.api_key_saved = api_key

if not api_key:
    st.error("❌ Please provide a Google API Key")
    st.stop()

genai.configure(api_key=api_key)

# ==================== DATA PROFILING FUNCTIONS ====================

def sanitize_column_names(df):
    """Convert column names to snake_case for consistency."""
    df.columns = [
        re.sub(r'[^a-zA-Z0-9]', '_', col.strip().lower()).strip('_') 
        for col in df.columns
    ]
    return df

def detect_and_convert_dtypes(df):
    """
    SMART Data Type Detection.
    Prioritizes Dates > Currency > Numerics > Strings.
    Does NOT blindly strip characters from Addresses/Names.
    """
    type_conversions_log = []
    
    for col in df.columns:
        # 1. Skip if already proper numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Get non-null samples for analysis
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue
        
        sample = non_null.head(50) # Look at first 50 rows
        
        # 2. CHECK FOR DATES FIRST
        try:
            # dayfirst=False is generally safer for mixed formats, but can adjust if needed
            temp_dates = pd.to_datetime(sample, dayfirst=False, errors='coerce') 
            valid_dates_count = temp_dates.notna().sum()
            
            # If > 60% are valid dates, convert the whole column
            if valid_dates_count / len(sample) > 0.6:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                type_conversions_log.append(f"📅 Converted '{col}' to datetime")
                continue 
        except:
            pass

        # 3. CHECK FOR CURRENCY / DIRTY NUMBERS
        # We only clean if it explicitly looks like money (has $ or digits/commas)
        def is_cleanable_number(val):
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[$,€£ ]', '', val)
            # Remove ONE decimal point to check if the rest are digits
            cleaned = cleaned.replace('.', '', 1)
            return cleaned.isdigit()

        cleanable_count = sample.apply(is_cleanable_number).sum()
        
        # If > 80% look like money/numbers, clean and convert
        if cleanable_count / len(sample) > 0.8:
            try:
                df[col] = df[col].astype(str).str.replace(r'[$,€£]', '', regex=True).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                type_conversions_log.append(f"📊 Converted '{col}' to numeric")
                continue
            except:
                pass
        
        # 4. Fallback: It stays Object (Text)
            
    return df, type_conversions_log

def save_data_profiling(df, filename="data_profiling"):
    """
    Save comprehensive data profiling to a JSON file.
    """
    from datetime import datetime
    
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{filename}.json"
    
    if profile_path.exists():
        try: profile_path.unlink()
        except: pass
            
    profile_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_info": {},
        "sample_data": df.head(5).astype(str).to_dict(orient="records"),
        "statistics": {}
    }
    
    for col in df.columns:
        profile_data["column_info"][col] = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique())
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            profile_data["statistics"][col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "sum": float(df[col].sum()) if not pd.isna(df[col].sum()) else None
            }
    
    # CRITICAL: This list is what the AI uses to know exact column names
    profile_data["columns_list"] = list(df.columns)
    
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=2, default=str)
    
    return str(profile_path)

def load_and_profile_csv(uploaded_file):
    """Orchestrator for loading, cleaning, and profiling."""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
    
    initial_rows = len(df)
    
    # Standard Cleaning Pipeline
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    df = sanitize_column_names(df) 
    df, conversions = detect_and_convert_dtypes(df) 
    
    return df, initial_rows

def clear_temp_charts():
    """Wipe temp charts directory."""
    if TEMP_CHARTS_DIR.exists():
        shutil.rmtree(TEMP_CHARTS_DIR)
    TEMP_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== DEPENDENCY INLINER ====================
def inline_html_dependencies(html_path):
    """Embeds JS/CSS dependencies directly into the HTML file."""
    if not html_path.exists():
        return False
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parent_dir = html_path.parent
        script_pattern = re.compile(r'<script src="([^"]+)"></script>')
        link_pattern = re.compile(r'<link href="([^"]+)" rel="stylesheet" />')
        
        def replace_script(match):
            src = match.group(1)
            if src.startswith("http") or src.startswith("//"): return match.group(0)
            local_path = parent_dir / src
            if local_path.exists():
                try:
                    with open(local_path, 'r', encoding='utf-8') as jsf:
                        return f'<script>\n{jsf.read()}\n</script>'
                except: pass
            return match.group(0)
            
        def replace_style(match):
            href = match.group(1)
            if href.startswith("http") or href.startswith("//"): return match.group(0)
            local_path = parent_dir / href
            if local_path.exists():
                try:
                    with open(local_path, 'r', encoding='utf-8') as cssf:
                        return f'<style>\n{cssf.read()}\n</style>'
                except: pass
            return match.group(0)
            
        content = script_pattern.sub(replace_script, content)
        content = link_pattern.sub(replace_style, content)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True
    except Exception as e:
        return False

# ==================== R EXECUTION FUNCTION ====================

def run_r_script(r_code):
    """Saves R code to a file and executes it via subprocess."""
    r_file_path = "temp_analysis.R"
    with open(r_file_path, "w", encoding="utf-8") as f:
        f.write(r_code)
    
    r_exec = shutil.which("Rscript")
    if not r_exec and os.name == 'nt':
        common_paths = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True)
        if common_paths:
            r_exec = common_paths[0]
            
    if not r_exec:
        return "❌ Error: 'Rscript' not found. Please install R.", False

    try:
        # Force UTF-8 execution
        result = subprocess.run(
            [r_exec, r_file_path], 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace',
            check=False
        )
        
        # Clean R startup noise
        stderr_cleaned = []
        if result.stderr:
            for line in result.stderr.splitlines():
                if any(x in line for x in ["Attaching", "masks", "conflicts", "library", "Loading", "geom_", "Warning", "Welcome"]): continue
                if line.strip() == "": continue
                stderr_cleaned.append(line)
        
        err_msg = "\n".join(stderr_cleaned).strip()
        
        if result.returncode != 0:
            return f"❌ R Execution Error:\n{err_msg}\n\nFull Output: {result.stdout}", False
            
        return result.stdout, True
    except Exception as e:
        return f"❌ System Error: {str(e)}", False

# ==================== UI LOGIC ====================
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# --- DATA PROCESSING & PROFILING ---
if uploaded_file is not None:
    df, initial_rows = load_and_profile_csv(uploaded_file)
    
    if df is not None:
        st.session_state.dataframe = df
        df.to_csv(SOURCE_DATA_PATH, index=False)
        profile_path = save_data_profiling(df, "data_profiling")
        
        with open(profile_path, 'r') as f:
            st.session_state.profile_data = json.load(f)
        
        st.sidebar.success(f"✅ Loaded & Profiled: {df.shape[0]} rows")
        
        st.markdown("---")
        st.header("📊 PHASE 1: Dataset Profile")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("📋 Rows", f"{df.shape[0]:,}")
        with col2: st.metric("🔢 Columns", f"{df.shape[1]}")
        with col3: st.metric("⚠️ Missing", f"{df.isnull().sum().sum()}")
        with col4: st.metric("🔁 Duplicates", f"{initial_rows - len(df)}")

        with st.expander("📝 View Data Preview & Types", expanded=True):
            st.dataframe(df.head(5), use_container_width=True)
            st.write(df.dtypes.astype(str))
            
        st.markdown("---")
        st.header("💬 PHASE 2: AI Analysis & Visualization")

# --- CHAT & ANALYSIS ---
if st.session_state.dataframe is not None:
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"): st.markdown(f"**Q: {message['content']}**")
        else:
            with st.chat_message("assistant"):
                if "code" in message:
                    with st.expander("📜 View R Code"): st.code(message["code"], language="r")
                st.markdown(message["content"])
                if "charts" in message and message["charts"]:
                    st.markdown("**📊 Generated Visualizations:**")
                    for b in message["charts"]: st.image(b)

    user_query = st.chat_input("Ask a question (e.g., 'Analyze sales by region and product')")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.current_query = user_query 
        
        with st.chat_message("user"): st.markdown(f"**Q: {user_query}**")
            
        with st.spinner("🧠 Analyst is thinking & writing R code..."):
            
            clear_temp_charts()
            profile_info = json.dumps(st.session_state.profile_data, indent=2)
            r_data_path_str = str(SOURCE_DATA_PATH).replace("\\", "/")
            
            # --- UPDATED SYSTEM PROMPT WITH STRICT FORMATTING & FORMULAS ---
            prompt = f"""
You are an expert R Data Analyst.
**Task:** Analyze data, generate **Strictly Formatted Text**, **Correct Charts**, and **Formula-Based KPIs**.

**DATASET:** "{r_data_path_str}"
**USER QUESTION:** "{user_query}"
**DATA PROFILE:** {profile_info}

**INSTRUCTIONS:**

1. **INITIAL SETUP:**
   - `options(warn=-1)`
   - `options(scipen=999)`
   - Libraries: `tidyverse`, `ggplot2`, `plotly`, `htmltools`, `scales`, `lubridate`, `jsonlite`.
   - `df <- read.csv('{r_data_path_str}', stringsAsFactors=FALSE, fileEncoding="UTF-8")`

2. **DATA PREPARATION:**
   - **Date Parsing:** `df$col <- parse_date_time(df$col, orders=c("dmy", "ymd", "mdy", "Ymd", "dmY", "mdY HM"))`
   - **Filtering:** `df <- df %>% filter(!is.na(relevant_column))` before aggregation.

3. **SMART KPI CONFIGURATION (FORMULAS REQUIRED):**
   - Create `temp_charts/kpi_config.json`.
   - **CRITICAL:** Do NOT just pick a column name. Write a Pandas-compatible **FORMULA**.
   - **Logic:**
     - If asking for "Revenue" and you see `price` and `quantity` -> formula: `price * quantity`.
     - If asking for "Orders" -> formula: `order_id` (with agg: `count_distinct`).
     - If asking for simple sum -> formula: `column_name`.
   - **JSON Format:**
     ```r
     kpi_config <- list(
       slicers = c("city", "product_line"), 
       kpis = list(
         list(label = "Total Revenue", formula = "quantity_ordered * price_each", agg = "sum", fmt = "$"),
         list(label = "Total Orders", formula = "order_id", agg = "count_distinct", fmt = "num"),
         list(label = "Avg Order Value", formula = "sales", agg = "mean", fmt = "$")
       )
     )
     write_json(kpi_config, "temp_charts/kpi_config.json", auto_unbox = TRUE)
     ```

4. **TEXT INSIGHTS (STRICT FORMAT):**
   - You MUST output the text in this EXACT format (headers with dashes, bullet points):
   - **Example Output Format:**
     ```text
     --- Sales Performance Insights ---
     The best month for sales was 'April' with a total revenue of $1,841,355.
     The city that generated the highest total revenue was 'San Francisco' with $454,141.
     
     Top 5 products contributing to sales are:
     - Macbook Pro Laptop ($409,700)
     - iPhone ($272,300)
     - ThinkPad Laptop ($222,998)
     ```
   - Print this using `cat()`.

5. **VISUALIZATION LOGIC:**
   - **Aggregation:** `summarise()` data before plotting.
   - **Code Pattern:**
     ```r
     p1 <- df %>% 
       group_by(Category) %>% 
       summarise(Total = sum(Sales, na.rm=TRUE)) %>%
       ggplot(aes(x=reorder(Category, -Total), y=Total, fill=Category)) + 
       geom_col() + 
       labs(title="Title", fill="Legend") + 
       theme_minimal()
     ggsave("temp_charts/plot_1.png", p1, width=10, height=6)
     plots_list[[1]] <- p1
     ```

6. **DASHBOARD OUTPUT:**
   - `save_html(tagList(lapply(plots_list, ggplotly)), "temp_charts/interactive_dashboard.html")`

**RETURN ONLY CLEAN R CODE.** No markdown.
"""
            
            try:
                # 1. Call Gemini
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                # 2. Extract Code
                r_code = response.text
                match = re.search(r"```[rR]\n(.*?)```", r_code, re.DOTALL)
                if match: r_code = match.group(1).strip()
                elif "```" in r_code: r_code = r_code.replace("```", "").strip()
                
                st.session_state.last_r_code = r_code
                
                # 3. Execute R Code
                output_text, success = run_r_script(r_code)
                
                # 4. Handle Visuals
                generated_files = sorted(glob.glob('temp_charts/plot_*.png'))
                saved_charts_bytes = []
                for fpath in generated_files:
                    with open(fpath, "rb") as f:
                        saved_charts_bytes.append(f.read())
                
                # 5. Fix HTML
                dash_path = Path("temp_charts/interactive_dashboard.html")
                if dash_path.exists():
                    inline_html_dependencies(dash_path)
                    st.session_state.dashboard_ready = True
                
                # 6. Display Response
                with st.chat_message("assistant"):
                    with st.expander("📜 View R Analysis Logic"):
                        st.code(r_code, language="r")
                    st.markdown(output_text)
                    for b in saved_charts_bytes:
                        st.image(b, caption="Analysis Chart")
                
                # 7. Update History
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": output_text,
                    "code": r_code,
                    "charts": saved_charts_bytes
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")

else:
    st.info("👆 Upload a CSV file in the sidebar to begin.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>AI Data Analyst Agent | R Edition</div>", unsafe_allow_html=True)