# 7.app.py

import streamlit as st
import pandas as pd
from agents.data_validator import validate_dataset
from agents.data_profiler import profile_dataset
from agents.gemini_planner import generate_analysis_plan
from agents.analysis_executor import run_analysis
from agents.excel_dashboard import build_excel_dashboard

st.set_page_config(page_title="AI Data Agent", layout="wide")
st.title("🤖 AI Data Analysis Agent")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file:
    # 1. Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    # 2. Validate
    is_clean, msg = validate_dataset(df)
    if not is_clean:
        st.error("❌ Dataset Issues Found")
        for m in msg: st.warning(m)
        # We allow continuation but warn the user
        if not st.checkbox("Continue anyway?"):
            st.stop()

    st.success("✅ Dataset Loaded")
    summary = profile_dataset(df)

    # UI for Dataset Summary
    with st.expander("📊 View Dataset Overview", expanded=True):
        # 1. Display Rows and Columns as Metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", summary["rows"])
        col2.metric("Total Columns", summary["columns"])
        
        st.divider()

        # 2. Display Datatypes
        st.subheader("🧬 Column Data Types")
        # Converting series to a DataFrame for a cleaner table look
        dtype_series = summary["dtypes"]
        dtype_df = dtype_series.reset_index()
        dtype_df.columns = ["Column Name", "Data Type"]
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.divider()

        # 3. Display Head and Tail in Table format
        st.subheader("📑 Data Preview")
        st.write("**First 5 Rows:**")
        st.dataframe(summary["head"], use_container_width=True)
        
        st.write("**Last 5 Rows:**")
        st.dataframe(summary["tail"], use_container_width=True)

    # 3. Questions Input
    questions = st.text_area("Enter Analytical Questions (one per line)", placeholder="e.g., What are the top 5 products sold by revenue?")

    if st.button("Generate Analysis Plan"):
        with st.spinner("Brainstorming with Gemini..."):
            plan = generate_analysis_plan(summary, questions)
            
            if isinstance(plan, dict) and "error" in plan:
                st.error(f"API Error: {plan['error']}")
            elif not plan:
                st.error("The AI returned an empty plan.")
            else:
                # Save to session state so visuals persist
                st.session_state['analysis_plan'] = plan
                st.session_state['analysis_triggered'] = True

    # 4. Persistence Layer (Outside the button click)
    if st.session_state.get('analysis_triggered'):
       
        st.divider()
        with st.container():
            st.subheader("📈 Visuals & Insights")
        # Capture the results and the updated dataframe
            results, updated_df = run_analysis(df, st.session_state['analysis_plan'])
        
        # --- NEW: Updated Dataset Preview Section ---
        if updated_df is not None and len(updated_df.columns) > len(df.columns):
            st.divider()
            with st.expander("✨ View Updated Dataset (New Columns Derived)", expanded=False):
                st.info(f"New columns detected! Total columns: {len(updated_df.columns)} (Original: {len(df.columns)})")
            
            # Display Metrics
                col1, col2 = st.columns(2)
                col1.metric("Updated Rows", updated_df.shape[0])
                col2.metric("Updated Columns", updated_df.shape[1])
            
            # Display Data Types for the new columns specifically
                st.subheader("🧬 Updated Column Types")
                new_dtypes = updated_df.dtypes.astype(str).reset_index()
                new_dtypes.columns = ["Column Name", "Data Type"]
                st.dataframe(new_dtypes, use_container_width=True, hide_index=True)
            
            # Display Head and Tail
                st.subheader("📑 Updated Data Preview")
                st.write("**First 5 Rows (Updated):**")
                st.dataframe(updated_df.head(5), use_container_width=True)
            
                st.write("**Last 5 Rows (Updated):**")
                st.dataframe(updated_df.tail(5), use_container_width=True)
        # --------------------------------------------
        st.divider()
        st.subheader("🧠 Current Analysis Plan")
        with st.expander("Show JSON Plan"):
            st.json(st.session_state['analysis_plan'])

        
        
        # 5. Export Section
        st.divider()
        st.subheader("💾 Export Report")
        if st.button("Prepare Excel Dashboard"):
            with st.spinner("Formatting Excel sheets and charts..."):
                file_path = build_excel_dashboard(df, st.session_state['analysis_plan'])
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="Download Excel Dashboard",
                        data=f,
                        file_name="AI_Data_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )