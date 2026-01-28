import streamlit as st
import pandas as pd
import plotly.express as px
from .data_processor import apply_dynamic_logic

def run_analysis(df, plan):
    results = []
    current_df = df.copy() 
    
    for i, task in enumerate(plan):
        try:
            with st.container(border=True):
                question = task.get('question', 'Analysis Task')
                st.subheader(question)

                # 1. Apply cleaning/binning/logic
                current_df = apply_dynamic_logic(current_df, task)

                # 2. Aggregation Setup
                agg_func = (task.get('aggregation') or 'sum').lower()
                
                # FIX: Map 'count_distinct' to 'nunique' to prevent Pandas errors
                if agg_func == 'count_distinct':
                    agg_func = 'nunique'

                x_col = task.get('x_axis')
                y_col = task.get('y_axis')
                c_type = (task.get('chart_type') or 'bar').lower()
                chart_key = f"chart_{i}_{x_col}_{y_col}"

                # --- 3. Robust Visualization Logic ---
                
                # Check if columns exist before plotting
                if x_col not in current_df.columns or y_col not in current_df.columns:
                    st.error(f"Missing columns for this analysis: {x_col} or {y_col}")
                    continue

                # SCATTER PLOT: For relationship questions
                if "scatter" in c_type or "relationship" in question.lower():
                    fig = px.scatter(current_df, x=x_col, y=y_col, 
                                   title=f"Relationship: {x_col} vs {y_col}",
                                   opacity=0.6) # Removed trendline to avoid statsmodels error
                    st.plotly_chart(fig, key=chart_key)
                
                else:
                    # AGGREGATED PLOTS: Bar, Line, Pie
                    chart_data = current_df.groupby(x_col)[y_col].agg(agg_func).reset_index()

                    if "bar" in c_type:
                        chart_data = chart_data.sort_values(by=y_col, ascending=False)
                        fig = px.bar(chart_data, x=x_col, y=y_col, color=x_col, text_auto='.2s')
                        st.plotly_chart(fig, key=chart_key)
                    
                    elif "line" in c_type:
                        chart_data = chart_data.sort_values(by=x_col)
                        fig = px.line(chart_data, x=x_col, y=y_col, markers=True)
                        st.plotly_chart(fig, key=chart_key)
                    
                    elif "pie" in c_type:
                        fig = px.pie(chart_data, names=x_col, values=y_col, hole=0.3)
                        st.plotly_chart(fig, key=chart_key)
                    
                    else:
                        st.dataframe(chart_data)
                
            results.append(f"Successfully rendered {question}")
            
        except Exception as e:
            st.error(f"Error executing task '{task.get('question')}': {e}")
            
    return results, current_df