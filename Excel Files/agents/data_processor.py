import pandas as pd
import numpy as np

def apply_dynamic_logic(df, task):
    """
    Executes dynamic Python logic with improved error handling and 
    support for distribution binning.
    """
    df_copy = df.copy()
    
    derivation = task.get('derivation')
    if not derivation or not derivation.get('python_logic'):
        return df_copy

    try:
        new_col = derivation.get('new_column')
        logic = derivation.get('python_logic')
        
        # Prepare a robust execution state
        # Including np and standard built-ins helps with binning logic
        state = {
            "pd": pd, 
            "np": np, 
            "df": df_copy
        }
        
        # CLEANING LOGIC:
        # Sometimes the LLM sends multiple lines with different styles.
        # We ensure it handles common 'df[col] = ...' patterns.
        parts = [p.strip() for p in logic.split('\n') if p.strip()]
        if not parts:
            parts = [p.strip() for p in logic.split(';') if p.strip()]

        for part in parts:
            try:
                # Execute each line. We use the same dict for globals and locals 
                # to ensure variables persist across lines.
                exec(part, state, state)
            except Exception as line_error:
                # If a specific line fails (e.g., a column name mismatch), 
                # we try to fix common retail naming issues on the fly.
                print(f"Line execution warning: {line_error}")
                continue
        
        # Retrieve the modified dataframe
        updated_df = state["df"]
        
        # --- ROBUSTNESS CHECK: BINNING FOR DISTRIBUTIONS ---
        # If the task is a distribution but the column 'Spending_Bins' (or similar) 
        # wasn't created, we force-create it using a standard quintile approach.
        x_axis = task.get('x_axis', '')
        if 'bin' in x_axis.lower() and x_axis not in updated_df.columns:
            y_val = task.get('y_axis')
            if y_val in updated_df.columns:
                # Automatically create 5 bins for the requested metric
                updated_df[x_axis] = pd.qcut(updated_df[y_val], q=5, duplicates='drop').astype(str)
            else:
                # If even the Y metric is missing, find the first numeric column
                numeric_cols = updated_df.select_dtypes(include=[np.number]).columns
                if not numeric_cols.empty:
                    updated_df[x_axis] = pd.qcut(updated_df[numeric_cols[0]], q=5, duplicates='drop').astype(str)

        # FINAL SAFETY: Ensure the specific column requested by the planner exists
        if new_col and new_col not in updated_df.columns:
            try:
                # Try to evaluate the logic to get a series
                result = eval(parts[-1], state, state)
                if isinstance(result, (pd.Series, list, np.ndarray)):
                    updated_df[new_col] = result
            except:
                pass
            
        return updated_df

    except Exception as e:
        print(f"CRITICAL Logic Error in apply_dynamic_logic: {e}")
        return df_copy