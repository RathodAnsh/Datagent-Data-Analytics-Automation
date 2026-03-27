import os
import io
import json
import re
import pandas as pd
from google import genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Conditional import for Windows-specific automation
try:
    import win32com.client as win32
    PYWIN32_AVAILABLE = True
except ImportError:
    PYWIN32_AVAILABLE = False

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

current_session = {"df": None, "raw_df": None, "questions": [], "analysis_plan": None}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), questions: str = Form(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format.")

    current_session["raw_df"] = df.copy()
    current_session["df"] = df
    try:
        parsed = json.loads(questions)
        if isinstance(parsed, list):
            current_session["questions"] = [q.strip() for q in parsed if q.strip()]
        else:
            current_session["questions"] = [q.strip() for q in questions.split('\n') if q.strip()]
    except Exception:
        current_session["questions"] = [q.strip() for q in questions.split('\n') if q.strip()]

    null_count = int(df.isnull().sum().sum())
    preview_rows = df.head(10).fillna("").to_dict(orient="records")
    row_count = len(df)
    col_count = len(df.columns)

    if null_count > 0:
        return {
            "status": "warning",
            "message": f"Found {null_count} missing values across {row_count} rows and {col_count} columns.",
            "preview_rows": preview_rows,
            "row_count": row_count,
            "col_count": col_count,
            "null_count": null_count
        }

    return {
        "status": "success",
        "message": "Dataset is clean.",
        "preview_rows": preview_rows,
        "row_count": row_count,
        "col_count": col_count,
        "null_count": 0
    }


@app.post("/clean")
async def clean_dataset():
    if current_session["df"] is None:
        raise HTTPException(status_code=400)
    current_session["df"] = current_session["df"].fillna("Unknown")
    return {
        "status": "success",
        "message": "Cleaning complete.",
        "preview_rows": current_session["df"].head(10).to_dict(orient="records")
    }


@app.post("/analyze")
async def analyze_with_gemini():
    if current_session["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset found.")

    df = current_session["df"]
    serializable_summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).fillna("").to_dict(orient="records")
    }

    prompt = f"""
You are a Senior Data Engineer and Retail Analytics Expert.
**Task:** Analyze the provided dataset and questions to create an execution plan consisting of Type Correction, Data Derivation, and Visualization.

**DATASET SUMMARY:**
{json.dumps(serializable_summary)}

**USER QUESTIONS:**
{current_session["questions"]}

---

### STEP 1: TYPE CORRECTION (MANDATORY CHECK) (If the datatypes are correct return none)
Inspect the 'dtypes'. For example - if a column is 'object' but contains dates or numbers, you MUST provide a correction query.
**Example:**
- Date (string) -> Date (datetime64)
- Query: `df['date'] = pd.to_datetime(df['date'])`

### STEP 2: DATA DERIVATION
Based on the user questions, create new columns from existing ones if required.
**Example:**
- Column to form: 'Month' (string)
- Source: 'date' (datetime64)
- Query: `df['Month'] = df['date'].dt.month_name()`

### STEP 3: VISUALIZATION LOGIC
Generate a visualization plan for EACH question.
- Use only Excel-supported charts: bar, line, pie, scatter.
- Specify which column would make a good 'Slicer' (filter) for this chart.
- **PIE CHART RULE (CRITICAL):** For pie charts, x_axis MUST be the categorical column (the labels/slices, e.g. gender, category, region) and y_axis MUST be the numeric column (the values, e.g. total_price, quantity, count). NEVER leave x_axis empty for a pie chart.

---

### OUTPUT SCHEMA (STRICT JSON ONLY):
Return a JSON object with this exact structure:

{{
  "type_updates": [{{ "column": "name", "updated_type": "type", "reason": "why", "python_query": "query" }}],
  "derivations": [{{ "new_column": "name", "new_type": "type", "source_column": "name", "python_query": "query" }}],
  "visualizations": [
    {{
      "question": "string",
      "chart_type": "bar | line | pie | scatter",
      "x_axis": "column_name",
      "y_axis": "column_name",
      "aggregation": "sum | mean | count",
      "slicer_column": "column_name",
      "plotly_preview_code": "..."
    }}
  ]
}}
**RETURN ONLY THE RAW JSON LIST. NO PREAMBLE.**
"""

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        raw_text = response.text
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("Invalid AI Response: No JSON object found")

        analysis_data = json.loads(json_match.group())

        exec_namespace = {"pd": pd, "df": df}

        for update in analysis_data.get("type_updates", []):
            query = update.get("python_query", "").strip()
            if query:
                try:
                    exec(query, exec_namespace)
                    df = exec_namespace["df"]
                except Exception as e:
                    print(f"Type update exec error for '{update.get('column')}': {e}")

        for der in analysis_data.get("derivations", []):
            query = der.get("python_query", "").strip()
            if query:
                try:
                    exec(query, exec_namespace)
                    df = exec_namespace["df"]
                except Exception as e:
                    print(f"Derivation exec error for '{der.get('new_column')}': {e}")

        chart_data_results = []
        for task in analysis_data.get("visualizations", []):
            x   = task.get('x_axis')
            y   = task.get('y_axis')
            agg = task.get('aggregation')
            chart_type = task.get('chart_type', '').lower()

            # ── PIE CHART FIX ─────────────────────────────────────────────────
            # Gemini sometimes puts the category column in y_axis and leaves
            # x_axis empty (or puts the numeric in x_axis) for pie charts.
            # Rule: for pie, x must be categorical (object/string dtype) and
            # y must be numeric. Detect and swap when this is violated.
            if chart_type == 'pie' and x and y and x in df.columns and y in df.columns:
                x_is_numeric = pd.api.types.is_numeric_dtype(df[x])
                y_is_numeric = pd.api.types.is_numeric_dtype(df[y])
                if x_is_numeric and not y_is_numeric:
                    # x is numeric, y is categorical — swap them
                    x, y = y, x
                    task = {**task, 'x_axis': x, 'y_axis': y}

            # If x_axis is empty for pie, try to recover by using y_axis as
            # category and find the first numeric column as the value column
            if chart_type == 'pie' and (not x) and y and y in df.columns:
                if not pd.api.types.is_numeric_dtype(df[y]):
                    # y is categorical — it should be x; find a numeric col for y
                    numeric_cols = df.select_dtypes(include='number').columns.tolist()
                    if numeric_cols:
                        x, y = y, numeric_cols[0]
                        task = {**task, 'x_axis': x, 'y_axis': y}
            # ─────────────────────────────────────────────────────────────────

            data_points = []
            if x and y and x in df.columns and y in df.columns:
                if agg:
                    try:
                        chart_df = df.groupby(x)[y].agg(agg).reset_index()
                        chart_df[y] = pd.to_numeric(chart_df[y], errors='coerce').fillna(0)
                        data_points = chart_df.fillna("").to_dict(orient="records")
                    except Exception as e:
                        print(f"Aggregation error: {e}")
                else:
                    data_points = df[[x, y]].fillna("").head(100).to_dict(orient="records")

            chart_data_results.append({**task, "data": data_points})

        current_session["df"] = df
        current_session["analysis_plan"] = chart_data_results

        return {
            "status": "success",
            "plan": chart_data_results,
            "type_logs": analysis_data.get("type_updates", []),
            "derivation_logs": analysis_data.get("derivations", []),
            "updated_preview": df.head(10).fillna("").to_dict(orient="records")
        }

    except Exception as e:
        return {"status": "error", "message": f"Automation Error: {str(e)}"}


# Helper: convert column number (1-based) to Excel letter(s)
def col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


@app.get("/download")
async def download_report(mode: str = "general"):
    if current_session["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset found.")

    file_name = "AI_Report.xlsx"
    file_path = os.path.abspath(file_name)

    generated_ok = False
    try:
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            current_session["raw_df"].to_excel(writer, sheet_name='Raw_Dataset', index=False)

            current_session["df"].to_excel(writer, sheet_name='Cleaned_Dataset', index=False)
            ws_clean = writer.sheets['Cleaned_Dataset']
            max_row, max_col = current_session["df"].shape
            ws_clean.add_table(0, 0, max_row, max_col - 1, {
                'columns': [{'header': c} for c in current_session["df"].columns],
                'name': 'MainTable',
                'style': 'TableStyleMedium9'
            })

            if current_session["analysis_plan"]:
                insight_cols = ['question', 'chart_type', 'x_axis', 'y_axis', 'aggregation', 'slicer_column']
                plan_df = pd.DataFrame(current_session["analysis_plan"])
                available = [c for c in insight_cols if c in plan_df.columns]
                plan_df[available].to_excel(writer, sheet_name='AI_Insights', index=False)

                for idx, viz in enumerate(current_session["analysis_plan"]):
                    data = viz.get("data", [])
                    pivot_sheet_name = f"Pivot_{idx + 1}"[:31]

                    if mode == "windows":
                        pd.DataFrame().to_excel(writer, sheet_name=pivot_sheet_name, index=False)
                    else:
                        if not data:
                            continue

                        x_col = viz.get("x_axis", "")
                        y_col = viz.get("y_axis", "")
                        question = viz.get("question", f"Question {idx + 1}")
                        chart_type_raw = viz.get("chart_type", "bar")

                        chart_df = pd.DataFrame(data)
                        chart_df.to_excel(writer, sheet_name=pivot_sheet_name, index=False)
                        pivot_ws = writer.sheets[pivot_sheet_name]
                        n_rows = len(chart_df)

                        if n_rows > 0 and x_col in chart_df.columns and y_col in chart_df.columns:
                            x_col_idx = list(chart_df.columns).index(x_col)
                            y_col_idx = list(chart_df.columns).index(y_col)
                            n_data_cols = len(chart_df.columns)

                            pivot_ws.add_table(0, 0, n_rows, n_data_cols - 1, {
                                'columns': [{'header': c} for c in chart_df.columns],
                                'name': f'PivotData_{idx + 1}',
                                'style': 'TableStyleLight9'
                            })

                            chart_type_map = {"bar": "column", "line": "line", "scatter": "scatter", "pie": "pie"}
                            xl_type = chart_type_map.get(chart_type_raw, "column")

                            chart = workbook.add_chart({'type': xl_type})
                            chart.add_series({
                                'name': question,
                                'categories': [pivot_sheet_name, 1, x_col_idx, n_rows, x_col_idx],
                                'values':     [pivot_sheet_name, 1, y_col_idx, n_rows, y_col_idx],
                            })
                            chart.set_title({'name': question})
                            chart.set_style(10)
                            chart.set_size({'width': 480, 'height': 300})

                            insert_col = col_letter(n_data_cols + 2)
                            pivot_ws.insert_chart(f'{insert_col}2', chart)

        generated_ok = True

    except Exception as e:
        print(f"XlsxWriter failed: {e}. Falling back to openpyxl.")

    if not generated_ok:
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                current_session["raw_df"].to_excel(writer, sheet_name='Raw_Dataset', index=False)
                current_session["df"].to_excel(writer, sheet_name='Cleaned_Dataset', index=False)
                if current_session["analysis_plan"]:
                    plan_df = pd.DataFrame(current_session["analysis_plan"])
                    available = [c for c in ['question', 'chart_type', 'x_axis', 'y_axis', 'aggregation', 'slicer_column'] if c in plan_df.columns]
                    plan_df[available].to_excel(writer, sheet_name='AI_Insights', index=False)
                    for idx, viz in enumerate(current_session["analysis_plan"]):
                        data = viz.get("data", [])
                        if data:
                            pd.DataFrame(data).to_excel(writer, sheet_name=f"Pivot_{idx + 1}"[:31], index=False)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Excel generation failed: {str(e2)}")

    if mode == "windows" and PYWIN32_AVAILABLE:
        try:
            excel = win32.Dispatch('Excel.Application')
            excel.DisplayAlerts = False
            excel.Visible = False
            wb = excel.Workbooks.Open(file_path)

            ws_data = wb.Sheets('Cleaned_Dataset')
            used = ws_data.UsedRange
            last_row = used.Rows.Count
            last_col = used.Columns.Count
            src_range = f"'Cleaned_Dataset'!$A$1:${col_letter(last_col)}${last_row}"

            pc = wb.PivotCaches().Create(SourceType=1, SourceData=src_range)

            xl_chart_type_map = {"bar": 51, "line": 4, "pie": 5, "scatter": -4169}

            for idx, viz in enumerate(current_session["analysis_plan"]):
                x_axis     = viz.get('x_axis', '')
                y_axis     = viz.get('y_axis', '')
                slicer_col = viz.get('slicer_column', '')
                question   = viz.get('question', f'Question {idx + 1}')
                chart_type_raw = viz.get('chart_type', 'bar')

                if not x_axis or not y_axis:
                    continue

                pivot_sheet_name = f"Pivot_{idx + 1}"
                try:
                    ws_pivot = wb.Sheets(pivot_sheet_name)
                except Exception:
                    print(f"Sheet '{pivot_sheet_name}' not found, skipping viz {idx + 1}")
                    continue

                try:
                    dest_cell = ws_pivot.Cells(1, 1)
                    pt = pc.CreatePivotTable(TableDestination=dest_cell, TableName=f"PivotTable_{idx + 1}")
                    pt.PivotFields(x_axis).Orientation = 1

                    agg_type = viz.get('aggregation', 'sum').lower()
                    xl_agg = -4157 if agg_type == 'sum' else -4106 if agg_type == 'mean' else -4112
                    pt.AddDataField(pt.PivotFields(y_axis), f"{agg_type.capitalize()} of {y_axis}", xl_agg)

                    chart_obj = ws_pivot.ChartObjects().Add(Left=300, Top=5, Width=420, Height=280)
                    ch = chart_obj.Chart
                    ch.ChartType = xl_chart_type_map.get(chart_type_raw, 51)
                    ch.SetSourceData(pt.TableRange1)
                    ch.HasTitle = True
                    ch.ChartTitle.Text = question

                    if (slicer_col and slicer_col in current_session["df"].columns
                            and slicer_col != x_axis and slicer_col != y_axis):
                        try:
                            pt.PivotFields(slicer_col).Orientation = 3
                            sc = wb.SlicerCaches.Add2(pt, slicer_col)
                            try:
                                n_pivot_rows = current_session["df"][x_axis].nunique() + 4
                            except Exception:
                                n_pivot_rows = 20
                            sc.Slicers.Add(ws_pivot, Name=f"Slicer_{idx + 1}", Caption=slicer_col,
                                           Top=n_pivot_rows * 15 + 10, Left=5, Width=190, Height=220)
                        except Exception as se:
                            print(f"Slicer error viz {idx + 1} field '{slicer_col}': {se}")

                except Exception as pe:
                    print(f"Pivot/chart error viz {idx + 1}: {pe}")

            wb.Save()
            wb.Close()
            excel.Quit()

        except Exception as e:
            print(f"Windows Automation Error: {e}")
            try:
                excel.Quit()
            except Exception:
                pass

    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="File generation failed.")

    return FileResponse(
        path=file_path,
        filename="AI_Report.xlsx",
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )