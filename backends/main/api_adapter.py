# backend/api_adapter.py
import polars as pl

def dataset_info_from_csv(csv_path: str, preview_rows: int = 10):
    df = pl.read_csv(csv_path)

    return {
        "fileName": csv_path.split("/")[-1],
        "totalRows": df.height,
        "totalColumns": df.width,
        "previewData": df.head(preview_rows).to_dicts(),
        "filePath": csv_path
    }
