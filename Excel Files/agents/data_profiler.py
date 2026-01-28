#3. data_profiler.py

def profile_dataset(df):
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": df.dtypes.astype(str),
        "head": df.head(5).to_dict(),
        "tail": df.tail(5).to_dict()
    }
    return summary
