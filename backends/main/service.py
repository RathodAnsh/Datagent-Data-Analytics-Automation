# backend/service.py
import json
import asyncio
from datagent import (
    ProfilerAgent,
    SemanticAgent,
    QualityAgent,
    CleaningAgent,
    ValidationAgent,
    Config,
    OrchestratorAgent
)
from api_adapter import dataset_info_from_csv

def make_json_safe(obj):
    """
    Recursively convert numpy / polars / non-JSON types
    into JSON-serializable Python types
    """
    import numpy as np
    import polars as pl

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (pl.datatypes.DataType,)):
        return str(obj)
    else:
        return obj


def _log(text: str, kind: str = "info") -> str:
    """Emit a terminal log line event."""
    return json.dumps({"type": "log", "kind": kind, "text": text}) + "\n"


async def clean_dataset_stream(file_path: str):
    api_key = Config.load_api_key()

    # Initialize Agents
    profiler    = ProfilerAgent(api_key)
    semantic    = SemanticAgent(api_key)
    quality     = QualityAgent(api_key)
    cleaner     = CleaningAgent(api_key)
    validator   = ValidationAgent(api_key)
    orchestrator = OrchestratorAgent(api_key)

    # ── STEP 1: PROFILING ─────────────────────────────────────────────────────
    yield json.dumps({"step": "profiling", "status": "processing"}) + "\n"
    yield _log("Initialising profiling agent…", "system")
    yield _log(f"Loading dataset: {file_path}", "system")

    df_original = profiler.load_dataset(file_path)
    yield _log(f"Dataset loaded — {len(df_original):,} rows × {len(df_original.columns)} columns", "success")

    profile = profiler.comprehensive_profile(df_original)

    # Emit per-column null summary
    col_stats = profile.get("columns", [])   # list of dicts, each with 'name', 'null_count', etc.
    null_cols  = [(v["name"], v["null_count"]) for v in col_stats if v.get("null_count", 0) > 0]
    yield _log(f"Profile complete — {len(col_stats)} columns scanned", "success")
    if null_cols:
        for col, cnt in null_cols[:8]:          # cap at 8 so it doesn't flood
            yield _log(f"  ↳ '{col}' has {cnt:,} null values", "warn")
        if len(null_cols) > 8:
            yield _log(f"  ↳ … and {len(null_cols) - 8} more columns with nulls", "warn")
    else:
        yield _log("  ↳ No null values found in any column", "info")

    dup_rows = profile.get("duplicate_rows", 0)
    if dup_rows:
        yield _log(f"  ↳ {dup_rows:,} duplicate rows detected", "warn")

    yield json.dumps({"step": "profiling", "status": "completed"}) + "\n"

    # ── STEP 2: SEMANTIC UNDERSTANDING ────────────────────────────────────────
    yield json.dumps({"step": "semantic", "status": "processing"}) + "\n"
    yield _log("Running semantic analysis on column names & values…", "system")

    semantic_map  = semantic.identify_semantics(df_original, profile)
    relationships = semantic.discover_relationships(df_original, semantic_map)

    for col, meta in list(semantic_map.items())[:10]:
        sem_type = meta.get("semantic_type", "unknown") if isinstance(meta, dict) else str(meta)
        yield _log(f"  ↳ '{col}'  →  {sem_type}", "info")
    if len(semantic_map) > 10:
        yield _log(f"  ↳ … {len(semantic_map) - 10} more columns mapped", "info")

    if relationships:
        yield _log(f"Discovered {len(relationships)} column relationship(s):", "success")
        for rel in relationships[:4]:
            if isinstance(rel, dict):
                desc = rel.get("description", str(rel))
                yield _log(f"  ↳ {desc}", "info")
    else:
        yield _log("No inter-column relationships found", "info")

    semantic_payload = {
        "semantic_map":  make_json_safe(semantic_map),
        "relationships": make_json_safe(relationships),
    }
    with open("semantic_metadata.json", "w") as f:
        json.dump(semantic_payload, f, indent=2)

    yield json.dumps({"step": "semantic", "status": "completed"}) + "\n"

    # ── STEP 3-6: DETECTING PROBLEMS ─────────────────────────────────────────
    yield _log("Starting quality problem detection across all categories…", "system")
    problems = quality.detect_all_problems(df_original, profile, semantic_map, relationships)

    detection_steps = ["missing", "outliers", "duplicates", "invalid"]
    step_keys       = ["missing_values", "outliers", "duplicates", "invalid_values"]
    step_labels     = ["Missing Values", "Outliers", "Duplicate Rows", "Invalid Values"]

    for step, key, label in zip(detection_steps, step_keys, step_labels):
        yield json.dumps({"step": step, "status": "processing"}) + "\n"
        issues = problems.get(key, [])
        if issues:
            total = sum(i.get("count", i.get("consensus_count", 1)) for i in issues)
            yield _log(f"[{label}] {total:,} issues across {len(issues)} column(s)", "warn")
            for issue in issues[:5]:
                col   = issue.get("column", issue.get("type", "—"))
                cnt   = issue.get("count", issue.get("consensus_count", ""))
                strat = issue.get("strategy", issue.get("imputation_strategy", ""))
                detail = f"  ↳ '{col}'" + (f": {cnt:,} affected" if cnt else "") + (f" — strategy: {strat}" if strat else "")
                yield _log(detail, "warn")
            if len(issues) > 5:
                yield _log(f"  ↳ … {len(issues) - 5} more", "warn")
        else:
            yield _log(f"[{label}] None found ✓", "success")
        await asyncio.sleep(0.1)
        yield json.dumps({"step": step, "status": "completed"}) + "\n"

    # ── STEP 7: APPLYING CLEANING ─────────────────────────────────────────────
    yield json.dumps({"step": "applying", "status": "processing"}) + "\n"
    yield _log("Applying intelligent cleaning pipeline…", "system")

    df_cleaned, cleaning_log = cleaner.clean_dataset(
        df_original, problems, semantic_map, relationships
    )

    # Imputation log
    for entry in cleaning_log.get("imputation", [])[:8]:
        col    = entry.get("column", "?")
        method = entry.get("method", "?")
        filled = entry.get("rows_filled", 0)
        if filled:
            yield _log(f"  ↳ Imputed '{col}' — {filled:,} cells filled via {method}", "success")

    # Outlier log
    for entry in cleaning_log.get("outliers", [])[:6]:
        col    = entry.get("column", "?")
        action = entry.get("action", "handled")
        rows   = entry.get("rows_affected", 0)
        if rows:
            yield _log(f"  ↳ Outliers in '{col}' — {rows:,} rows {action}", "success")

    # Duplicate log
    for entry in cleaning_log.get("duplicates", []):
        removed = entry.get("rows_removed", 0)
        if removed:
            yield _log(f"  ↳ Removed {removed:,} duplicate rows", "success")

    # Calculation / relationship enforcement log
    for entry in cleaning_log.get("calculations", [])[:4]:
        col  = entry.get("column", "?")
        rows = entry.get("rows_recalculated", 0)
        if rows:
            yield _log(f"  ↳ Recalculated '{col}' — {rows:,} rows corrected", "success")

    # Normalisation
    norm_suggestions = cleaning_log.get("normalization_suggestions", [])
    if norm_suggestions:
        yield _log(f"  ↳ {len(norm_suggestions)} categorical value(s) normalised", "success")

    yield _log(f"Cleaning complete — output: {len(df_cleaned):,} rows × {len(df_cleaned.columns)} cols", "success")

    # Validate & export
    yield _log("Validating cleaning quality…", "system")
    validation = validator.validate_cleaning(df_original, df_cleaned, semantic_map, relationships)

    score = validation.get("quality_score", validation.get("score", None))
    if score is not None:
        yield _log(f"Quality score: {score}", "success" if float(score) >= 0.8 else "warn")

    report = orchestrator._generate_report(
        df_original, df_cleaned, profile, semantic_map,
        relationships, problems, cleaning_log, validation
    )

    output_file = "cleaned_dataset.csv"
    orchestrator._export_results(df_cleaned, report, output_file)
    cleaned_data_info = dataset_info_from_csv(output_file)
    yield _log(f"Exported cleaned dataset → {output_file}", "success")

    yield json.dumps({"step": "applying", "status": "completed"}) + "\n"

    yield json.dumps({
        "type": "suggestions",
        "data": cleaning_log.get("normalization_suggestions", [])
    }) + "\n"


    # =========================================================
    # FINAL SUMMARY — FIXED COUNTING LOGIC
    # =========================================================
    #
    # ROOT CAUSE OF WRONG COUNTS (old code):
    #
    # 1. totalIssues — was summing len() of each problem category.
    #    Each category is a LIST OF AFFECTED COLUMNS, not a count of
    #    individual bad cells/rows. So 5 columns with missing values
    #    counted as 5, not as the actual 7,229 missing cells.
    #    FIX: sum the actual `.count` field inside each problem item.
    #
    # 2. missingValuesFixed — was using len(cleaning_log['imputation']).
    #    imputation list has ONE entry PER COLUMN processed, not per row.
    #    So 5 imputed columns showed as "5 fixed" instead of the real
    #    number of cells that were filled.
    #    FIX: sum `rows_filled` from each imputation log entry.
    #
    # 3. outliersHandled — was using len(cleaning_log['outliers']).
    #    outliers list has ONE entry PER COLUMN, not per outlier row.
    #    FIX: sum `rows_affected` from each outlier log entry.
    #
    # 4. duplicatesRemoved — was summing `rows_removed` from dup_log
    #    with a fallback of len(dup_log). The fallback is wrong because
    #    len(dup_log) counts number of duplicate-type operations, not rows.
    #    FIX: only sum `rows_removed`; if 0 duplicates exist, show 0.
    #
    # 5. columnsRecalculated — len(cleaning_log['calculations']) counts
    #    how many relationship formulas were enforced (e.g., 1 for
    #    Total = Qty × Price). Each entry has `rows_recalculated` inside.
    #    FIX: sum `rows_recalculated` so it shows the real row count.
    #
    # =========================================================

    # 1. Strategy Name — unchanged, this was correct
    stats = cleaning_log.get('strategy_stats', {})
    ml_count = stats.get('ml_used', 0)
    
    if ml_count > 0:
        strategy_name = f"Adaptive ML ({ml_count}) + Rules"
    else:
        strategy_name = "Statistical & Rules"

    # 2. totalIssues
    #    Sum the actual COUNT of bad records inside each detected problem,
    #    not just the number of problem entries (columns).
    total_issues = 0

    for issue in problems.get('missing_values', []):
        # Each entry: {'column': ..., 'count': 1213, ...}
        total_issues += issue.get('count', 0)

    for issue in problems.get('outliers', []):
        # Each entry: {'column': ..., 'consensus_count': 60, ...}
        total_issues += issue.get('consensus_count', 0)

    for issue in problems.get('duplicates', []):
        # Each entry: {'type': 'exact_duplicates', 'count': N, ...}
        total_issues += issue.get('count', 0)

    for issue in problems.get('invalid_values', []):
        # Each entry: {'column': ..., 'count': N, ...}
        total_issues += issue.get('count', 0)

    for issue in problems.get('inconsistencies', []):
        # Each entry has either 'variants' or 'count'
        total_issues += issue.get('variants', issue.get('count', 0))

    for issue in problems.get('business_violations', []):
        # Each entry: {'column': ..., 'count': N, ...}
        total_issues += issue.get('count', 0)

    # 3. missingValuesFixed
    #    Sum rows_filled across all imputation log entries.
    #    Entries with method='skipped' or 'will_be_calculated' have no
    #    rows_filled key — .get(..., 0) handles that safely.
    missing_fixed = sum(
        entry.get('rows_filled', 0)
        for entry in cleaning_log.get('imputation', [])
    )

    # 4. outliersHandled
    #    Sum rows_affected across all outlier log entries.
    outliers_handled = sum(
        entry.get('rows_affected', 0)
        for entry in cleaning_log.get('outliers', [])
    )

    # 5. duplicatesRemoved
    #    Sum rows_removed from the duplicates log.
    #    This is already the actual row count removed from the DataFrame.
    duplicates_removed = sum(
        entry.get('rows_removed', 0)
        for entry in cleaning_log.get('duplicates', [])
    )

    # 6. columnsRecalculated
    #    Each entry in calc_log has rows_recalculated (rows where the
    #    formula was enforced). Sum those for a meaningful number.
    #    Fall back to the entry count if rows_recalculated is absent.
    columns_recalculated = sum(
        entry.get('rows_recalculated', 1)
        for entry in cleaning_log.get('calculations', [])
    )

    # 7. Construct final summary
    final_summary = {
        "totalIssues": total_issues,
        "missingValuesFixed": missing_fixed,
        "outliersHandled": outliers_handled,
        "duplicatesRemoved": duplicates_removed,
        "columnsRecalculated": columns_recalculated,
        "strategyUsed": strategy_name
    }

    # Send the final payload
    final_payload = make_json_safe({
        "cleanedData": cleaned_data_info,
        "summary": final_summary
    })
    yield json.dumps({"result": final_payload}) + "\n"