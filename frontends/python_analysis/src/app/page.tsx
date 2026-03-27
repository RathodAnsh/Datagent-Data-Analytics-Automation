"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { uploadCsv, analyzeQuestion, AnalyzeResponse } from "@/lib/api";
import { Loader2, UploadCloud, Send, AlertTriangle, LayoutDashboard, CheckCircle2, Code2, ChevronDown, ChevronUp, Pencil } from "lucide-react";

type ChartSpec = {
  id: string;
  imageData: string;
};

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [question, setQuestion] = useState("");
  const [uploadInfo, setUploadInfo] = useState<{
    rows: number;
    columns: number;
    logs: string[];
  } | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [codeExpanded, setCodeExpanded] = useState(false);
  const router = useRouter();

  // ── NEW: Context passed from DATAGENT platform selection ─────────────────
  // When redirected from PlatformPage, URL contains:
  //   ?dataset_url=...&requirements=...&dataset_file=...
  // We auto-fetch the CSV and pre-fill the question so user skips upload.
  const [fromDatagent, setFromDatagent]     = useState(false);
  const [autoUploading, setAutoUploading]   = useState(false);
  const [isEditingQ, setIsEditingQ]         = useState(false);

  useEffect(() => {
    const params      = new URLSearchParams(window.location.search);
    const datasetUrl  = params.get('dataset_url');
    const reqs        = params.get('requirements');

    if (!datasetUrl) return;   // normal standalone mode — nothing to do

    setFromDatagent(true);
    if (reqs) setQuestion(decodeURIComponent(reqs));

    // Clean URL so params don't persist on refresh
    window.history.replaceState({}, '', window.location.pathname);

    // Auto-fetch the cleaned CSV from DATAGENT backend and upload to this backend
    const autoUpload = async () => {
      setAutoUploading(true);
      setError(null);
      try {
        const blob = await fetch(datasetUrl).then(r => r.blob());
        const csvFile = new File([blob], 'dashboard_dataset.csv', { type: 'text/csv' });
        const res = await uploadCsv(csvFile);
        setUploadInfo({ rows: res.rows, columns: res.columns, logs: res.logs });
      } catch {
        setError('Failed to auto-load dataset from DATAGENT. Please upload manually.');
        setFromDatagent(false);
      } finally {
        setAutoUploading(false);
      }
    };
    autoUpload();
  }, []);

  // ── Upload handler (calls POST /upload-data) ──────────────────────────────
  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const res = await uploadCsv(file);
      setUploadInfo({ rows: res.rows, columns: res.columns, logs: res.logs });
      setAnalysis(null);
      setQuestion("");
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      setError(err?.response?.data?.detail ?? "Failed to upload file. Ensure the backend is running.");
    } finally {
      setUploading(false);
    }
  };

  // ── Analyze handler (calls POST /analyze) ────────────────────────────────
  const handleAnalyze = async () => {
    if (!question.trim()) return;
    setAnalyzing(true);
    setError(null);
    setCodeExpanded(false);
    try {
      const res = await analyzeQuestion(question.trim());
      setAnalysis(res);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      setError(err?.response?.data?.detail ?? "Failed to run analysis. Ensure a dataset is uploaded and backend is running.");
    } finally {
      setAnalyzing(false);
    }
  };

  const chartSpecs: ChartSpec[] =
    analysis?.charts.map((b64, idx) => ({ id: `chart-${idx}`, imageData: b64 })) ?? [];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">

      {/* ── AUTO-LOADING BANNER (shown when coming from DATAGENT) ── */}
      {autoUploading && (
        <div className="flex items-center gap-3 rounded-2xl border border-violet-500/30 bg-violet-950/20 px-5 py-4 text-sm text-violet-300 animate-in fade-in">
          <Loader2 className="h-5 w-5 animate-spin shrink-0" />
          <span>Loading your cleaned dataset from DATAGENT pipeline…</span>
        </div>
      )}

      {/* ── DATAGENT CONTEXT BANNER (shown after auto-load succeeds) ── */}
      {fromDatagent && uploadInfo && !autoUploading && (
        <div className="flex items-center justify-between gap-3 rounded-2xl border border-emerald-500/30 bg-emerald-950/10 px-5 py-4 animate-in fade-in">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-emerald-300">Dataset loaded from DATAGENT</p>
              <p className="text-[11px] text-emerald-500/70">
                {uploadInfo.rows.toLocaleString()} rows · {uploadInfo.columns} columns · requirements pre-filled
              </p>
            </div>
          </div>
          <span className="text-[10px] bg-emerald-900/40 text-emerald-400 border border-emerald-500/20 px-2 py-1 rounded-full font-mono">
            AUTO-LOADED
          </span>
        </div>
      )}

      <section className="grid gap-6 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="space-y-4">

          {/* ── STEP 1: Upload — HIDDEN when coming from DATAGENT ── */}
          {!fromDatagent && (
            <div className={`rounded-2xl border transition-all duration-300 ${uploadInfo ? 'border-emerald-500/30 bg-emerald-950/10' : 'border-slate-800 bg-slate-900/60'} p-4`}>
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  1. Upload dataset (CSV)
                  {uploadInfo && <CheckCircle2 className="h-4 w-4 text-emerald-500" />}
                </h2>
              </div>
              <div className="space-y-3">
                <label className="flex items-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-900/80 px-3 py-3 cursor-pointer hover:border-violet-500/70 transition-colors">
                  <UploadCloud className="h-5 w-5 text-violet-400 shrink-0" />
                  <div className="flex-1">
                    <div className="text-xs text-slate-100">{file ? file.name : "Click to choose a CSV file"}</div>
                    <div className="text-[11px] text-slate-500">Max size depends on backend limits</div>
                  </div>
                  <input type="file" accept=".csv" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; setFile(f ?? null); }} />
                </label>
                <button onClick={handleUpload} disabled={!file || uploading}
                  className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white shadow-sm hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all">
                  {uploading && <Loader2 className="h-4 w-4 animate-spin text-violet-100" />}
                  <span>{uploading ? "Uploading & profiling..." : "Upload & profile"}</span>
                </button>
              </div>
              {uploadInfo && (
                <div className="mt-4 space-y-2 rounded-lg bg-slate-950/60 p-3 text-xs animate-in fade-in slide-in-from-top-2">
                  <div className="flex gap-4">
                    <div><div className="text-slate-400 text-[11px]">Rows</div><div className="font-semibold text-emerald-400">{uploadInfo.rows.toLocaleString()}</div></div>
                    <div><div className="text-slate-400 text-[11px]">Columns</div><div className="font-semibold text-emerald-400">{uploadInfo.columns}</div></div>
                  </div>
                  {uploadInfo.logs.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-slate-800">
                      <div className="text-[11px] font-medium text-slate-400 mb-1">Cleaning log</div>
                      <ul className="space-y-1 text-[11px] text-slate-300 max-h-24 overflow-auto scrollbar-thin scrollbar-thumb-slate-700">
                        {uploadInfo.logs.map((log, idx) => <li key={idx}>{log}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── STEP 2: Question — shown after upload OR auto-loaded from DATAGENT ── */}
          {(uploadInfo || fromDatagent) && !autoUploading && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 space-y-3 animate-in fade-in slide-in-from-top-4 duration-500">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-100">
                  {fromDatagent ? 'Analysis requirements' : '2. Ask your analysis question'}
                </h2>
                {fromDatagent && !isEditingQ && (
                  <button onClick={() => setIsEditingQ(true)}
                    className="flex items-center gap-1 text-[11px] text-violet-400 hover:text-violet-300 transition-colors">
                    <Pencil className="h-3 w-3" /> Edit
                  </button>
                )}
              </div>

              {/* Read-only view when from DATAGENT and not editing */}
              {fromDatagent && !isEditingQ ? (
                <div className="rounded-lg border border-slate-700 bg-slate-950/50 px-3 py-3 text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">
                  {question || <span className="text-slate-500 italic">No requirements provided</span>}
                </div>
              ) : (
                <textarea
                  className="w-full resize-none rounded-lg border border-slate-800 bg-slate-950/70 px-3 py-2 text-xs text-slate-100 outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500/60 min-h-[84px]"
                  placeholder="Ex: Show total sales by month and region, and top 5 products"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
              )}

              <div className="flex justify-between items-center">
                <p className="text-[11px] text-slate-500 max-w-[60%]">
                  {fromDatagent ? 'Pre-filled from your DATAGENT feature selection.' : 'The backend uses your existing Gemini-powered Python automation unchanged.'}
                </p>
                <button onClick={handleAnalyze} disabled={!question.trim() || analyzing}
                  className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-2 text-xs font-medium text-emerald-950 shadow-sm hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all">
                  {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                  <span>{analyzing ? "Running analysis..." : "Run analysis"}</span>
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-xl border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-100 animate-in fade-in">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
              <div>{error}</div>
            </div>
          )}
        </div>

        <div className="space-y-4" />
      </section>

      {/* ── STEP 3: Visual Insights ── */}
      {(analyzing || analysis || chartSpecs.length > 0) && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 space-y-3 animate-in fade-in duration-700">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-sm font-semibold text-slate-100">
              {fromDatagent ? '2.' : '3.'} Visual insights
            </h2>
            <p className="text-[11px] text-slate-500">
              Visuals are generated by your existing Python logic and streamed back as images.
            </p>
          </div>
          {analyzing && (
            <div className="flex items-center justify-center gap-2 p-8 text-xs text-violet-400 bg-slate-950/40 rounded-xl border border-dashed border-slate-700">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Generating charts from Python backend…</span>
            </div>
          )}
          {!analyzing && chartSpecs.length === 0 && (
            <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-6 text-center text-xs text-slate-500">
              Charts will appear here once the analysis code in your backend has saved them.
            </div>
          )}
          {chartSpecs.length > 0 && (
            <div className="grid gap-4 md:grid-cols-2">
              {chartSpecs.map((chart) => (
                <div key={chart.id} className="rounded-xl border border-slate-800 bg-slate-950/70 p-3 animate-in zoom-in-95 duration-500">
                  <img src={`data:image/png;base64,${chart.imageData}`} alt="Analysis chart" className="w-full rounded-md border border-slate-900/60" />
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ── CODE SECTION ── */}
      {analysis?.last_analysis_code && (
        <section className="rounded-2xl border border-slate-700/60 bg-slate-900/60 p-4 space-y-3 animate-in fade-in duration-700">
          <button type="button" onClick={() => setCodeExpanded((v) => !v)} className="w-full flex items-center justify-between gap-2 group">
            <div className="flex items-center gap-2">
              <Code2 className="h-4 w-4 text-violet-400" />
              <h2 className="text-sm font-semibold text-slate-100">Generated Python Code</h2>
              <span className="text-[10px] text-slate-500 bg-slate-800 rounded px-1.5 py-0.5">Matplotlib · Seaborn · Pandas · NumPy</span>
            </div>
            <span className="text-slate-400 group-hover:text-slate-200 transition-colors">
              {codeExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </span>
          </button>
          {codeExpanded && (
            <div className="animate-in fade-in slide-in-from-top-2 duration-300">
              <pre className="overflow-auto rounded-xl bg-slate-950/80 border border-slate-800 p-4 text-[11px] leading-relaxed text-slate-300 max-h-[520px] scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                <code>{analysis.last_analysis_code}</code>
              </pre>
            </div>
          )}
        </section>
      )}

      {/* ── STEP 4: Dashboard ── */}
      {analysis && analysis.dashboard_ready && (
        <section className="rounded-2xl border border-amber-500/30 bg-amber-950/10 p-4 flex items-center justify-between animate-in slide-in-from-bottom-4 duration-500">
          <div>
            <h2 className="text-sm font-semibold text-amber-100">
              {fromDatagent ? '3.' : '4.'} Create interactive dashboard
            </h2>
            <p className="text-[11px] text-amber-500/70">
              After charts are generated, you can open a separate dashboard view with interactive visuals and KPI slicers driven by the backend.
            </p>
          </div>
          <button type="button" onClick={() => router.push("/dashboard")}
            className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-amber-500 px-4 py-2.5 text-xs font-bold text-amber-950 shadow-sm hover:bg-amber-400 transition-all hover:scale-105">
            <LayoutDashboard className="h-4 w-4" />
            <span>Open dashboard</span>
          </button>
        </section>
      )}
    </div>
  );
}