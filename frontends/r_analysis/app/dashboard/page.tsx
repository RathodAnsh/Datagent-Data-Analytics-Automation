"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import html2canvas from "html2canvas";
import {
  getDashboardConfig,
  updateDashboard,
  KpiValue,
  SlicerMeta,
  PlotlyChart,
} from "@/lib/api";
import {
  Loader2,
  SlidersHorizontal,
  BarChart3,
  AlertTriangle,
  ArrowLeft,
  RefreshCw,
  Download,
} from "lucide-react";

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// ── KPI Formatting ────────────────────────────────────────────────────────────
function formatKpiValue(value: number, fmt: string): string {
  if (fmt === "currency") {
    if (Math.abs(value) >= 1_000_000)
      return `$${(value / 1_000_000).toFixed(1)}M`;
    if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
    return `$${value.toFixed(0)}`;
  }
  if (fmt === "percentage") return `${(value * 100).toFixed(1)}%`;
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

// ── Parse Plotly JSON safely ─────────────────────────────────────────────────
function parsePlotlyJson(json_str: string): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } | null {
  try {
    const parsed = JSON.parse(json_str);

    let data: Plotly.Data[] = [];
    let layout: Record<string, unknown> = {};

    if (Array.isArray(parsed)) {
      const first = parsed[0];
      data   = first?.data   ?? first?.x?.data   ?? [];
      layout = first?.layout ?? first?.x?.layout ?? {};
    } else if (parsed?.x?.data) {
      data   = parsed.x.data;
      layout = parsed.x.layout ?? {};
    } else if (parsed?.data) {
      data   = parsed.data;
      layout = parsed.layout ?? {};
    } else {
      return null;
    }

    if (!Array.isArray(data) || data.length === 0) return null;

    const layoutObj = typeof layout === "object" && layout !== null
      ? (layout as unknown as Record<string, unknown>)
      : {};

    const buildTickMap = (axis: unknown): Map<number, string> => {
      const m = new Map<number, string>();
      if (!axis || typeof axis !== "object") return m;
      const a = axis as Record<string, unknown>;
      const vals = Array.isArray(a.tickvals) ? a.tickvals : [];
      const txts = Array.isArray(a.ticktext) ? a.ticktext : [];
      vals.forEach((v: unknown, i: number) => {
        if (txts[i] !== undefined) m.set(Number(v), String(txts[i]));
      });
      return m;
    };

    const xMap = buildTickMap(layoutObj?.xaxis);
    const yMap = buildTickMap(layoutObj?.yaxis);

    const isUnix = (v: unknown): v is number =>
      typeof v === "number" && v > 1_000_000_000 && v < 4_000_000_000;

    const unixToDate = (v: number): string =>
      new Date(v * 1000).toISOString().slice(0, 10);

    const resolveVal = (v: unknown, map: Map<number, string>, isX: boolean): unknown => {
      if (typeof v === "number") {
        if (map.has(v))        return map.get(v);
        if (isX && isUnix(v)) return unixToDate(v);
      }
      return v;
    };

    const resolveField = (field: unknown, map: Map<number, string>, isX: boolean): unknown => {
      if (Array.isArray(field)) return field.map((v) => resolveVal(v, map, isX));
      const resolved = resolveVal(field, map, isX);
      if (resolved !== undefined && resolved !== null) return [resolved];
      return resolved;
    };

    const fixedData = data.map((trace: Record<string, unknown>) => {
      const t = { ...trace };

      // if (t.x !== undefined) t.x = resolveField(t.x, xMap, true);
      // if (t.y !== undefined) t.y = resolveField(t.y, yMap, false);

      if (!Array.isArray(t.x) && t.x !== undefined && t.x !== null) t.x = [t.x];
      if (!Array.isArray(t.y) && t.y !== undefined && t.y !== null) t.y = [t.y];

      if (!t.type) {
        const mode = String(t.mode ?? "");
        if (t.labels && t.values)                           t.type = "pie";
        else if (t.z)                                       t.type = "heatmap";
        else if (mode.includes("lines"))                    t.type = "scatter";
        else if (mode.includes("markers"))                  t.type = "scatter";
        else if (mode === "text")                           t.type = "scatter";
        else if (Array.isArray(t.x) && Array.isArray(t.y)) t.type = "bar";
        else                                                t.type = "scatter";
      }

      const tt = String(t.type ?? "");

      if (tt === "bar" || tt === "histogram") {
        const em = (t.marker as unknown as Record<string, unknown>) ?? {};
        t.marker = {
          ...em,
          color:   em.color ?? "#7c3aed",
          opacity: em.opacity ?? 0.9,
          line:    { width: 0, ...((em.line as object) ?? {}) },
        };
        if (!t.orientation) t.orientation = "v";
      } else if (tt === "scatter" || tt === "scattergl") {
        if (String(t.mode ?? "") === "text") {
          t.textfont = { ...((t.textfont as object) ?? {}), color: "#e2e8f0" };
        } else {
          const el = (t.line   as unknown as Record<string, unknown>) ?? {};
          const em = (t.marker as unknown as Record<string, unknown>) ?? {};
          t.line   = { width: 2, ...el };
          t.marker = { size: 6, opacity: 0.85, ...em };
          if (!t.mode) t.mode = "lines";
        }
        if (t.fill && !t.fillcolor) {
          const lc = (t.line as unknown as Record<string, unknown>)?.color;
          t.fillcolor = lc
            ? String(lc).replace("rgb(", "rgba(").replace(")", ",0.2)")
            : "rgba(124,58,237,0.15)";
        }
      } else if (tt === "pie") {
        const em = (t.marker as unknown as Record<string, unknown>) ?? {};
        t.marker = {
          ...em,
          colors: em.colors ?? em.color ?? ["#7c3aed","#10b981","#f59e0b","#3b82f6","#ef4444","#8b5cf6","#06b6d4","#f97316"],
          line: { color: "#0a0f1e", width: 1 },
        };
        t.textinfo      = t.textinfo ?? "percent+label";
        t.textfont      = { color: "#e2e8f0", size: 11 };
        t.hole          = t.hole ?? 0;
        t.hovertemplate = t.hovertemplate ?? "%{label}: %{value} (%{percent})<extra></extra>";
      } else if (tt === "box") {
        const em = (t.marker as unknown as Record<string, unknown>) ?? {};
        t.marker    = { color: "#7c3aed", opacity: 0.8, ...em };
        t.line      = { color: "#a78bfa", ...((t.line as object) ?? {}) };
        t.fillcolor = t.fillcolor ?? "rgba(124,58,237,0.2)";
      } else if (tt === "violin") {
        t.fillcolor = t.fillcolor ?? "rgba(124,58,237,0.3)";
        t.line      = { color: "#a78bfa", ...((t.line as object) ?? {}) };
        t.meanline  = { visible: true, color: "#10b981" };
      } else if (tt === "heatmap" || tt === "heatmapgl") {
        if (!t.colorscale) t.colorscale = [
          [0,"#0a0f1e"],[0.25,"#1e2a47"],[0.5,"#7c3aed"],[0.75,"#a78bfa"],[1,"#e2e8f0"],
        ];
        t.showscale = t.showscale ?? true;
      } else if (tt === "funnel") {
        t.textinfo     = t.textinfo     ?? "value+percent initial";
        t.textposition = t.textposition ?? "inside";
      } else if (tt === "waterfall") {
        const c = (t.connector as unknown as Record<string, unknown>) ?? {};
        t.connector = { ...c, line: { color: "#2a3552", width: 1 } };
      }

      return t;
    });

    const xAxisFromR  = (layoutObj?.xaxis  as unknown as Record<string,unknown>) ?? {};
    const yAxisFromR  = (layoutObj?.yaxis  as unknown as Record<string,unknown>) ?? {};
    const legendFromR = (layoutObj?.legend as unknown as Record<string,unknown>) ?? {};

    // Build layout: spread R values, then override only theme-related keys
    const finalLayout: Partial<Plotly.Layout> = {
      ...layoutObj as Partial<Plotly.Layout>,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor:  "rgba(15,22,41,0.4)",
      font:   { color: "#e2e8f0", family: "DM Sans, sans-serif", size: 12 },
      margin: (layoutObj?.margin as Plotly.Layout["margin"]) ?? { t: 48, b: 56, l: 72, r: 24 },
      legend: {
        ...legendFromR as Partial<Plotly.Layout["legend"]>,
        bgcolor:     "rgba(0,0,0,0)",
        bordercolor: "rgba(42,53,82,0.6)",
        borderwidth: 1,
        font:        { color: "#94a3b8", size: 11 },
      },
      xaxis: {
        ...xAxisFromR as Partial<Plotly.Layout["xaxis"]>,
        gridcolor:     "#1e2a47",
        zerolinecolor: "#2a3552",
        tickfont:      { ...((xAxisFromR?.tickfont as object) ?? {}), color: "#94a3b8" },
        title:         { ...(typeof xAxisFromR?.title === "object" ? xAxisFromR.title as object : {}), font: { color: "#94a3b8" } },
      },
      yaxis: {
        ...yAxisFromR as Partial<Plotly.Layout["yaxis"]>,
        gridcolor:     "#1e2a47",
        zerolinecolor: "#2a3552",
        tickfont:      { ...((yAxisFromR?.tickfont as object) ?? {}), color: "#94a3b8" },
        title:         { ...(typeof yAxisFromR?.title === "object" ? yAxisFromR.title as object : {}), font: { color: "#94a3b8" } },
      },
      colorway: ["#7c3aed","#10b981","#f59e0b","#3b82f6","#ef4444","#06b6d4","#f97316","#8b5cf6","#84cc16","#ec4899"],
    };

    return { data: fixedData as Plotly.Data[], layout: finalLayout };

  } catch {
    return null;
  }
}

export default function DashboardPage() {
  const router = useRouter();

  // ── NEW: PDF Download State & Ref ─────────────────────────────────────────
  const dashboardRef = useRef<HTMLDivElement>(null);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

  // ── State ─────────────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [kpiValues, setKpiValues] = useState<KpiValue[]>([]);
  const [slicers, setSlicers] = useState<SlicerMeta[]>([]);
  const [charts, setCharts] = useState<PlotlyChart[]>([]);
  const [pngCharts, setPngCharts] = useState<{ filename: string; image_b64: string }[]>([]);

  // activeFilters: { column_name: Set<string> }
  const [activeFilters, setActiveFilters] = useState<Record<string, Set<string>>>({});
  const [slicersOpen, setSlicersOpen] = useState(true);

  // ── Load initial config (GET /dashboard-config) ──────────────────────────
  const loadConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = await getDashboardConfig();
      setKpiValues(cfg.kpi_values ?? []);
      setSlicers(cfg.slicers ?? []);
      setCharts(cfg.dashboard_charts ?? []);
      setPngCharts(cfg.png_charts ?? []);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      setError(
        err?.response?.data?.detail ??
          "Failed to load dashboard. Ensure the backend is running and analysis has been run."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  // ── Toggle a slicer value ────────────────────────────────────────────────
  const toggleFilter = (colName: string, value: string) => {
    setActiveFilters((prev) => {
      const current = new Set(prev[colName] ?? []);
      if (current.has(value)) {
        current.delete(value);
      } else {
        current.add(value);
      }
      return { ...prev, [colName]: current };
    });
  };

  // ── Apply filters (POST /dashboard-update) ───────────────────────────────
  const applyFilters = async () => {
    setUpdating(true);
    setError(null);
    try {
      // Convert Set → Array for API
      const filtersPayload: Record<string, string[]> = {};
      for (const [col, vals] of Object.entries(activeFilters)) {
        if (vals.size > 0) filtersPayload[col] = Array.from(vals);
      }
      const res = await updateDashboard(filtersPayload);
      setKpiValues(res.kpi_values ?? []);
      setCharts(res.dashboard_charts ?? []);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      setError(err?.response?.data?.detail ?? "Failed to apply filters.");
    } finally {
      setUpdating(false);
    }
  };

  // ── Clear all filters ────────────────────────────────────────────────────
  const clearFilters = () => {
    setActiveFilters({});
  };

// ── NEW: Download PDF Handler (Ultimate Crop Fix) ────────────────────────
const handleDownloadPdf = async () => {
  if (!dashboardRef.current) return;
  setIsDownloadingPdf(true);
  
  try {
    // 1. Give the React UI 100ms to settle before taking the picture
    await new Promise(resolve => setTimeout(resolve, 100));

    const element = dashboardRef.current;
    
    // 2. The Ultimate html2canvas Configuration
    const canvas = await html2canvas(element, { 
      scale: 1.5, 
      backgroundColor: "#0f1629", 
      useCORS: true,
      // Force the capture height to be exactly the height of the content
      height: element.scrollHeight,
      windowHeight: element.scrollHeight,
      // Offset the camera by exactly how far you have scrolled down
      scrollY: -window.scrollY 
    });

    const dashboardImageB64 = canvas.toDataURL("image/png");

    // 3. Send to FastAPI
    const response = await fetch("http://localhost:8003/download-pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dashboard_screenshot: dashboardImageB64 })
    });

    if (!response.ok) throw new Error("Failed to generate PDF on backend");

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Data_Analysis_Report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    
  } catch (err) {
    console.error(err);
    alert("Failed to download the PDF report.");
  } finally {
    setIsDownloadingPdf(false);
  }
};

  const totalActiveFilters = Object.values(activeFilters).reduce(
    (sum, s) => sum + s.size,
    0
  );

  // ── Render ────────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-violet-400" />
        <p className="text-sm text-slate-400">Loading dashboard...</p>
      </div>
    );
  }

  if (error && kpiValues.length === 0 && charts.length === 0) {
    return (
      <div className="max-w-2xl mx-auto mt-16 space-y-4">
        <div className="flex items-start gap-3 rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-100">
          <AlertTriangle className="h-5 w-5 mt-0.5 shrink-0" />
          <div>
            <div className="font-semibold mb-1">Dashboard not ready</div>
            <div className="text-xs text-red-300">{error}</div>
          </div>
        </div>
        <button
          onClick={() => router.push("/")}
          className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Go back to upload &amp; analysis
        </button>
      </div>
    );
  }

  return (
    // Added ref={dashboardRef} here to capture the entire layout
    <div className="space-y-6 max-w-6xl mx-auto" ref={dashboardRef}>

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-violet-400" />
            Interactive KPI Dashboard
          </h1>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Powered by R · ggplot2 · Plotly · Gemini
          </p>
        </div>
        <div className="flex items-center gap-3">
          
          {/* ── NEW: Download PDF Button ── */}
          <button
            onClick={handleDownloadPdf}
            disabled={isDownloadingPdf}
            className="inline-flex items-center gap-1.5 rounded-lg border border-violet-700 bg-violet-900/50 px-3 py-1.5 text-xs text-violet-200 hover:text-white hover:bg-violet-800 transition-all disabled:opacity-50"
          >
            {isDownloadingPdf ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Download className="h-3.5 w-3.5" />
            )}
            {isDownloadingPdf ? "Generating PDF..." : "Download PDF"}
          </button>

          <button
            onClick={loadConfig}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 hover:text-slate-100 hover:border-slate-600 transition-all"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button
            onClick={() => router.push("/")}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 hover:text-slate-100 hover:border-slate-600 transition-all"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>
        </div>
      </div>

      {/* ── KPI Cards ── */}
      {kpiValues.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            📈 Key Performance Indicators
          </h2>
          <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
            {kpiValues.map((kpi, idx) => (
              <div
                key={idx}
                className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 space-y-1 animate-in fade-in slide-in-from-bottom-2"
                style={{ animationDelay: `${idx * 60}ms` }}
              >
                <div className="text-[11px] text-slate-400 truncate">{kpi.label}</div>
                <div className="text-xl font-bold text-violet-300 tabular-nums">
                  {formatKpiValue(kpi.value, kpi.fmt)}
                </div>
                <div className="text-[10px] text-slate-600">
                  {kpi.agg} · {kpi.column}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Slicers ── */}
      {slicers.length > 0 && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 space-y-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSlicersOpen((o) => !o)}
              className="flex items-center gap-2 text-sm font-semibold text-slate-100 hover:text-slate-50 transition-colors"
            >
              <SlidersHorizontal className="h-4 w-4 text-violet-400" />
              Slicers
              {totalActiveFilters > 0 && (
                <span className="rounded-full bg-violet-600/30 border border-violet-500/40 px-2 py-0.5 text-[10px] text-violet-300">
                  {totalActiveFilters} active
                </span>
              )}
            </button>
            <div className="flex items-center gap-2">
              {totalActiveFilters > 0 && (
                <button
                  onClick={clearFilters}
                  className="text-[11px] text-slate-400 hover:text-red-400 transition-colors"
                >
                  Clear all
                </button>
              )}
              <button
                onClick={applyFilters}
                disabled={updating}
                className="inline-flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {updating ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : null}
                {updating ? "Applying..." : "Apply filters"}
              </button>
            </div>
          </div>

          {slicersOpen && (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 animate-in fade-in slide-in-from-top-2 duration-300">
              {slicers.map((slicer) => {
                const selected = activeFilters[slicer.name] ?? new Set<string>();
                return (
                  <div key={slicer.name} className="space-y-2">
                    <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                      {slicer.name.replace(/_/g, " ")}
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-2 max-h-40 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
                      {slicer.values.map((val) => {
                        const isSelected = selected.has(val);
                        return (
                          <button
                            key={val}
                            onClick={() => toggleFilter(slicer.name, val)}
                            className={`w-full text-left px-2 py-1.5 rounded-lg text-[11px] transition-all flex items-center gap-2 ${
                              isSelected
                                ? "bg-violet-600/20 text-violet-200 border border-violet-500/30"
                                : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-transparent"
                            }`}
                          >
                            <span
                              className={`h-3.5 w-3.5 rounded border flex items-center justify-center shrink-0 ${
                                isSelected
                                  ? "bg-violet-600 border-violet-500"
                                  : "border-slate-600"
                              }`}
                            >
                              {isSelected && (
                                <svg className="h-2 w-2 text-white" viewBox="0 0 12 12" fill="none">
                                  <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                              )}
                            </span>
                            <span className="truncate">{val}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 text-xs text-red-300">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </section>
      )}

      {/* ── Interactive Plotly Charts ── */}
      {charts.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Dashboard Visuals
          </h2>
          <div className="grid gap-4 md:grid-cols-2">
            {charts.map((chart, idx) => {
              const parsed = parsePlotlyJson(chart.plotly_json);
              return (
                <div
                  // ✅ UPDATE THIS KEY: Add totalActiveFilters so it remounts when slicers change
                  key={`chart-${idx}-filters-${totalActiveFilters}`}
                  className="rounded-2xl border border-slate-800 bg-slate-900/70 p-3 animate-in fade-in zoom-in-95 duration-500"
                  style={{ animationDelay: `${idx * 80}ms` }}
                >
                  {parsed ? (
                    <Plot
                      data={parsed.data}
                      layout={parsed.layout}
                      config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                      style={{ width: "100%", height: "340px" }}
                      useResizeHandler
                    />
                  ) : (
                    <div className="flex items-center justify-center h-[340px] text-xs text-slate-500 flex-col gap-2">
                      <span>⚠️ Could not render interactive chart</span>
                      <span className="text-[10px]">Chart {idx + 1} — JSON parse failed</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Fallback: PNG Charts from backend ── */}
      {charts.length === 0 && pngCharts.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Dashboard Visuals (Static)
          </h2>
          <div className="grid gap-4 md:grid-cols-2">
            {pngCharts.map((c, idx) => (
              <div
                key={idx}
                className="rounded-2xl border border-slate-800 bg-slate-900/70 p-3"
              >
                <img
                  src={`data:image/png;base64,${c.image_b64}`}
                  alt={c.filename}
                  className="w-full rounded-md"
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Empty state ── */}
      {charts.length === 0 && pngCharts.length === 0 && !loading && (
        <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/40 p-12 text-center space-y-2">
          <BarChart3 className="h-8 w-8 text-slate-600 mx-auto" />
          <p className="text-sm text-slate-400">No charts available yet.</p>
          <p className="text-xs text-slate-600">
            Run an analysis from the home page first, then come back here.
          </p>
          <button
            onClick={() => router.push("/")}
            className="mt-2 inline-flex items-center gap-2 text-xs text-violet-400 hover:text-violet-300 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Go to analysis
          </button>
        </div>
      )}
    </div>
  );
}