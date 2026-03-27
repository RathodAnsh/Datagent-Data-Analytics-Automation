"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Script from "next/script";
import { useRouter } from "next/navigation";
import { getDashboardConfig, updateDashboardHtml } from "@/lib/api";
import {
  Loader2,
  SlidersHorizontal,
  BarChart3,
  AlertTriangle,
  ArrowLeft,
  RefreshCw,
  Download,
} from "lucide-react";
import html2canvas from "html2canvas";

// ── PlotlyChart: renders a chart using window.Plotly (loaded via CDN script) ──
function PlotlyChart({ plotlyJson, id }: { plotlyJson: string; id: string }) {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!divRef.current) return;

    const render = () => {
      try {
        const fig = JSON.parse(plotlyJson);
        const orig = fig.layout ?? {};

        // Spread full original layout and patch dark-theme visual properties
        const layout: any = {
          ...orig,
          autosize: true,
          height: 340,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(15,22,41,0.4)", // Matched R dark theme exactly
          font: { color: "#e2e8f0", family: "DM Sans, sans-serif", size: 12, ...(orig.font ?? {}) },
          margin: {
            l: orig.margin?.l ?? 55,
            r: orig.margin?.r ?? 20,
            t: orig.margin?.t ?? 45,
            b: orig.margin?.b ?? 55,
          },
          legend: {
            ...(orig.legend ?? {}),
            bgcolor: "rgba(0,0,0,0)",
            bordercolor: "rgba(42,53,82,0.6)",
            borderwidth: 1,
            font: { color: "#94a3b8", size: 11 },
          },
        };

        // Patch all xaxis/yaxis keys
        Object.keys(orig).forEach((key) => {
          if (key.startsWith("xaxis") || key.startsWith("yaxis")) {
            layout[key] = {
              ...orig[key],
              gridcolor: "#1e2a47",
              zerolinecolor: "#2a3552",
              tickcolor: "#475569",
              color: "#94a3b8",
            };
          }
        });
        if (!layout.xaxis) layout.xaxis = { gridcolor: "#1e2a47", color: "#94a3b8" };
        if (!layout.yaxis) layout.yaxis = { gridcolor: "#1e2a47", color: "#94a3b8" };

        (window as any).Plotly.react(divRef.current, fig.data ?? [], layout, {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d"],
        });
      } catch (e) {
        console.error("Plotly render error", e);
      }
    };

    if ((window as any).Plotly) {
      render();
    } else {
      const interval = setInterval(() => {
        if ((window as any).Plotly) {
          clearInterval(interval);
          render();
        }
      }, 100);
      return () => clearInterval(interval);
    }
  }, [plotlyJson, id]);

  return <div ref={divRef} style={{ width: "100%" }} />;
}

// ── Types ──────────────────────────────────────────────────────────────────
type SlicerConfig = { name: string; values: string[] };
type KpiValue = { label: string; value: number; fmt: string; column: string; agg: string };
type ChartEntry = { title: string; plotly_json: string };

// ── Formatters ─────────────────────────────────────────────────────────────
function formatKpi(value: number, fmt: string): string {
  if (fmt === "currency") {
    if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
    if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
    return `$${value.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  }
  if (fmt === "percentage") return `${(value * 100).toFixed(1)}%`;
  if (value % 1 === 0) return value.toLocaleString("en-US");
  return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── Main Component ─────────────────────────────────────────────────────────
export default function DashboardPage() {
  const router = useRouter();
  const dashboardRef = useRef<HTMLDivElement>(null);

  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plotlyReady, setPlotlyReady] = useState(false);

  const [slicers, setSlicers] = useState<SlicerConfig[]>([]);
  const [kpiValues, setKpiValues] = useState<KpiValue[]>([]);
  const [charts, setCharts] = useState<ChartEntry[]>([]);

  // Preserved Python logic array structure, renamed to activeFilters for UI consistency
  const [activeFilters, setActiveFilters] = useState<Record<string, string[]>>({});
  const [slicersOpen, setSlicersOpen] = useState(true);

  const loadConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const config = await getDashboardConfig();
      setSlicers(config.slicers ?? []);
      setKpiValues(config.kpi_values ?? []);
      setCharts(config.dashboard_charts ?? []);
      // Note: Removed pngCharts fetching entirely
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Failed to load dashboard. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadConfig(); }, [loadConfig]);

  const toggleFilter = (colName: string, value: string) => {
    setActiveFilters((prev) => {
      const current = prev[colName] ?? [];
      const exists = current.includes(value);
      const next = exists ? current.filter((v) => v !== value) : [...current, value];
      return { ...prev, [colName]: next };
    });
  };

  const clearFilters = () => setActiveFilters({});

  const applyFilters = async () => {
    setUpdating(true);
    setError(null);
    try {
      // Passes exactly the same payload structure your Python backend expects
      const res = await updateDashboardHtml(activeFilters);
      if (res.kpi_values) setKpiValues(res.kpi_values);
      if (res.dashboard_charts) setCharts(res.dashboard_charts);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? "Failed to apply filters.");
    } finally {
      setUpdating(false);
    }
  };

  // ── NEW: Ultimate Crop Fix PDF Handler ───────────────────────────────────
  const handleDownloadPdf = async () => {
    if (!dashboardRef.current) return;
    setIsDownloadingPdf(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 100));
      const element = dashboardRef.current;
      
      const canvas = await html2canvas(element, { 
        scale: 1.5, 
        backgroundColor: "#0f1629", 
        useCORS: true,
        height: element.scrollHeight,
        windowHeight: element.scrollHeight,
        scrollY: -window.scrollY 
      });

      const dashboardImageB64 = canvas.toDataURL("image/png");

      const response = await fetch("http://localhost:8002/download-pdf", {
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

  const totalActiveFilters = Object.values(activeFilters).reduce((sum, arr) => sum + arr.length, 0);

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
    <>
      <Script
        src="https://cdn.plot.ly/plotly-2.27.0.min.js"
        strategy="afterInteractive"
        onLoad={() => setPlotlyReady(true)}
      />
      <div className="space-y-6 max-w-6xl mx-auto" ref={dashboardRef}>

        {/* ── Header ── */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-violet-400" />
              Interactive KPI Dashboard
            </h1>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Powered by Python · FastAPI · Plotly · Gemini
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleDownloadPdf}
              disabled={isDownloadingPdf || charts.length === 0}
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
                    {formatKpi(kpi.value, kpi.fmt)}
                  </div>
                  <div className="text-[10px] text-slate-600">
                    {kpi.agg} · {kpi.column.replace(/_/g, " ")}
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
                  {updating && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  {updating ? "Applying..." : "Apply filters"}
                </button>
              </div>
            </div>

            {slicersOpen && (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 animate-in fade-in slide-in-from-top-2 duration-300">
                {slicers.map((slicer) => {
                  const selectedValues = activeFilters[slicer.name] ?? [];
                  return (
                    <div key={slicer.name} className="space-y-2">
                      <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                        {slicer.name.replace(/_/g, " ")}
                      </div>
                      <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-2 max-h-40 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">
                        {slicer.values.map((val) => {
                          const isSelected = selectedValues.includes(val);
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
                                  isSelected ? "bg-violet-600 border-violet-500" : "border-slate-600"
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
              {charts.map((chart, idx) => (
                <div
                  key={`chart-${idx}-${chart.plotly_json.length}-${JSON.stringify(activeFilters)}-${plotlyReady}`}
                  className="rounded-2xl border border-slate-800 bg-slate-900/70 p-3 animate-in fade-in zoom-in-95 duration-500"
                  style={{ animationDelay: `${idx * 80}ms` }}
                >
                  {chart.title && (
                    <p className="text-[11px] font-semibold text-slate-300 mb-2 truncate px-2">
                      {chart.title}
                    </p>
                  )}
                  <PlotlyChart
                    plotlyJson={chart.plotly_json}
                    id={`chart-${idx}-${chart.plotly_json.length}-${JSON.stringify(activeFilters)}-${plotlyReady}`}
                  />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ── Empty state ── */}
        {charts.length === 0 && !loading && (
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
    </>
  );
}