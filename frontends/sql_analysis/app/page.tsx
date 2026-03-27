"use client";

import { useState } from "react";
import { 
  Database, Server, User, Key, CheckCircle2, 
  Loader2, Play, Sparkles, AlertCircle, TableProperties, Download
} from "lucide-react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

export default function SqlAutomationPage() {
  // ── Application State ──
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Database State ──
  const [creds, setCreds] = useState({ host: "localhost", port: "3306", user: "root", password: "", database: "" });
  const [availableTables, setAvailableTables] = useState<string[]>([]);
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [schemaJson, setSchemaJson] = useState<string>("");

  // ── Analysis State ──
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<any[]>([]);

  // ── API Handlers ──

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      const res = await fetch("http://localhost:8004/api/connect", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(creds),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to connect.");
      setAvailableTables(data.tables);
      setStep(2);
    } catch (err: any) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const handleProfile = async () => {
    if (selectedTables.length === 0) return setError("Please select at least one table to profile.");
    setLoading(true); setError(null);
    try {
      const res = await fetch("http://localhost:8004/api/profile", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ creds, tables: selectedTables }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to profile tables.");
      setSchemaJson(data.schema_json);
      setStep(3);
    } catch (err: any) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const handleAnalyze = async () => {
    if (!question.trim()) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch("http://localhost:8004/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ creds, schema_json: schemaJson, question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Analysis failed.");
      
      setHistory((prev) => [...data.results, ...prev]);
      setQuestion(""); 
    } catch (err: any) {
      setError(err.message);
    } finally { setLoading(false); }
  };

  const toggleTable = (t: string) => {
    setSelectedTables(prev => prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t]);
  };

  // ── PDF Download Handler with Datagent Watermark ──
  const handleDownloadPDF = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    let yOffset = 20;

    // Report Title
    doc.setFontSize(22);
    doc.setTextColor(139, 92, 246); // Violet 600
    doc.setFont("helvetica", "bold");
    doc.text("AI SQL Retail Analysis Report", 14, yOffset);
    yOffset += 15;

    // Reverse history to show oldest questions first chronologically in PDF
    const exportHistory = [...history].reverse();

    exportHistory.forEach((entry, index) => {
      // Add new page if we are too close to the bottom
      if (yOffset > 270) { doc.addPage(); yOffset = 20; }

      // 1. Question
      doc.setFontSize(14);
      doc.setTextColor(15, 23, 42); // Slate 900
      doc.setFont("helvetica", "bold");
      const qText = `Q${index + 1}: ${entry.question}`;
      const splitQ = doc.splitTextToSize(qText, pageWidth - 28);
      doc.text(splitQ, 14, yOffset);
      yOffset += (splitQ.length * 7) + 4;

      // 2. SQL Query
      doc.setFontSize(10);
      doc.setTextColor(5, 150, 105); // Emerald 600
      doc.setFont("courier", "normal");
      const sqlText = `Query: ${entry.sql}`;
      const splitSQL = doc.splitTextToSize(sqlText, pageWidth - 28);
      doc.text(splitSQL, 14, yOffset);
      yOffset += (splitSQL.length * 5) + 6;

      // 3. Output Table
      if (entry.error) {
        doc.setFontSize(10);
        doc.setTextColor(220, 38, 38); // Red 600
        doc.setFont("helvetica", "bold");
        doc.text(`Error: ${entry.error}`, 14, yOffset);
        yOffset += 12;
      } else if (entry.data && entry.data.length > 0) {
        const head = [entry.columns];
        const body = entry.data.map((row: any) => entry.columns.map((col: string) => String(row[col] ?? "—")));

        autoTable(doc, {
          startY: yOffset,
          head: head,
          body: body,
          theme: 'grid',
          styles: { fontSize: 9, cellPadding: 3 },
          headStyles: { fillColor: [139, 92, 246], textColor: 255 }, // Violet Header
          alternateRowStyles: { fillColor: [248, 250, 252] },
          margin: { left: 14, right: 14 },
        });

        yOffset = (doc as any).lastAutoTable.finalY + 8;
      } else {
        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);
        doc.setFont("helvetica", "italic");
        doc.text("No data returned.", 14, yOffset);
        yOffset += 12;
      }

      // 4. AI Explanation
      if (entry.explanation) {
        if (yOffset > 270) { doc.addPage(); yOffset = 20; }
        
        doc.setFontSize(11);
        doc.setTextColor(139, 92, 246); // Violet 600
        doc.setFont("helvetica", "bold");
        doc.text("AI Insight:", 14, yOffset);
        yOffset += 6;

        doc.setFontSize(10);
        doc.setTextColor(51, 65, 85); // Slate 700
        doc.setFont("helvetica", "normal");
        const splitExp = doc.splitTextToSize(entry.explanation, pageWidth - 28);
        doc.text(splitExp, 14, yOffset);
        yOffset += (splitExp.length * 5) + 15; // Extra padding before next question
      } else {
        yOffset += 10;
      }
    });

    // --- NEW: Add "Datagent" Watermark to ALL pages ---
    const totalPages = (doc.internal as any).getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      
      // 1. Large Diagonal Watermark
      doc.setFontSize(80);
      doc.setTextColor(235, 235, 235); // Very light, elegant gray
      doc.setFont("helvetica", "bold");
      doc.text("Datagent", pageWidth / 2, pageHeight / 2, {
        align: "center",
        angle: 45
      });

      // 2. Small Professional Footer
      doc.setFontSize(9);
      doc.setTextColor(150, 150, 150);
      doc.setFont("helvetica", "normal");
      doc.text("Report securely generated by Datagent AI", pageWidth / 2, pageHeight - 10, { 
        align: "center" 
      });
    }

    doc.save("Datagent_SQL_Analysis_Report.pdf");
  };

  // ── UI Render ──
  return (
    <div className="min-h-screen bg-[#020617] text-slate-300 p-6 md:p-12 font-sans selection:bg-violet-500/30">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-6">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-violet-600/20 border border-violet-500/30 flex items-center justify-center shadow-[0_0_15px_rgba(139,92,246,0.2)]">
              <Database className="h-5 w-5 text-violet-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-100 tracking-tight">AI SQL Retail Analyst</h1>
              <p className="text-xs text-slate-500 font-medium mt-0.5">Connect to MySQL • Profile Data • Get AI Insights</p>
            </div>
          </div>
          
          {/* Top Right Actions */}
          {history.length > 0 && step === 3 && (
            <button 
              onClick={handleDownloadPDF}
              className="inline-flex items-center gap-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700 px-4 py-2 text-xs font-semibold text-slate-200 shadow-sm transition-all"
            >
              <Download className="h-4 w-4 text-violet-400" />
              Export Report
            </button>
          )}
        </div>

        {/* Global Error Alert */}
        {error && (
          <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-200 flex items-center gap-3 animate-in fade-in zoom-in-95">
            <AlertCircle className="h-5 w-5 shrink-0 text-red-400" /> 
            <span className="font-medium">{error}</span>
          </div>
        )}

        {/* ==========================================
            STEP 1: DATABASE CONNECTION
        ========================================== */}
        {step === 1 && (
          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 md:p-8 animate-in fade-in slide-in-from-bottom-4 shadow-xl">
            <h2 className="text-base font-semibold text-slate-100 mb-6 flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-violet-600 text-[11px] text-white font-bold">1</span> 
              Database Connection
            </h2>
            <form onSubmit={handleConnect} className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="space-y-1.5">
                <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5"><Server className="h-3 w-3"/> Host</label>
                <input required type="text" value={creds.host} onChange={e => setCreds({...creds, host: e.target.value})} className="w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-2.5 text-sm text-slate-200 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 transition-all" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Port</label>
                <input required type="text" value={creds.port} onChange={e => setCreds({...creds, port: e.target.value})} className="w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-2.5 text-sm text-slate-200 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 transition-all" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5"><Database className="h-3 w-3"/> Database Name</label>
                <input required type="text" value={creds.database} onChange={e => setCreds({...creds, database: e.target.value})} className="w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-2.5 text-sm text-slate-200 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 transition-all" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5"><User className="h-3 w-3"/> User</label>
                <input required type="text" value={creds.user} onChange={e => setCreds({...creds, user: e.target.value})} className="w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-2.5 text-sm text-slate-200 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 transition-all" />
              </div>
              <div className="space-y-1.5 md:col-span-2">
                <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5"><Key className="h-3 w-3"/> Password</label>
                <input type="password" value={creds.password} onChange={e => setCreds({...creds, password: e.target.value})} className="w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-2.5 text-sm text-slate-200 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 transition-all" />
              </div>
              <button type="submit" disabled={loading} className="md:col-span-2 mt-4 inline-flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/20 hover:bg-violet-500 disabled:opacity-50 transition-all active:scale-[0.98]">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Connect & Fetch Tables"}
              </button>
            </form>
          </div>
        )}

        {/* ==========================================
            STEP 2: TABLE SELECTION
        ========================================== */}
        {step === 2 && (
          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 md:p-8 animate-in fade-in slide-in-from-right-8 shadow-xl">
            <h2 className="text-base font-semibold text-slate-100 mb-1 flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-violet-600 text-[11px] text-white font-bold">2</span> 
              Select Tables to Profile
            </h2>
            <p className="text-xs text-slate-400 mb-6 ml-8">Choose the retail tables you want the AI to learn from.</p>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 mb-8">
              {availableTables.map(t => (
                <button key={t} onClick={() => toggleTable(t)} className={`px-4 py-3 rounded-xl text-[13px] font-medium border transition-all text-left flex items-center justify-between ${selectedTables.includes(t) ? "bg-violet-600/20 border-violet-500 text-violet-200 shadow-[0_0_10px_rgba(139,92,246,0.1)]" : "bg-slate-950/50 border-slate-700/50 text-slate-400 hover:border-slate-500 hover:bg-slate-900"}`}>
                  <span className="truncate mr-2">{t}</span>
                  {selectedTables.includes(t) && <CheckCircle2 className="h-4 w-4 shrink-0 text-violet-400" />}
                </button>
              ))}
            </div>

            <button onClick={handleProfile} disabled={loading || selectedTables.length === 0} className="w-full md:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/20 hover:bg-violet-500 disabled:opacity-50 transition-all active:scale-[0.98]">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Profile Data & Prepare AI"}
            </button>
          </div>
        )}

        {/* ==========================================
            STEP 3: ANALYSIS & CHAT INTERFACE
        ========================================== */}
        {step === 3 && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
            
            {/* Input Form */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5 shadow-xl">
              <label className="text-xs font-bold text-violet-300 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-violet-600 text-[10px] text-white">3</span>
                Ask Analysis Questions
              </label>
              <div className="flex flex-col md:flex-row gap-3">
                <textarea 
                  value={question} onChange={e => setQuestion(e.target.value)} 
                  placeholder="E.g., What are our top 5 selling products this month? Also, calculate the total revenue."
                  className="flex-1 rounded-xl border border-slate-700 bg-slate-950/50 p-4 text-sm text-slate-200 resize-none h-24 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 placeholder:text-slate-600 transition-all"
                />
                <button onClick={handleAnalyze} disabled={loading || !question.trim()} className="rounded-xl bg-violet-600 px-8 font-semibold text-white hover:bg-violet-500 disabled:opacity-50 transition-all active:scale-[0.98] flex items-center justify-center shadow-lg shadow-violet-900/20">
                  {loading ? <Loader2 className="h-6 w-6 animate-spin" /> : <Play className="h-6 w-6 fill-current ml-1" />}
                </button>
              </div>
            </div>

            {/* Results Stream */}
            <div className="space-y-8">
              {history.map((entry, idx) => (
                <div key={idx} className="rounded-2xl border border-slate-800 bg-slate-900/30 overflow-hidden shadow-2xl animate-in zoom-in-95 duration-300">
                  
                  {/* Header: User Question */}
                  <div className="bg-slate-800/30 px-6 py-5 border-b border-slate-800 flex items-start gap-4">
                    <div className="mt-0.5 rounded-full bg-violet-500/20 p-1.5 border border-violet-500/30 shadow-[0_0_10px_rgba(139,92,246,0.2)]">
                      <TableProperties className="h-4 w-4 text-violet-400" />
                    </div>
                    <h3 className="text-base font-medium text-slate-100 leading-snug">{entry.question}</h3>
                  </div>

                  <div className="p-6 space-y-5">
                    
                    {/* Expandable SQL Query */}
                    <details className="group">
                      <summary className="text-xs font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-slate-300 list-none flex items-center gap-2 select-none">
                        <span className="text-violet-500 group-open:rotate-90 transition-transform duration-200">▶</span> 
                        View Optimized SQL
                      </summary>
                      <div className="mt-3 rounded-xl bg-slate-950/80 p-4 border border-slate-800 overflow-x-auto shadow-inner">
                        <pre className="text-[13px] text-emerald-400 font-mono leading-relaxed"><code>{entry.sql}</code></pre>
                      </div>
                    </details>

                    {/* Data Table or Error */}
                    {entry.error ? (
                      <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300 flex gap-3 items-start">
                        <AlertCircle className="h-5 w-5 shrink-0" />
                        <span className="leading-relaxed">{entry.error}</span>
                      </div>
                    ) : (
                      <>
                        {/* Data Table */}
                        <div className="rounded-xl border border-slate-800 overflow-hidden bg-slate-950/50 shadow-inner">
                          <div className="overflow-x-auto max-h-[400px] scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                            <table className="min-w-full divide-y divide-slate-800 text-sm text-left">
                              <thead className="bg-slate-900/80 sticky top-0 backdrop-blur-sm z-10">
                                <tr>
                                  {entry.columns.map((c: string) => (
                                    <th key={c} className="px-5 py-3.5 font-bold text-slate-400 uppercase tracking-wider text-[11px] whitespace-nowrap">{c.replace(/_/g, ' ')}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-800/50 bg-transparent">
                                {entry.data.map((row: any, i: number) => (
                                  <tr key={i} className="hover:bg-slate-800/40 transition-colors">
                                    {entry.columns.map((c: string) => {
                                      let val = row[c];
                                      // Formatting for neatness
                                      if (val === null || val === undefined) val = "—";
                                      else if (typeof val === 'number' && !Number.isInteger(val)) val = val.toFixed(2);
                                      return (
                                        <td key={c} className="px-5 py-3 whitespace-nowrap text-slate-300 font-medium">{String(val)}</td>
                                      )
                                    })}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                        
                        {/* Explainable AI Block */}
                        {entry.explanation && (
                          <div className="rounded-xl bg-violet-900/20 border border-violet-500/30 p-5 flex gap-4 items-start mt-2 shadow-[0_0_15px_rgba(139,92,246,0.05)]">
                            <div className="rounded-full bg-violet-500/20 p-1.5 shrink-0 mt-0.5">
                              <Sparkles className="h-4 w-4 text-violet-400" />
                            </div>
                            <div>
                              <p className="text-[11px] font-bold text-violet-400 uppercase tracking-widest mb-1.5">AI Business Insight</p>
                              <p className="text-sm text-slate-200 leading-relaxed">{entry.explanation}</p>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>

          </div>
        )}
      </div>
    </div>
  );
}