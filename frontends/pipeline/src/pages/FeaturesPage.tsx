import { useState } from 'react';
import {
  Brain, Layers, Settings, CheckCircle, Play,
  Download, BarChart2, Filter, Pencil, Pin,
} from 'lucide-react';
import Button    from '../components/Button';
import StatCard  from '../components/StatCard';
import DataTable from '../components/DataTable';
import { DatasetInfo as _DatasetInfo } from '../types';
import { prepareDashboardStream } from '../api/backend';

// Augment DatasetInfo with the columns array the backend always returns.
// types.ts may not declare it, so we extend locally to avoid red-line errors.
type DatasetInfo = _DatasetInfo & { columns?: string[]; previewData?: Record<string, unknown>[] };

// ─── Types ────────────────────────────────────────────────────────────────────

interface FeaturesPageProps { cleanedData: DatasetInfo; onProceedToPlatform: () => void; }

type StageStatus = 'waiting' | 'running' | 'completed' | 'error';
interface Stage {
  id: string; title: string; description: string;
  icon: React.ReactNode; status: StageStatus;
}
interface ColumnInfo {
  column_name: string; data_type: string; unique_count: number;
  null_pct: number; relevance_score: number; role: string;
  relevant: boolean; reasoning: string;
  method_scores: { domain: number; variance: number; univariate: number; rf: number; };
}
interface FinalResult {
  intent: Record<string, boolean>;
  finalDataset: {
    filePath: string; fileName: string; totalRows: number;
    totalColumns: number; columns: string[]; previewData: Record<string, unknown>[];
  };
  columnsKept:    string[];
  columnsDropped: string[];
  profileFile:    string;   // path to dashboard_profile.json on the backend
}

// ─── Stage definitions ────────────────────────────────────────────────────────

const INITIAL_STAGES: Stage[] = [
  { id: 'intent',   title: 'Understanding Business Intent',   description: 'Analysing your requirements to identify dimensions and metrics',                  icon: <Brain className="w-6 h-6" />,       status: 'waiting' },
  { id: 'columns',  title: 'Selecting Relevant Columns',      description: 'Hybrid scoring: domain rules + variance + Pearson/MI + Random Forest',           icon: <Filter className="w-6 h-6" />,      status: 'waiting' },
  { id: 'apply',    title: 'Applying Feature Selection',      description: 'Filtering columns and deriving time / category features',                        icon: <Settings className="w-6 h-6" />,    status: 'waiting' },
  { id: 'finalize', title: 'Finalising Dashboard Dataset',    description: 'Writing the dashboard-ready CSV',                                                 icon: <CheckCircle className="w-6 h-6" />, status: 'waiting' },
];

// ─── Small helpers ────────────────────────────────────────────────────────────

function scoreBar(score: number): React.ReactNode {
  const pct = Math.round(score * 100);
  const col = pct >= 85 ? 'bg-green-500' : pct >= 60 ? 'bg-blue-400' : 'bg-gray-300';
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${col}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600 w-8">{pct}%</span>
    </div>
  );
}

function intentBadges(intent: Record<string, boolean>): React.ReactNode {
  const active = Object.entries(intent).filter(([, v]) => v).map(([k]) => k);
  if (!active.length) return <span className="text-gray-400 text-xs">none detected</span>;
  return (
    <div className="flex flex-wrap gap-2">
      {active.map(k => (
        <span key={k} className="px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-full text-xs font-medium">
          {k.replace(/_/g, ' ')}
        </span>
      ))}
    </div>
  );
}

const BASE_URL = 'http://localhost:8000';

// ─── Component ────────────────────────────────────────────────────────────────

export default function FeaturesPage({ cleanedData, onProceedToPlatform }: FeaturesPageProps) {
  const [requirements,    setRequirements]    = useState('');
  const [isEditing,       setIsEditing]       = useState(false);
  const [pinnedColumns,   setPinnedColumns]   = useState<Set<string>>(new Set());
  const [started,         setStarted]         = useState(false);
  const [error,           setError]           = useState<string | null>(null);
  const [stages,          setStages]          = useState<Stage[]>(INITIAL_STAGES);
  const [intent,          setIntent]          = useState<Record<string, boolean> | null>(null);
  const [columnRelevance, setColumnRelevance] = useState<Record<string, ColumnInfo> | null>(null);
  const [finalResult,     setFinalResult]     = useState<FinalResult | null>(null);

  const allDone    = started && !isEditing && stages.every(s => s.status === 'completed');
  // columns is always sent by the backend; fall back to previewData keys if missing
  const allColumns: string[] =
    cleanedData.columns ??
    (cleanedData.previewData?.length ? Object.keys(cleanedData.previewData[0]) : []);
  const inputsLocked = started && !isEditing;

  // ─── Pin helpers ──────────────────────────────────────────────────────────────

  function togglePin(col: string) {
    setPinnedColumns(prev => {
      const next = new Set(prev);
      next.has(col) ? next.delete(col) : next.add(col);
      return next;
    });
  }
  function toggleAllPins(check: boolean) {
    setPinnedColumns(check ? new Set(allColumns) : new Set());
  }

  function setStageStatus(id: string, status: StageStatus) {
    setStages(prev => prev.map(s => s.id === id ? { ...s, status } : s));
  }

  // ─── Pipeline ─────────────────────────────────────────────────────────────────

  async function startPreparation() {
    if (!requirements.trim()) { alert('Please describe your business requirements first.'); return; }
    let cleanedFilePath = '';
    try { const stored = localStorage.getItem('cleanedDataset'); cleanedFilePath = stored ? (JSON.parse(stored).filePath ?? '') : ''; } catch { /* ignore */ }
    if (!cleanedFilePath) { alert('Cleaned dataset path not found. Please complete the cleaning step first.'); return; }

    setStarted(true); setIsEditing(false); setError(null);
    setStages(INITIAL_STAGES); setIntent(null); setColumnRelevance(null); setFinalResult(null);

    try {
      await prepareDashboardStream(cleanedFilePath, requirements, Array.from(pinnedColumns), (event) => {
        const step   = event.step   as string | undefined;
        const status = event.status as string | undefined;
        if (step && status) setStageStatus(step, status === 'processing' ? 'running' : 'completed');
        if (event.intent)           setIntent(event.intent as Record<string, boolean>);
        if (event.column_relevance) setColumnRelevance(event.column_relevance as Record<string, ColumnInfo>);
        if (event.result)           setFinalResult(event.result as FinalResult);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStages(prev => prev.map(s => s.status === 'running' ? { ...s, status: 'error' } : s));
    }
  }

  function enterEditMode() {
    setIsEditing(true); setStarted(false);
    setStages(INITIAL_STAGES); setIntent(null);
    setColumnRelevance(null); setFinalResult(null); setError(null);
  }

  // ─── Proceed to Dashboard ─────────────────────────────────────────────────
  const [proceedLoading, setProceedLoading] = useState(false);

  async function handleProceedToDashboard() {
    if (!finalResult) return;
    setProceedLoading(true);
    try {
      const params = new URLSearchParams({ requirements });
      const res = await fetch(`${BASE_URL}/get-dashboard-context?${params}`);
      if (!res.ok) throw new Error(`Failed to fetch dashboard context (${res.status})`);
      const ctx = await res.json();
      if (ctx.error) throw new Error(ctx.error);

      // Persist all three things to localStorage so the next page can read them
      localStorage.setItem('dashboardContext', JSON.stringify({
        datasetProfile:  ctx.dataset_profile,   // full column-level profile JSON
        requirements:    ctx.requirements,       // user's business requirements
        datasetFile:     ctx.dataset_file,       // path to dashboard_dataset.csv
        datasetFileUrl:  ctx.dataset_file_url,   // download URL
      }));

      // Navigate to platform selection page
      onProceedToPlatform();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to prepare dashboard context.');
    } finally {
      setProceedLoading(false);
    }
  }

  // ─── Derived display values ───────────────────────────────────────────────────

  const resultColumns = columnRelevance ? Object.values(columnRelevance) : [];
  const keptCount     = resultColumns.filter(c => c.relevant).length;
  const droppedCount  = resultColumns.filter(c => !c.relevant).length;

  const columnRows = resultColumns.map(c => ({
    Column:          c.column_name,
    Type:            c.data_type,
    'Unique #':      c.unique_count,
    'Null %':        `${c.null_pct}%`,
    Role:            c.role,
    Score:           scoreBar(c.relevance_score),
    Selected:        c.relevant
                       ? <span className="text-green-700 font-medium text-xs">✓ Keep</span>
                       : <span className="text-red-500 text-xs">✗ Drop</span>,
    'Method Scores': c.method_scores
      ? `dom=${(c.method_scores.domain*100).toFixed(0)}% var=${(c.method_scores.variance*100).toFixed(0)}% uni=${(c.method_scores.univariate*100).toFixed(0)}% rf=${(c.method_scores.rf*100).toFixed(0)}%`
      : '—',
    Reason: c.reasoning,
  }));

  // ─── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Feature Selection</h1>
          <p className="text-gray-600">
            Describe what you want to analyse — the pipeline selects relevant columns using
            statistical methods and derives time and category features, just like Power BI's
            data preparation workflow.
          </p>
        </div>

        {/* Cleaned dataset overview */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-gray-500" /> Cleaned Dataset
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="File"    value={cleanedData.fileName} />
            <StatCard label="Rows"    value={cleanedData.totalRows.toLocaleString()} color="green" />
            <StatCard label="Columns" value={String(cleanedData.totalColumns)}       color="orange" />
            <StatCard label="Status"  value="Cleaned ✓"                             color="gray" />
          </div>
        </div>

        {/* ── Business requirements box ── */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-2">Business Requirements</h2>
          <p className="text-gray-500 text-sm mb-4">
            Describe what you want to understand. Example:&nbsp;
            <em>"Analyse revenue and profit by product category and city over time"</em>.
          </p>

          <textarea
            value={requirements}
            onChange={e => setRequirements(e.target.value)}
            disabled={inputsLocked}
            placeholder="Type your requirements here…"
            rows={4}
            className="w-full border border-gray-300 rounded-lg p-4 text-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-400
                       disabled:bg-gray-50 disabled:text-gray-400 resize-none"
          />

          {/* Action row */}
          <div className="mt-3 flex items-center gap-3 flex-wrap">

            {/* Start / re-run */}
            {(!started || isEditing) && (
              <Button onClick={startPreparation} variant="primary" size="lg" icon={<Play className="w-5 h-5" />}>
                {isEditing ? 'Re-run Feature Selection' : 'Start Feature Selection'}
              </Button>
            )}

            {/* Edit button — only after a completed run */}
            {started && !isEditing && (
              <Button onClick={enterEditMode} variant="outline" size="md" icon={<Pencil className="w-4 h-4" />}>
                Edit Requirements
              </Button>
            )}

            {/* Running pulse */}
            {started && !isEditing && !allDone && !error && (
              <span className="text-sm text-blue-600 animate-pulse font-medium">Pipeline running…</span>
            )}
          </div>
        </div>

        {/* ── Pin columns panel ── */}
        <div className={`bg-white rounded-xl shadow-lg p-6 mb-8 transition-opacity duration-200 ${
          inputsLocked ? 'opacity-40 pointer-events-none select-none' : ''
        }`}>
          <div className="flex items-start justify-between mb-1">
            <div>
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <Pin className="w-5 h-5 text-indigo-500" /> Pin Columns to Keep
              </h2>
              <p className="text-gray-500 text-sm mt-1">
                Tick any column you want to <strong>force-keep</strong> in the output,
                regardless of the AI relevance score — useful for IDs, foreign keys,
                or any column critical to your analysis.
              </p>
            </div>
            {allColumns.length > 0 && (
              <div className="flex items-center gap-3 shrink-0 ml-6 mt-1">
                <button onClick={() => toggleAllPins(true)}  className="text-xs text-indigo-600 hover:underline font-medium">Select all</button>
                <span className="text-gray-300 text-xs">|</span>
                <button onClick={() => toggleAllPins(false)} className="text-xs text-gray-400 hover:underline">Clear all</button>
              </div>
            )}
          </div>

          {pinnedColumns.size > 0 && (
            <div className="mt-2 mb-1">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-indigo-100 text-indigo-700 text-xs font-semibold rounded-full">
                <Pin className="w-3 h-3" />
                {pinnedColumns.size} column{pinnedColumns.size > 1 ? 's' : ''} pinned — will always be kept
              </span>
            </div>
          )}

          {allColumns.length === 0 ? (
            <p className="text-gray-400 text-sm mt-3">No columns available.</p>
          ) : (
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
              {allColumns.map(col => {
                const pinned = pinnedColumns.has(col);
                return (
                  <label
                    key={col}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer
                                transition-all text-sm select-none ${
                      pinned
                        ? 'bg-indigo-50 border-indigo-400 text-indigo-800 font-medium shadow-sm'
                        : 'bg-gray-50 border-gray-200 text-gray-700 hover:border-indigo-300 hover:bg-indigo-50/40'
                    }`}
                  >
                    <input type="checkbox" checked={pinned} onChange={() => togglePin(col)}
                           className="w-4 h-4 accent-indigo-600 shrink-0" />
                    <span className="truncate" title={col}>{col.replace(/_/g, ' ')}</span>
                    {pinned && <Pin className="w-3 h-3 shrink-0 text-indigo-400 ml-auto" />}
                  </label>
                );
              })}
            </div>
          )}
        </div>

        {/* Error banner */}
        {error && (
          <div className="bg-red-50 border border-red-300 text-red-700 rounded-xl p-4 mb-8 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Progress cards */}
        {started && !isEditing && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-10">
            {stages.map(stage => (
              <div key={stage.id}
                   className={`rounded-xl border p-5 transition-colors ${
                     stage.status === 'completed' ? 'bg-green-50 border-green-300' :
                     stage.status === 'running'   ? 'bg-blue-50  border-blue-300'  :
                     stage.status === 'error'     ? 'bg-red-50   border-red-300'   :
                     'bg-white border-gray-200'
                   }`}>
                <div className="flex gap-4 items-start">
                  <div className={`p-3 rounded-lg shrink-0 ${
                    stage.status === 'completed' ? 'bg-green-600 text-white' :
                    stage.status === 'running'   ? 'bg-blue-600  text-white animate-pulse' :
                    stage.status === 'error'     ? 'bg-red-500   text-white' :
                    'bg-gray-200 text-gray-500'
                  }`}>
                    {stage.icon}
                  </div>
                  <div className="min-w-0">
                    <p className="font-semibold text-sm">{stage.title}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{stage.description}</p>
                    {stage.id === 'intent' && stage.status === 'completed' && intent && (
                      <div className="mt-2">{intentBadges(intent)}</div>
                    )}
                    <p className="mt-2 text-xs font-medium text-gray-700">
                      {stage.status === 'waiting'   && 'Waiting…'}
                      {stage.status === 'running'   && 'In progress…'}
                      {stage.status === 'completed' && '✓ Completed'}
                      {stage.status === 'error'     && '✗ Failed'}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Results */}
        {allDone && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <StatCard label="Columns Kept"     value={String(keptCount)}          color="green"  />
              <StatCard label="Columns Dropped"  value={String(droppedCount)}       color="orange" />
              <StatCard label="Columns Pinned"   value={String(pinnedColumns.size)} color="blue"   />
              <StatCard label="Total Output Cols" value={String(finalResult?.finalDataset.totalColumns ?? 0)} color="gray" />
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
              <h2 className="text-xl font-semibold mb-1 flex items-center gap-2">
                <Layers className="w-5 h-5 text-gray-500" /> Column Relevance Analysis
              </h2>
              <p className="text-xs text-gray-500 mb-4">
                Scores combine domain heuristic, variance filter, Pearson/MI correlation,
                and Random Forest importance. Pinned columns are always marked Keep.
              </p>
              {columnRows.length > 0
                ? <DataTable data={columnRows} />
                : <p className="text-gray-400 text-sm">No relevance data yet.</p>
              }
            </div>

            {finalResult?.finalDataset?.previewData && (
              <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
                <h2 className="text-xl font-semibold mb-1">Dashboard Dataset Preview</h2>
                <p className="text-xs text-gray-500 mb-4">
                  {finalResult.finalDataset.totalRows.toLocaleString()} rows ×{' '}
                  {finalResult.finalDataset.totalColumns} columns — showing first 5 rows.
                  {finalResult.columnsKept.length > 0 && (
                    <> Kept: <span className="font-medium">{finalResult.columnsKept.join(', ')}</span>.</>
                  )}
                </p>
                <DataTable data={finalResult.finalDataset.previewData as Record<string, unknown>[]} />
              </div>
            )}

            <div className="flex justify-center gap-4 mb-10 flex-wrap">
              {/* Download the final CSV */}
              <Button variant="outline" size="lg" icon={<Download className="w-5 h-5" />}
                onClick={() => {
                  const path = finalResult?.finalDataset?.filePath ?? 'dashboard_dataset.csv';
                  const a = document.createElement('a');
                  a.href = `${BASE_URL}/download?path=${encodeURIComponent(path)}`;
                  a.download = 'dashboard_dataset.csv';
                  document.body.appendChild(a); a.click(); document.body.removeChild(a);
                }}>
                Download Dataset
              </Button>

              {/* Proceed — fetches profile + requirements + file path and saves to localStorage */}
              <Button variant="success" size="lg"
                      loading={proceedLoading}
                      icon={!proceedLoading ? <CheckCircle className="w-5 h-5" /> : undefined}
                      onClick={handleProceedToDashboard}>
                Select Analysis Platform →
              </Button>
            </div>
          </>
        )}

      </div>
    </div>
  );
}