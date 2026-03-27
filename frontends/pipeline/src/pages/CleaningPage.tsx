import { useState, useEffect, useRef } from 'react';
import { Download, ArrowRight, FileCheck, AlertCircle, Copy, Trash2, RefreshCcw, Database, ChevronDown, LogOut, Zap } from 'lucide-react';
import Button from '../components/Button';
import DataTable from '../components/DataTable';
import StepIndicator from '../components/StepIndicator';
import StatCard from '../components/StatCard';
import { DatasetInfo, CleaningStep, CleaningSummary } from '../types';

interface CleaningPageProps {
  datasetInfo: DatasetInfo;
  onProceedToFeatures: (cleanedData: DatasetInfo, summary: CleaningSummary) => void;
}

// ── Terminal log window ──────────────────────────────────────────────────────────

type LogKind = 'system' | 'info' | 'success' | 'warn' | 'error';

interface LogLine {
  id:   number;
  kind: LogKind;
  text: string;
  ts:   string;
}

const KIND_COLOR: Record<LogKind, string> = {
  system:  'text-cyan-400',
  info:    'text-gray-300',
  success: 'text-green-400',
  warn:    'text-yellow-400',
  error:   'text-red-400',
};

const KIND_PREFIX: Record<LogKind, string> = {
  system:  '●',
  info:    '·',
  success: '✓',
  warn:    '⚠',
  error:   '✗',
};

function TerminalWindow({ lines }: { lines: LogLine[]; running: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    /* No title bar — label is rendered outside above the panel */
    <div className="relative rounded-xl overflow-hidden border border-gray-700 shadow-xl bg-gray-950"
         style={{ height: '240px' }}>

      {/* Top fade — soft whitish blur melting into dark */}
      <div className="absolute top-0 left-0 right-0 h-14 pointer-events-none z-10"
           style={{ background: 'linear-gradient(to bottom, rgba(55,65,70,0.98) 0%, rgba(40,50,55,0.75) 40%, transparent 100%)' }} />

      {/* Bottom fade — soft whitish blur melting into dark */}
      <div className="absolute bottom-0 left-0 right-0 h-14 pointer-events-none z-10"
           style={{ background: 'linear-gradient(to top, rgba(55,65,70,0.98) 0%, rgba(40,50,55,0.75) 40%, transparent 100%)' }} />

      {/* Scrollable logs */}
      <div ref={scrollRef}
           className="h-full overflow-y-auto px-3 py-3 font-mono text-xs leading-relaxed"
           style={{ scrollbarWidth: 'none' }}>
        {lines.length === 0 ? (
          <p className="text-gray-600 mt-1">Waiting for pipeline to start…</p>
        ) : (
          lines.map(line => (
            <div key={line.id} className="flex gap-2 py-[2px]">
              <span className={`shrink-0 select-none ${KIND_COLOR[line.kind]}`}>
                {KIND_PREFIX[line.kind]}
              </span>
              <span className={`${KIND_COLOR[line.kind]} break-all`}>{line.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default function CleaningPage({ datasetInfo, onProceedToFeatures }: CleaningPageProps) {
  const [cleaningStarted, setCleaningStarted] = useState(false);

  // ── User info ──────────────────────────────────────────────────────────────
  const userEmail   = localStorage.getItem('user_email') ?? '';
  const userName    = userEmail ? userEmail.split('@')[0] : 'User';
  const userInitial = userEmail ? userEmail[0].toUpperCase() : 'U';
  const [profileOpen, setProfileOpen] = useState(false);

  const handleLogout = () => {
    localStorage.removeItem('user_email');
    localStorage.removeItem('user_id');
    window.location.href = 'http://localhost:3000/login';
  };
  const [cleaningComplete, setCleaningComplete] = useState(false);
  const [cleanedData, setCleanedData] = useState<DatasetInfo | null>(null);
  const [summary, setSummary] = useState<CleaningSummary | null>(null);
  const [logLines,    setLogLines]    = useState<LogLine[]>([]);
  const logCounter = useRef(0);

  const pushLog = (text: string, kind: LogKind = 'info') => {
    const now = new Date();
    const ts  = now.toTimeString().slice(0, 8);
    setLogLines(prev => [...prev, { id: logCounter.current++, kind, text, ts }]);
  };

  // Steps aligned with backend execution order
  const [steps, setSteps] = useState<CleaningStep[]>([
    { id: 'profiling', name: 'Profiling Dataset', status: 'pending' },
    { id: 'semantic', name: 'Semantic Understanding', status: 'pending' },
    { id: 'missing', name: 'Detecting Missing Values', status: 'pending' },
    { id: 'outliers', name: 'Detecting Outliers', status: 'pending' },
    { id: 'duplicates', name: 'Detecting Duplicates', status: 'pending' },
    { id: 'invalid', name: 'Detecting Invalid Values', status: 'pending' },
    { id: 'applying', name: 'Applying Intelligent Cleaning', status: 'pending' },
  ]);

  const handleStepUpdate = (stepId: string, status: 'processing' | 'completed') => {
    setSteps((prev) =>
      prev.map((step) => (step.id === stepId ? { ...step, status } : step))
    );
  };

  const startCleaning = async () => {
    setCleaningStarted(true);
    setLogLines([]);
    pushLog('Pipeline started — loading dataset…', 'system');

    // 1. Get the file path from where we saved it during upload
    const storedDataset = JSON.parse(localStorage.getItem('uploadedDataset') || '{}');
    if (!storedDataset.filePath) {
        alert("Error: File path not found.");
        return;
    }

    try {
        // 2. Open the Real-time Stream
        const response = await fetch(`http://localhost:8000/clean-stream?file_path=${storedDataset.filePath}`);
        
        if (!response.body) throw new Error("No response body");
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        // 3. Read the stream chunk by chunk
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            
            // Handle multiple JSON objects arriving in one chunk
            const lines = chunk.split('\n').filter(line => line.trim() !== '');
            
            for (const line of lines) {
                try {
                    const data = JSON.parse(line);
                    
                    // CASE A: Step Update
                    if (data.step) {
                        handleStepUpdate(data.step, data.status);
                        if (data.status === 'processing') {
                            const labels: Record<string, string> = {
                                profiling:  'Starting dataset profiling…',
                                semantic:   'Running semantic analysis…',
                                missing:    'Scanning for missing values…',
                                outliers:   'Detecting outliers…',
                                duplicates: 'Checking for duplicate rows…',
                                invalid:    'Validating data values…',
                                applying:   'Applying cleaning operations…',
                            };
                            if (labels[data.step]) pushLog(labels[data.step], 'system');
                        }
                        if (data.status === 'completed') {
                            const labels: Record<string, string> = {
                                profiling:  'Profiling complete',
                                semantic:   'Semantic mapping complete',
                                missing:    'Missing value scan complete',
                                outliers:   'Outlier detection complete',
                                duplicates: 'Duplicate check complete',
                                invalid:    'Validation complete',
                                applying:   'Cleaning applied successfully',
                            };
                            if (labels[data.step]) pushLog(labels[data.step], 'success');
                        }
                    }

                    // CASE LOG: Terminal log line from backend
                    if (data.type === "log") {
                        pushLog(data.text, (data.kind as LogKind) || 'info');
                    }
                    
                    // CASE B: Final Result
                    if (data.result) {
                        setCleanedData(data.result.cleanedData);
                        setSummary(data.result.summary);
                        localStorage.setItem('cleanedDataset', JSON.stringify(data.result.cleanedData));
                        setCleaningComplete(true);
                    }
                } catch (e) {
                    console.error("Error parsing JSON chunk", e);
                }
            }
        }
    } catch (error) {
        console.error("Cleaning failed:", error);
        alert("An error occurred during cleaning.");
    }
  };

  const handleDownload = () => {
    const stored = JSON.parse(localStorage.getItem('cleanedDataset') || '{}');
    if (!stored.filePath) {
      alert('No cleaned dataset available to download.');
      return;
    }
    const url = `http://localhost:8000/download?path=${encodeURIComponent(stored.filePath)}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = stored.fileName || 'cleaned_dataset.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleProceed = () => {
    if (cleanedData && summary) {
      onProceedToFeatures(cleanedData, summary);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">

      {/* ── Top Header ── */}
      <header className="bg-white border-b border-slate-200 px-8 h-16 flex items-center justify-between sticky top-0 z-50 shadow-sm">

        {/* Left — logo + page label */}
        <div className="flex items-center gap-3">
          <span className="text-xl font-black text-blue-600 tracking-tighter">DATAGENT</span>
          <span className="text-slate-300 text-lg">|</span>
          <div className="flex items-center gap-1.5 text-slate-500 text-sm italic">
            <Zap size={14} className="text-yellow-500 fill-yellow-500" />
            Data Cleaning Workflow
          </div>
        </div>

        {/* Right — user dropdown */}
        <div className="relative">
          <button
            onClick={() => setProfileOpen(p => !p)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-transparent hover:border-slate-200 hover:bg-slate-50 transition-all"
          >
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-sm font-black">
              {userInitial}
            </div>
            <span className="text-sm font-bold text-slate-700">{userName}</span>
            <ChevronDown size={14} className={`text-slate-400 transition-transform ${profileOpen ? 'rotate-180' : ''}`} />
          </button>

          {profileOpen && (
            <div className="absolute right-0 mt-2 w-52 bg-white rounded-2xl shadow-xl border border-slate-100 py-2 z-50">
              <div className="px-4 py-2 border-b border-slate-100 mb-1">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Signed in as</p>
                <p className="text-xs font-bold text-slate-800 truncate">{userEmail}</p>
              </div>
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm font-bold text-red-500 hover:bg-red-50 transition-colors"
              >
                <LogOut size={14} /> Sign Out
              </button>
            </div>
          )}
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Page title */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Data Cleaning Workflow</h1>
            <p className="text-gray-600">Automated data quality assessment and intelligent cleaning</p>
          </div>

          {/* Dataset Overview */}
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Dataset Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <StatCard label="File Name" value={datasetInfo.fileName} icon={<FileCheck className="w-6 h-6" />} color="blue" />
              <StatCard label="Total Rows" value={datasetInfo.totalRows.toLocaleString()} icon={<Database className="w-6 h-6" />} color="green" />
              <StatCard label="Total Columns" value={datasetInfo.totalColumns} icon={<Copy className="w-6 h-6" />} color="orange" />
              <StatCard label="Status" value="Original" icon={<AlertCircle className="w-6 h-6" />} color="gray" />
            </div>
            <DataTable data={datasetInfo.previewData} title="Data Preview (First 10 Rows)" />
          </div>

          {/* Workflow Section */}
          {!cleaningStarted ? (
            <div className="bg-white rounded-xl shadow-lg p-8 text-center mb-8">
              <RefreshCcw className="w-16 h-16 text-blue-600 mx-auto mb-4" />
              <h3 className="text-2xl font-semibold text-gray-900 mb-4">Ready to Clean Your Data</h3>
              <p className="text-gray-600 mb-6">
                Our AI-powered pipeline will automatically detect and fix data quality issues
              </p>
              <Button onClick={startCleaning} variant="primary" size="lg"
                      icon={<ArrowRight className="w-5 h-5" />}>
                Start Data Cleaning
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8 items-center">
              {/* Step indicator — left column */}
              <div className="lg:col-span-3">
                <StepIndicator steps={steps} />
              </div>

              {/* Terminal log window — right column, narrower */}
              <div className="lg:col-span-2">
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-sm font-semibold text-gray-700">Processing Logs</p>
                  {cleaningStarted && !cleaningComplete && (
                    <span className="flex items-center gap-1.5 text-green-600 text-xs font-medium">
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500 animate-ping" />
                      Running
                    </span>
                  )}
                  {cleaningComplete && (
                    <span className="text-gray-400 text-xs">Done</span>
                  )}
                </div>
                <TerminalWindow
                  lines={logLines}
                  running={cleaningStarted && !cleaningComplete}
                />
              </div>
            </div>
          )}

          {/* Results Section */}
          {cleaningComplete && summary && cleanedData && (
            <>
              {/* Cleaning Summary */}
              <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
                <h2 className="text-2xl font-semibold text-gray-900 mb-6">Cleaning Summary</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  <StatCard label="Total Issues" value={summary.totalIssues} color="red" />
                  <StatCard label="Missing Fixed" value={summary.missingValuesFixed} color="green" />
                  <StatCard label="Outliers" value={summary.outliersHandled} color="orange" />
                  <StatCard label="Duplicates" value={summary.duplicatesRemoved} icon={<Trash2 className="w-5 h-5" />} color="blue" />
                  <StatCard label="Recalculated" value={summary.columnsRecalculated} color="green" />
                  <StatCard label="Strategy" value={summary.strategyUsed} color="gray" />
                </div>
              </div>

              {/* Cleaned Data Preview */}
              <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
                <DataTable data={cleanedData.previewData} title="Cleaned Data Preview (First 10 Rows)" />
              </div>

              <div className="flex gap-4 justify-center">
                <Button onClick={handleDownload} variant="outline" size="lg" icon={<Download className="w-5 h-5" />}>
                  Download Cleaned Dataset
                </Button>
                <Button onClick={handleProceed} variant="success" size="lg" icon={<ArrowRight className="w-5 h-5" />}>
                  Proceed to Feature / KPI Creation
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}