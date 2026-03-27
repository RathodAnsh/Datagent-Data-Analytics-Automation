import { Database, Wand2, CheckCircle2, ArrowRight, AlertTriangle, Loader2, Upload } from 'lucide-react';
import { useState, useRef } from 'react';
import Button from '../components/Button';
import { uploadFile } from '../api/backend';
import { DatasetInfo } from '../types';

const BASE_URL = 'http://localhost:8000';

interface UploadPageProps {
  onUploadComplete:  (data: DatasetInfo) => void;  // → CleaningPage
  onSkipToFeatures?: (data: DatasetInfo) => void;  // → FeaturesPage directly
}

interface CheckResult {
  is_clean: boolean;
  reason:   string;
  stats: {
    total_rows:     number;
    total_cols:     number;
    null_cells:     number;
    null_pct:       number;
    duplicate_rows: number;
    duplicate_pct:  number;
  };
}

type BoxState = 'idle' | 'uploading' | 'checking' | 'done' | 'error';

interface BoxStatus {
  state:       BoxState;
  file:        File | null;
  datasetInfo: DatasetInfo | null;
  checkResult: CheckResult | null;
  message:     string;
}

const EMPTY_BOX: BoxStatus = {
  state: 'idle', file: null, datasetInfo: null, checkResult: null, message: '',
};

// ── Inline drop-zone ──────────────────────────────────────────────────────────
function DropZone({ label, sublabel, accentBorder, accentBg, onFile, disabled }: {
  label: string; sublabel: string; accentBorder: string; accentBg: string;
  onFile: (f: File) => void; disabled?: boolean;
}) {
  const [drag, setDrag] = useState(false);
  const ref = useRef<HTMLInputElement>(null);
  const go  = (f: File) => { if (!disabled) onFile(f); };

  return (
    <div
      onDragEnter={e => { e.preventDefault(); if (!disabled) setDrag(true);  }}
      onDragLeave={e => { e.preventDefault(); setDrag(false); }}
      onDragOver ={e => { e.preventDefault(); }}
      onDrop     ={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) go(f); }}
      onClick    ={() => { if (!disabled) ref.current?.click(); }}
      className  ={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed
                   p-10 cursor-pointer transition-all min-h-[175px] select-none
                   ${disabled ? 'opacity-50 cursor-not-allowed bg-gray-50 border-gray-200'
                     : drag   ? `${accentBorder} ${accentBg} scale-[1.01]`
                     : `border-gray-300 bg-white hover:${accentBorder} hover:${accentBg}`}`}
    >
      <input ref={ref} type="file" accept=".csv,.xlsx,.xls" className="hidden"
             onChange={e => { const f = e.target.files?.[0]; if (f) go(f); }} />
      <Upload className="w-9 h-9 text-gray-400" />
      <div className="text-center pointer-events-none">
        <p className="font-semibold text-gray-700">{label}</p>
        <p className="text-sm text-gray-400 mt-1">{sublabel}</p>
        <p className="text-xs text-gray-300 mt-1">CSV or Excel · drag & drop or click</p>
      </div>
    </div>
  );
}

// ── Status banner under each box ──────────────────────────────────────────────
function StatusBanner({ box }: { box: BoxStatus }) {
  if (box.state === 'idle') return null;

  if (box.state === 'uploading' || box.state === 'checking') {
    return (
      <div className="flex items-center gap-2 text-blue-600 text-sm font-medium py-1">
        <Loader2 className="w-4 h-4 animate-spin shrink-0" />
        {box.state === 'uploading' ? 'Uploading…' : 'Validating dataset quality…'}
      </div>
    );
  }

  if (box.state === 'error') {
    return (
      <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
        <span>{box.message}</span>
      </div>
    );
  }

  if (box.state === 'done' && box.checkResult) {
    const { is_clean, stats } = box.checkResult;
    return (
      <div className={`rounded-lg border px-4 py-3 text-sm
        ${is_clean ? 'bg-green-50 border-green-200 text-green-800'
                   : 'bg-amber-50 border-amber-200 text-amber-800'}`}>
        <p className="font-semibold flex items-center gap-1.5">
          {is_clean
            ? <CheckCircle2 className="w-4 h-4 text-green-600 shrink-0" />
            : <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />}
          {box.file?.name}
        </p>
        <div className="mt-1.5 flex flex-wrap gap-3 text-xs opacity-70">
          <span>{stats.total_rows.toLocaleString()} rows · {stats.total_cols} cols</span>
          <span>{stats.null_pct}% missing</span>
          <span>{stats.duplicate_pct}% duplicates</span>
        </div>
      </div>
    );
  }
  return null;
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function UploadPage({ onUploadComplete, onSkipToFeatures }: UploadPageProps) {
  const [rawBox,     setRawBox]     = useState<BoxStatus>(EMPTY_BOX);
  const [cleanedBox, setCleanedBox] = useState<BoxStatus>(EMPTY_BOX);

  async function processFile(
    file: File,
    expectedClean: boolean,
    setter: React.Dispatch<React.SetStateAction<BoxStatus>>,
  ) {
    setter(() => ({ ...EMPTY_BOX, state: 'uploading', file }));

    let info: DatasetInfo;
    try {
      info = await uploadFile(file);
    } catch {
      setter(s => ({ ...s, state: 'error', message: 'Upload failed. Please try again.' }));
      return;
    }

    setter(s => ({ ...s, state: 'checking', datasetInfo: info }));

    let check: CheckResult;
    try {
      const res = await fetch(`${BASE_URL}/check-dataset?file_path=${encodeURIComponent((info as any).filePath)}`);
      check = await res.json();
    } catch {
      setter(s => ({ ...s, state: 'error', message: 'Could not verify dataset quality.' }));
      return;
    }

    setter(s => ({ ...s, state: 'done', checkResult: check }));

    // Persist appropriately
    if (!expectedClean && !check.is_clean) {
      localStorage.setItem('uploadedDataset', JSON.stringify(info));
    }
    if (expectedClean && check.is_clean) {
      localStorage.setItem('uploadedDataset', JSON.stringify(info));
      localStorage.setItem('cleanedDataset',  JSON.stringify(info));
    }
  }

  // ── Box 1 CTA ──────────────────────────────────────────────────────────────
  function rawProceed(skipClean: boolean) {
    const { datasetInfo } = rawBox;
    if (!datasetInfo) return;
    localStorage.setItem('uploadedDataset', JSON.stringify(datasetInfo));
    if (skipClean) {
      localStorage.setItem('cleanedDataset', JSON.stringify(datasetInfo));
      onSkipToFeatures?.(datasetInfo);
    } else {
      onUploadComplete(datasetInfo);
    }
  }

  // ── Box 2 CTA ──────────────────────────────────────────────────────────────
  function cleanedProceed(forceClean: boolean) {
    const { datasetInfo } = cleanedBox;
    if (!datasetInfo) return;
    localStorage.setItem('uploadedDataset', JSON.stringify(datasetInfo));
    if (forceClean) {
      onUploadComplete(datasetInfo);
    } else {
      localStorage.setItem('cleanedDataset', JSON.stringify(datasetInfo));
      onSkipToFeatures?.(datasetInfo);
    }
  }

  const rawReady     = rawBox.state     === 'done' && !!rawBox.checkResult;
  const cleanedReady = cleanedBox.state === 'done' && !!cleanedBox.checkResult;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-14 max-w-5xl">

        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-5">
            <div className="bg-blue-600 p-4 rounded-2xl shadow-lg">
              <Database className="w-11 h-11 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Data Cleaning &amp; Feature Selection
          </h1>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            Pick the right box based on your data — we'll validate it
            and route you automatically.
          </p>
        </div>

        {/* Two upload boxes */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">

          {/* ── BOX 1: Raw dataset ── */}
          <div className="bg-white rounded-2xl shadow-lg p-6 flex flex-col gap-4 border-t-4 border-blue-500">
            <div className="flex items-center gap-3">
              <div className="bg-blue-100 p-2 rounded-lg shrink-0">
                <Wand2 className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <h2 className="font-bold text-gray-900">Raw / Uncleaned Dataset</h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  Upload here if your data hasn't been cleaned yet
                </p>
              </div>
            </div>

            <DropZone
              label="Drop your raw dataset"
              sublabel="Missing values, duplicates & outliers will be fixed"
              accentBorder="border-blue-400"
              accentBg="bg-blue-50"
              disabled={rawBox.state === 'uploading' || rawBox.state === 'checking'}
              onFile={f => processFile(f, false, setRawBox)}
            />

            <StatusBanner box={rawBox} />

            {rawReady && (
              rawBox.checkResult!.is_clean ? (
                /* Clean file in raw box */
                <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-sm text-green-800">
                  <p className="font-semibold flex items-center gap-1.5 mb-1">
                    <CheckCircle2 className="w-4 h-4 shrink-0" />
                    This dataset already looks clean!
                  </p>
                  <p className="text-xs text-green-700 mb-3 opacity-80">
                    No significant issues detected. You can skip cleaning and go directly
                    to feature selection, or run the pipeline anyway.
                  </p>
                  <div className="flex gap-2 flex-wrap">
                    <Button variant="success" size="sm"
                            icon={<ArrowRight className="w-4 h-4" />}
                            onClick={() => rawProceed(true)}>
                      Skip to Feature Selection
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => rawProceed(false)}>
                      Clean Anyway
                    </Button>
                  </div>
                </div>
              ) : (
                /* Dirty file — proceed to cleaning */
                <Button variant="primary" size="md" fullWidth
                        icon={<ArrowRight className="w-4 h-4" />}
                        onClick={() => rawProceed(false)}>
                  Proceed to Data Cleaning
                </Button>
              )
            )}
          </div>

          {/* ── BOX 2: Already cleaned ── */}
          <div className="bg-white rounded-2xl shadow-lg p-6 flex flex-col gap-4 border-t-4 border-emerald-500">
            <div className="flex items-center gap-3">
              <div className="bg-emerald-100 p-2 rounded-lg shrink-0">
                <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              </div>
              <div>
                <h2 className="font-bold text-gray-900">Already Cleaned Dataset</h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  Upload here to skip cleaning &amp; go straight to feature selection
                </p>
              </div>
            </div>

            <DropZone
              label="Drop your cleaned dataset"
              sublabel="Will be validated and sent to feature selection"
              accentBorder="border-emerald-400"
              accentBg="bg-emerald-50"
              disabled={cleanedBox.state === 'uploading' || cleanedBox.state === 'checking'}
              onFile={f => processFile(f, true, setCleanedBox)}
            />

            <StatusBanner box={cleanedBox} />

            {cleanedReady && (
              !cleanedBox.checkResult!.is_clean ? (
                /* Dirty file in clean box */
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
                  <p className="font-semibold flex items-center gap-1.5 mb-1">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    This dataset needs cleaning first!
                  </p>
                  <p className="text-xs text-amber-700 mb-3 opacity-80">
                    {cleanedBox.checkResult!.reason}
                  </p>
                  <Button variant="primary" size="sm"
                          icon={<ArrowRight className="w-4 h-4" />}
                          onClick={() => cleanedProceed(true)}>
                    Run Data Cleaning First
                  </Button>
                </div>
              ) : (
                /* Clean — go to features */
                <Button variant="success" size="md" fullWidth
                        icon={<ArrowRight className="w-4 h-4" />}
                        onClick={() => cleanedProceed(false)}>
                  Proceed to Feature Selection
                </Button>
              )
            )}
          </div>
        </div>

        {/* How it works */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {[
            { n: '01', title: 'Upload Dataset',   desc: 'Choose the correct box — raw or pre-cleaned CSV / Excel' },
            { n: '02', title: 'Auto Validate',    desc: 'We instantly check null values, duplicates & data quality' },
            { n: '03', title: 'Clean or Select',  desc: 'AI cleans dirty data, or jump straight to feature selection' },
          ].map(({ n, title, desc }) => (
            <div key={n} className="bg-white rounded-xl shadow-sm p-5">
              <div className="text-blue-600 font-bold text-3xl mb-2">{n}</div>
              <h3 className="font-semibold text-gray-900 mb-1">{title}</h3>
              <p className="text-sm text-gray-500">{desc}</p>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}