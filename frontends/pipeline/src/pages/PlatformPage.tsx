import { useState } from 'react';
import { ChevronDown, ChevronUp, Zap, LogOut } from 'lucide-react';

// ── Platform definitions ──────────────────────────────────────────────────────
const PLATFORMS = [
  {
    id:    'python',
    name:  'Python Analysis',
    port:  3001,
    color: 'blue',
    logo:  '🐍',
    bg:    'bg-blue-50 border-blue-200 hover:border-blue-400 hover:bg-blue-100',
    badge: 'bg-blue-100 text-blue-700',
    btn:   'bg-blue-600 hover:bg-blue-700 shadow-blue-200',
    desc:  'AI-powered Python dashboard with KPI slicers, Plotly charts, and business insights powered by Gemini.',
    tags:  ['Plotly', 'Pandas', 'Gemini AI', 'KPI Slicers'],
  },
  {
    id:    'r',
    name:  'R Analysis',
    port:  3002,
    color: 'indigo',
    logo:  '📊',
    bg:    'bg-indigo-50 border-indigo-200 hover:border-indigo-400 hover:bg-indigo-100',
    badge: 'bg-indigo-100 text-indigo-700',
    btn:   'bg-indigo-600 hover:bg-indigo-700 shadow-indigo-200',
    desc:  'Generate stunning ggplot2 charts and interactive R visuals with Gemini-powered narrative insights.',
    tags:  ['ggplot2', 'tidyverse', 'Plotly R', 'Gemini AI'],
  },
  {
    id:    'excel',
    name:  'Excel AI Automator',
    port:  3000,
    color: 'emerald',
    logo:  '📗',
    bg:    'bg-emerald-50 border-emerald-200 hover:border-emerald-400 hover:bg-emerald-100',
    badge: 'bg-emerald-100 text-emerald-700',
    btn:   'bg-emerald-600 hover:bg-emerald-700 shadow-emerald-200',
    desc:  'Generate AI-powered Excel reports with pivot tables, slicers, and Gemini-driven chart automation.',
    tags:  ['Pivot Tables', 'Slicers', 'openpyxl', 'Gemini AI'],
    route: '/dashboard/excel',  // internal Next.js route
  },
  {
    id:    'sql',
    name:  'SQL Analysis',
    port:  3003,
    color: 'orange',
    logo:  '🗄️',
    bg:    'bg-orange-50 border-orange-200 hover:border-orange-400 hover:bg-orange-100',
    badge: 'bg-orange-100 text-orange-700',
    btn:   'bg-orange-600 hover:bg-orange-700 shadow-orange-200',
    desc:  'Connect to MySQL, profile your schema, and run AI-generated optimized SQL queries with business insights.',
    tags:  ['MySQL', 'Gemini AI', 'Schema Profiler', 'Query Builder'],
  },
];

interface PlatformPageProps {
  onBack: () => void;
}

export default function PlatformPage({ onBack }: PlatformPageProps) {
  const [selected,  setSelected]  = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

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

  const handleLaunch = () => {
    if (!selected) return;
    const platform = PLATFORMS.find(p => p.id === selected);
    if (!platform) return;

    setLaunching(true);

    // Read dashboardContext saved by FeaturesPage
    let datasetFilePath  = '';
    let requirements     = '';
    let datasetFileUrl   = '';

    try {
      const ctx = JSON.parse(localStorage.getItem('dashboardContext') ?? '{}');
      datasetFilePath  = ctx.datasetFile     ?? '';
      requirements     = ctx.requirements    ?? '';
      datasetFileUrl   = ctx.datasetFileUrl  ?? '';
    } catch { /* ignore */ }

    // Build URL params to pass context to the target platform
    const params = new URLSearchParams();
    if (datasetFilePath) params.set('dataset_file',   datasetFilePath);
    if (requirements)    params.set('requirements',   requirements);
    if (datasetFileUrl)  params.set('dataset_url',    datasetFileUrl);
    if (userEmail)       params.set('user_email',     userEmail);

    // Excel lives inside the datagent Next.js app at port 3000
    if (platform.id === 'excel') {
      window.open(`http://localhost:3000/dashboard/excel?${params.toString()}`, '_blank');
    } else {
      window.open(`http://localhost:${platform.port}?${params.toString()}`, '_blank');
    }

    setLaunching(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">

      {/* ── Header ── */}
      <header className="bg-white border-b border-slate-200 px-8 h-16 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-xl font-black text-blue-600 tracking-tighter">DATAGENT</span>
          <span className="text-slate-300 text-lg">|</span>
          <div className="flex items-center gap-1.5 text-slate-500 text-sm italic">
            <Zap size={14} className="text-yellow-500 fill-yellow-500" />
            Select Analysis Platform
          </div>
        </div>
        <div className="relative">
          <button
            onClick={() => setProfileOpen(p => !p)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-transparent hover:border-slate-200 hover:bg-slate-50 transition-all"
          >
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-sm font-black">
              {userInitial}
            </div>
            <span className="text-sm font-bold text-slate-700">{userName}</span>
            {profileOpen
              ? <ChevronUp size={14} className="text-slate-400" />
              : <ChevronDown size={14} className="text-slate-400" />}
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

      {/* ── Body ── */}
      <div className="container mx-auto px-4 py-12 max-w-6xl">

        {/* Page title */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-black text-gray-900 mb-3">
            Choose Your Analysis Platform
          </h1>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            Your cleaned dataset and requirements are ready. Select a platform to generate
            AI-powered insights and dashboards.
          </p>
        </div>

        {/* Platform cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
          {PLATFORMS.map(platform => (
            <div
              key={platform.id}
              onClick={() => setSelected(platform.id)}
              className={`relative cursor-pointer rounded-2xl border-2 p-6 transition-all duration-200
                ${selected === platform.id
                  ? `border-blue-500 bg-blue-50 shadow-lg shadow-blue-100 scale-[1.01]`
                  : platform.bg}
              `}
            >
              {/* Selected indicator */}
              {selected === platform.id && (
                <div className="absolute top-4 right-4 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-white text-xs font-black">✓</span>
                </div>
              )}

              {/* Logo + Name */}
              <div className="flex items-center gap-4 mb-4">
                <div className="text-4xl">{platform.logo}</div>
                <div>
                  <h2 className="text-xl font-black text-gray-900">{platform.name}</h2>
                  <div className="flex flex-wrap gap-1.5 mt-1.5">
                    {platform.tags.map(tag => (
                      <span key={tag} className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wide ${platform.badge}`}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Description */}
              <p className="text-sm text-gray-600 leading-relaxed">{platform.desc}</p>
            </div>
          ))}
        </div>

        {/* Action buttons */}
        <div className="flex justify-center gap-4">
          <button
            onClick={onBack}
            className="px-6 py-3 rounded-xl font-bold text-gray-600 border border-gray-200 hover:bg-gray-100 transition-all"
          >
            ← Back to Feature Selection
          </button>
          <button
            onClick={handleLaunch}
            disabled={!selected || launching}
            className="px-8 py-3 rounded-xl font-bold text-white bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-200 transition-all active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {launching ? (
              <>
                <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                Launching...
              </>
            ) : (
              <>
                🚀 Launch {selected ? PLATFORMS.find(p => p.id === selected)?.name : 'Platform'}
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
}