import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Data Analyst Agent — R Edition",
  description: "Gemini-powered R data analysis with interactive Plotly dashboards",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-slate-950 text-slate-100 min-h-screen`}>

        {/* ── Navbar ── */}
        <header className="sticky top-0 z-50 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-md">
          <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">

            {/* Logo */}
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="h-7 w-7 rounded-lg bg-violet-600/20 border border-violet-500/30 flex items-center justify-center text-sm">
                🤖
              </div>
              <div>
                <div className="text-sm font-semibold text-slate-100 leading-tight">
                  AI Data Analyst Agent
                </div>
                <div className="text-[10px] text-slate-500 leading-tight">
                  Python Automation · FastAPI · Next.js
                </div>
              </div>
            </Link>

            {/* Nav links */}
            <nav className="flex items-center gap-1">
              <Link
                href="/"
                className="rounded-lg px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 transition-all"
              >
                Home
              </Link>
              <Link
                href="/dashboard"
                className="rounded-lg px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 transition-all"
              >
                Dashboard
              </Link>
            </nav>

            {/* Badge */}
            <div className="flex items-center gap-2">
              <span className="rounded-full border border-violet-500/30 bg-violet-600/10 px-2.5 py-0.5 text-[10px] font-medium text-violet-400">
                R Edition
              </span>
              <span className="rounded-full border border-emerald-500/30 bg-emerald-600/10 px-2.5 py-0.5 text-[10px] font-medium text-emerald-400">
                Gemini 2.5 Flash
              </span>
            </div>
          </div>
        </header>

        {/* ── Page content ── */}
        <main className="max-w-5xl mx-auto px-4 py-8">
          {children}
        </main>

        {/* ── Footer ── */}
        <footer className="border-t border-slate-800/60 mt-16">
          <div className="max-w-5xl mx-auto px-4 py-4 text-center text-[11px] text-slate-600">
            AI Data Analyst Agent | R Edition
          </div>
        </footer>

      </body>
    </html>
  );
}
