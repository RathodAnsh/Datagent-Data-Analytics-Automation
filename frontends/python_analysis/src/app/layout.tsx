import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Data Analyst Agent",
  description: "Decoupled Next.js frontend for the AI Data Analyst Agent",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-50 antialiased">
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
            <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-9 w-9 rounded-xl bg-violet-600 flex items-center justify-center text-xl">
                  🤖
                </div>
                <div>
                  <div className="font-semibold">AI Data Analyst Agent</div>
                  <div className="text-xs text-slate-400">
                    Python Automation · FastAPI · Next.js
                  </div>
                </div>
              </div>
            </div>
          </header>
          <main className="flex-1">
            <div className="mx-auto max-w-6xl px-4 py-6">{children}</div>
          </main>
          <footer className="border-t border-slate-800 bg-slate-900/80">
            <div className="mx-auto max-w-6xl px-4 py-3 text-center text-xs text-slate-500">
              AI Data Analyst Agent | Python Automation
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

