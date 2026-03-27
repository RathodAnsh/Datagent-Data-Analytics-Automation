import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// Using the clean, modern Inter font
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI SQL Retail Analyst",
  description: "Connect to MySQL, profile data, and get AI insights.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[#020617] text-slate-300 min-h-screen`}>
        {children}
      </body>
    </html>
  );
}