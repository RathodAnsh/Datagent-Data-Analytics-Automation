/**
 * lib/api.ts
 * -----------
 * API client for the R Automation backend (FastAPI).
 * Endpoints are IDENTICAL to Python automation's main.py:
 *   POST /upload-data
 *   POST /analyze
 *   GET  /dashboard-config
 *   POST /dashboard-update
 */

import axios from "axios";

// Backend URL — change this if your FastAPI runs on a different port
const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });

// ── Types ────────────────────────────────────────────────────────────────────

export type UploadResponse = {
  rows: number;
  columns: number;
  logs: string[];
  profile: Record<string, unknown>;
};

export type KpiValue = {
  label: string;
  value: number;
  fmt: string;
  column: string;
  agg: string;
};

export type AnalyzeResponse = {
  summary: string;
  charts: string[];            // base64 PNG strings
  dashboard_ready: boolean;
  last_analysis_code: string;  // shown as "Generated R Code" in the UI
  kpi_values: KpiValue[];
};

export type SlicerMeta = {
  name: string;
  values: string[];
};

export type PlotlyChart = {
  title: string;
  plotly_json: string;
};

export type DashboardConfig = {
  slicers: SlicerMeta[];
  kpi_values: KpiValue[];
  dashboard_charts: PlotlyChart[];
  png_charts: { filename: string; image_b64: string }[];
};

export type DashboardUpdateResponse = {
  kpi_values: KpiValue[];
  dashboard_charts: PlotlyChart[];
};

// ── API Functions ─────────────────────────────────────────────────────────────

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<UploadResponse>("/upload-data", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function analyzeQuestion(question: string): Promise<AnalyzeResponse> {
  const { data } = await api.post<AnalyzeResponse>("/analyze", { question });
  return data;
}

export async function getDashboardConfig(): Promise<DashboardConfig> {
  const { data } = await api.get<DashboardConfig>("/dashboard-config");
  return data;
}

export async function updateDashboard(
  filters: Record<string, string[]>
): Promise<DashboardUpdateResponse> {
  const { data } = await api.post<DashboardUpdateResponse>("/dashboard-update", { filters });
  return data;
}
