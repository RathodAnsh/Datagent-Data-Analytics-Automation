import axios from "axios";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export interface UploadResponse {
  rows: number;
  columns: number;
  logs: string[];
  profile: any;
}

export interface AnalyzeResponse {
  summary: string;
  charts: string[];
  dashboard_ready: boolean;
  last_analysis_code: string;
}

export interface KpiValue {
  label: string;
  value: number;
  fmt: string;       // "currency" | "percentage" | "number"
  column: string;
  agg: string;
}

export interface ChartEntry {
  title: string;
  plotly_json: string;
}

export interface DashboardConfigResponse {
  slicers: { name: string; values: string[] }[];
  kpi_values: KpiValue[];
  dashboard_charts: ChartEntry[];
}

export interface DashboardUpdateResponse {
  kpi_values: KpiValue[];
  dashboard_charts: ChartEntry[];
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await apiClient.post<UploadResponse>(
    "/upload-data",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    }
  );

  return data;
}

export async function analyzeQuestion(
  question: string
): Promise<AnalyzeResponse> {
  const { data } = await apiClient.post<AnalyzeResponse>("/analyze", {
    question,
  });
  return data;
}

export async function getDashboardConfig(): Promise<DashboardConfigResponse> {
  const { data } = await apiClient.get<DashboardConfigResponse>(
    "/dashboard-config"
  );
  return data;
}

export async function updateDashboardHtml(
  filters: Record<string, string[]>
): Promise<DashboardUpdateResponse> {
  const { data } = await apiClient.post<DashboardUpdateResponse>(
    "/dashboard-update",
    { filters }
  );
  return data;
}