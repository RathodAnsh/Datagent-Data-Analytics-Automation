export interface UploadedFile {
  name: string;
  size: number;
  type: string;
}

export interface DatasetInfo {
  fileName: string;
  totalRows: number;
  totalColumns: number;
  previewData: Record<string, any>[];
}

export interface CleaningStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed';
}

export interface CleaningSummary {
  totalIssues: number;
  missingValuesFixed: number;
  outliersHandled: number;
  duplicatesRemoved: number;
  columnsRecalculated: number;
  strategyUsed: string;
}

export interface ColumnInfo {
  name: string;
  dataType: string;
  suggestedRole: 'kpi' | 'dimension' | 'filter';
  relevanceScore: number;
  reason: string;
  selected: boolean;
}

export interface MeasureSuggestion {
  id: string;
  name: string;
  formula: string;
  description: string;
  category: string;
  selected: boolean;
}

export type Page = 'upload' | 'cleaning' | 'features' | 'platform';
