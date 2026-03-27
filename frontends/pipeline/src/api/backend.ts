import { DatasetInfo } from '../types';

const BASE_URL = 'http://localhost:8000';

// ─── Upload ───────────────────────────────────────────────────────────────────
export async function uploadFile(file: File): Promise<DatasetInfo> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Failed to upload dataset');
  return res.json();
}

// ─── Dashboard stream ─────────────────────────────────────────────────────────
/**
 * POST /prepare-dashboard-stream
 * Streams JSON-line events. Calls onEvent for each parsed event object.
 *
 * The cleaned file path and business requirements are sent as JSON body
 * (not query params) so they survive URL-length limits on large paths.
 */
export async function prepareDashboardStream(
  cleanedFile: string,
  businessRequirements: string,
  pinnedColumns: string[],
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/prepare-dashboard-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      cleaned_file: cleanedFile,
      business_requirements: businessRequirements,
      pinned_columns: pinnedColumns,
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`Stream failed (${res.status}): ${detail}`);
  }

  if (!res.body) throw new Error('Streaming not supported by this browser');

  const reader  = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let   buffer  = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        onEvent(JSON.parse(line));
      } catch {
        console.warn('Could not parse stream line:', line);
      }
    }
  }
}