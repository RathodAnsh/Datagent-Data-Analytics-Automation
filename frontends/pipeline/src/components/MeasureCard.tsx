import { MeasureSuggestion } from '../types';

interface MeasureCardProps {
  measure: MeasureSuggestion;
  onToggle: (id: string) => void;
}

export default function MeasureCard({ measure, onToggle }: MeasureCardProps) {
  const categoryColors: Record<string, string> = {
    Revenue: 'bg-green-100 text-green-800',
    Customer: 'bg-blue-100 text-blue-800',
    Time: 'bg-orange-100 text-orange-800',
    Performance: 'bg-red-100 text-red-800',
  };

  return (
    <div
      className={`border rounded-lg p-5 transition-all cursor-pointer ${
        measure.selected
          ? 'border-blue-500 bg-blue-50 shadow-md'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
      onClick={() => onToggle(measure.id)}
    >
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          checked={measure.selected}
          onChange={() => onToggle(measure.id)}
          className="mt-1 w-5 h-5 text-blue-600 rounded focus:ring-blue-500"
          onClick={(e) => e.stopPropagation()}
        />
        <div className="flex-1">
          <div className="flex items-start justify-between mb-2">
            <h4 className="font-semibold text-gray-900">{measure.name}</h4>
            <span
              className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                categoryColors[measure.category] || 'bg-gray-100 text-gray-800'
              }`}
            >
              {measure.category}
            </span>
          </div>
          <p className="text-sm text-gray-600 mb-2">{measure.description}</p>
          <div className="bg-gray-50 rounded p-2 mt-2">
            <code className="text-xs text-gray-700 font-mono">{measure.formula}</code>
          </div>
        </div>
      </div>
    </div>
  );
}
