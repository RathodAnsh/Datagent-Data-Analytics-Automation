import { Check, Loader2 } from 'lucide-react';
import { CleaningStep } from '../types';

interface StepIndicatorProps {
  steps: CleaningStep[];
}

export default function StepIndicator({ steps }: StepIndicatorProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Cleaning Pipeline</h3>
      <div className="space-y-3">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className="flex items-center gap-4 p-3 rounded-lg transition-all"
            style={{
              backgroundColor:
                step.status === 'completed'
                  ? '#f0fdf4'
                  : step.status === 'processing'
                  ? '#eff6ff'
                  : '#f9fafb',
            }}
          >
            <div
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                step.status === 'completed'
                  ? 'bg-green-500'
                  : step.status === 'processing'
                  ? 'bg-blue-500'
                  : 'bg-gray-300'
              }`}
            >
              {step.status === 'completed' ? (
                <Check className="w-5 h-5 text-white" />
              ) : step.status === 'processing' ? (
                <Loader2 className="w-5 h-5 text-white animate-spin" />
              ) : (
                <span className="text-white text-sm font-medium">{index + 1}</span>
              )}
            </div>
            <div className="flex-1">
              <p
                className={`font-medium ${
                  step.status === 'completed'
                    ? 'text-green-900'
                    : step.status === 'processing'
                    ? 'text-blue-900'
                    : 'text-gray-700'
                }`}
              >
                {step.name}
              </p>
            </div>
            {step.status === 'processing' && (
              <span className="text-sm text-blue-600 font-medium">Processing...</span>
            )}
            {step.status === 'completed' && (
              <span className="text-sm text-green-600 font-medium">Completed</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
