interface StatCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  color?: 'blue' | 'green' | 'orange' | 'red' | 'gray';
}

export default function StatCard({ label, value, icon, color = 'blue' }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    orange: 'bg-orange-50 text-orange-700 border-orange-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    gray: 'bg-gray-50 text-gray-700 border-gray-200',
  };

  return (
    <div
      className={`rounded-lg border p-6 ${colorClasses[color]}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium opacity-80 mb-1">{label}</p>
          <p className="text-3xl font-bold">{value}</p>
        </div>
        {icon && <div className="ml-4">{icon}</div>}
      </div>
    </div>
  );
}
