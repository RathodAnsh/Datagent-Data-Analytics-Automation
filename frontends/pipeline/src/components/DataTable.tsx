interface DataTableProps {
  data: Record<string, any>[];
  title?: string;
  highlightNew?: boolean;
}

export default function DataTable({ data, title, highlightNew = false }: DataTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
        <p className="text-gray-500">No data to display</p>
      </div>
    );
  }

  const columns = Object.keys(data[0]);
  const newColumns = highlightNew ? columns.slice(-3) : [];

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {title && (
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                    newColumns.includes(column)
                      ? 'text-green-700 bg-green-50'
                      : 'text-gray-500'
                  }`}
                >
                  {column.replace(/_/g, ' ')}
                  {newColumns.includes(column) && (
                    <span className="ml-2 text-xs font-normal text-green-600">NEW</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50">
                {columns.map((column) => (
                  <td
                    key={`${rowIndex}-${column}`}
                    className={`px-6 py-4 whitespace-nowrap text-sm ${
                      newColumns.includes(column)
                        ? 'text-green-900 bg-green-50 font-medium'
                        : 'text-gray-900'
                    }`}
                  >
                    {row[column]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
