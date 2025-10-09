import { useState, useEffect } from 'react';
import { Clock, Search } from 'lucide-react';

const History = () => {
  const [queries, setQueries] = useState([]);

  useEffect(() => {
    const lastQuery = localStorage.getItem('lastQuery');
    if (lastQuery) {
      setQueries([JSON.parse(lastQuery)]);
    }
  }, []);

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Query History</h1>

      {queries.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <Clock className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No queries yet</h3>
          <p className="text-gray-600">Your query history will appear here</p>
        </div>
      ) : (
        <div className="space-y-4">
          {queries.map((query, idx) => (
            <div key={idx} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center mb-2">
                    <Search className="w-4 h-4 mr-2 text-gray-500" />
                    <h3 className="font-semibold">{query.query}</h3>
                  </div>
                  <p className="text-sm text-gray-600">
                    Pipelines: {query.pipelines_run?.join(', ')}
                  </p>
                </div>
                <span className="text-sm text-gray-500">
                  {query.total_execution_time_ms?.toFixed(0)} ms
                </span>
              </div>

              <div className="grid grid-cols-4 gap-4 text-sm">
                {query.pipelines_run?.map((pipeline) => (
                  <div key={pipeline} className="bg-gray-50 p-3 rounded">
                    <p className="text-gray-600 text-xs mb-1">{pipeline}</p>
                    <p className="font-semibold">{query.metrics?.latencies_ms?.[pipeline]?.toFixed(0)} ms</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default History;
