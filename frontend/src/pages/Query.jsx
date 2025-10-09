import { useState } from 'react';
import { Search, Download } from 'lucide-react';
import { comparePipelines } from '../services/api';

const Query = () => {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [pipelines, setPipelines] = useState({
    naive_rag: true,
    rrf_fusion: true,
    ensemble: true,
    cohere_compression: false,
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const skipPipelines = Object.keys(pipelines).filter(p => !pipelines[p]);
      const data = await comparePipelines({ query, k, skip_pipelines: skipPipelines });
      setResults(data);
      localStorage.setItem('lastQuery', JSON.stringify(data));
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rag-comparison-${Date.now()}.json`;
    a.click();
  };

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Query & Compare</h1>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Query</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full px-3 py-2 border rounded-md"
              placeholder="Enter your question..."
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Top K: {k}</label>
              <input
                type="range"
                min="1"
                max="10"
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Pipelines</label>
              <div className="space-y-2">
                {Object.keys(pipelines).map((pipeline) => (
                  <label key={pipeline} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={pipelines[pipeline]}
                      onChange={(e) => setPipelines({ ...pipelines, [pipeline]: e.target.checked })}
                      className="mr-2"
                    />
                    <span className="text-sm">{pipeline.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center"
          >
            <Search className="w-4 h-4 mr-2" />
            {loading ? 'Running...' : 'Run Comparison'}
          </button>
        </form>
      </div>

      {results && (
        <>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">Results</h2>
            <button onClick={downloadResults} className="flex items-center px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200">
              <Download className="w-4 h-4 mr-2" />
              Export JSON
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {results.pipelines_run?.map((pipeline) => (
              <div key={pipeline} className="bg-white rounded-lg shadow p-6">
                <h3 className="text-xl font-semibold mb-4 capitalize">{pipeline.replace('_', ' ')}</h3>

                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2">Answer:</p>
                  <p className="text-sm">{results.results[pipeline]?.answer}</p>
                </div>

                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2">Retrieved Documents:</p>
                  {results.results[pipeline]?.retrieved_docs?.slice(0, 3).map((doc, idx) => (
                    <div key={idx} className="text-xs bg-gray-50 p-2 rounded mb-2">
                      <p className="font-mono text-gray-500">ID: {doc.chunk_id.substring(0, 8)}...</p>
                      <p className="mt-1">{doc.snippet}</p>
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600">Latency</p>
                    <p className="font-semibold">{results.metrics?.latencies_ms?.[pipeline]?.toFixed(0)} ms</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Grounding</p>
                    <p className="font-semibold">{(results.metrics?.grounding_percentages?.[pipeline] * 100)?.toFixed(0)}%</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {results.metrics && (
            <div className="mt-6 bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-4">Metrics Summary</h3>
              <p className="text-sm text-gray-600">Total execution time: {results.total_execution_time_ms?.toFixed(0)} ms</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Query;
