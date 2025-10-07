import React, { useState } from 'react'

function App() {
  const [status, setStatus] = useState('checking...')

  React.useEffect(() => {
    fetch('/api/v1/health')
      .then(res => res.json())
      .then(data => setStatus(data.status))
      .catch(() => setStatus('error'))
  }, [])

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
          RAG Compare
        </h1>
        <div className="bg-white rounded-lg shadow-lg p-6">
          <p className="text-center text-lg">
            API Status: <span className={`font-semibold ${status === 'ok' ? 'text-green-600' : 'text-red-600'}`}>
              {status}
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
