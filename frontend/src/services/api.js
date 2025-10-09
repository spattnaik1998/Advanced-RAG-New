import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Document upload
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// Query endpoints
export const queryNaiveRAG = async (query, k = 5, model = 'gpt-4o-mini') => {
  const response = await api.post('/query', { query, k, model });
  return response.data;
};

export const queryRRFFusion = async (params) => {
  const response = await api.post('/query/rrf_fusion', params);
  return response.data;
};

export const queryEnsemble = async (params) => {
  const response = await api.post('/query/ensemble', params);
  return response.data;
};

export const queryCohereCompression = async (params) => {
  const response = await api.post('/query/cohere_compression', params);
  return response.data;
};

// Comparison endpoint
export const comparePipelines = async (params) => {
  const response = await api.post('/compare', params);
  return response.data;
};

export const getComparison = async (comparisonId) => {
  const response = await api.get(`/compare/${comparisonId}`);
  return response.data;
};

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
