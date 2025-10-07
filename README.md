# RAG Compare

A full-stack application for comparing different RAG (Retrieval-Augmented Generation) techniques including Simple RAG, Hybrid Search with Reranking, and RAG Fusion.

## 🏗️ Architecture

- **Backend**: Python 3.10+, FastAPI, LangChain, FAISS, SQLite
- **Frontend**: React + Vite, TailwindCSS
- **Deployment**: Docker & Docker Compose

## 📋 Prerequisites

- Python 3.10 or higher
- Node.js 14 or higher
- Docker & Docker Compose (optional)
- OpenAI API Key
- Cohere API Key (optional, for reranking)

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   cd rag-compare
   ```

2. **Set up environment variables**
   ```bash
   cp backend/.env.example backend/.env
   ```

3. **Edit `backend/.env` with your API keys**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   FAISS_PERSIST_PATH=./faiss_index
   METADATA_DB=./metadata.sqlite
   COHERE_API_KEY=your_cohere_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

4. **Start the application**
   ```bash
   docker-compose up --build
   ```

5. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Manual Setup

#### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the backend**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run the development server**
   ```bash
   npm run dev
   ```

## 🔑 Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM

### Optional
- `FAISS_PERSIST_PATH`: Path to persist FAISS index (default: `./faiss_index`)
- `METADATA_DB`: Path to SQLite metadata database (default: `./metadata.sqlite`)
- `COHERE_API_KEY`: Cohere API key for reranking (optional)
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o-mini`)

## 📁 Project Structure

```
rag-compare/
├── backend/
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── core/         # Configuration and settings
│   │   ├── models/       # Pydantic models
│   │   ├── services/     # RAG service implementations
│   │   └── main.py       # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── App.jsx       # Main app component
│   │   └── main.jsx      # Entry point
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── docs/                 # Documentation
├── docker-compose.yml
└── README.md
```

## 🧪 API Endpoints

- `GET /`: Health check
- `GET /api/v1/health`: API health status
- `POST /api/v1/query`: Execute RAG query (to be implemented)
- `POST /api/v1/documents`: Upload documents (to be implemented)

## 🛠️ Development

### Backend Development
```bash
cd backend
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Building for Production
```bash
# Backend
cd backend
docker build -t rag-compare-backend .

# Frontend
cd frontend
npm run build
```

## 📚 RAG Methods

1. **Simple RAG**: Basic vector similarity search
2. **Hybrid Search**: Combines vector search (FAISS) + keyword search (BM25) with Cohere reranking
3. **RAG Fusion**: Query expansion with Reciprocal Rank Fusion

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License
