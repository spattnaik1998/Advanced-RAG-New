from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Compare API is running", "status": "healthy"}


@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    """API health check"""
    return {"status": "ok", "model": settings.OPENAI_MODEL}
