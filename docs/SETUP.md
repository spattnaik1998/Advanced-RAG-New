# Setup Guide

## Detailed Setup Instructions

### Environment Variables Setup

Create a `.env` file in the `backend/` directory with the following variables:

```env
# Required - OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# Optional - Paths (defaults shown)
FAISS_PERSIST_PATH=./faiss_index
METADATA_DB=./metadata.sqlite

# Optional - Cohere API Key (for reranking)
COHERE_API_KEY=your-cohere-key-here

# Optional - Model Configuration
OPENAI_MODEL=gpt-4o-mini
```

### Getting API Keys

#### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and add it to your `.env` file

#### Cohere API Key (Optional)
1. Go to https://dashboard.cohere.com/api-keys
2. Sign in or create an account
3. Copy your API key
4. Add it to your `.env` file

### Docker Setup

1. Ensure Docker and Docker Compose are installed
2. Copy environment file: `cp backend/.env.example backend/.env`
3. Add your API keys to `backend/.env`
4. Run: `docker-compose up --build`

### Troubleshooting

**Issue**: Port already in use
- Solution: Change ports in `docker-compose.yml` or stop conflicting services

**Issue**: API key errors
- Solution: Verify your keys are correctly set in `.env` file

**Issue**: Module not found
- Solution: Rebuild containers with `docker-compose up --build`
