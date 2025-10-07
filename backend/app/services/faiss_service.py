"""
FAISS vector store service
"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings
import os


class FAISSService:
    """Service for managing FAISS vector store"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.index_path = settings.FAISS_PERSIST_PATH

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get embeddings instance"""
        return self.embeddings

    def load_index(self) -> FAISS:
        """
        Load existing FAISS index

        Returns:
            FAISS vectorstore instance

        Raises:
            FileNotFoundError: If index doesn't exist
        """
        if not self.index_exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")

        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def index_exists(self) -> bool:
        """Check if FAISS index exists"""
        index_file = os.path.join(self.index_path, "index.faiss")
        return os.path.exists(index_file)

    def get_index_size(self) -> int:
        """
        Get the number of vectors in the index

        Returns:
            Number of vectors in index, or 0 if index doesn't exist
        """
        if not self.index_exists():
            return 0

        try:
            vectorstore = self.load_index()
            return vectorstore.index.ntotal
        except Exception:
            return 0


# Singleton instance
faiss_service = FAISSService()
