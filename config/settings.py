"""Configuration settings for the RAG Q&A system."""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator
from typing import Optional, Literal
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    KNOWLEDGE_BASE_PATH: Path = DATA_DIR / "knowledge_base.pdf"
    
    # LLM Configuration
    LLM_PROVIDER: Literal["groq", "ollama", "huggingface"] = "groq"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    GROQ_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: Path = PROJECT_ROOT / "chroma_db"
    COLLECTION_NAME: str = "knowledge_base"
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    RELEVANCE_THRESHOLD: float = 0.50  
    MIN_CONFIDENCE_SCORE: float = 0.40

    # Web Search Configuration
    MAX_SEARCH_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT: int = 10
    MAX_WEB_CONTENT_LENGTH: int = 5000
    
    # System Configuration
    LOG_LEVEL: str = "INFO"
    CACHE_ENABLED: bool = True
    MAX_RETRIES: int = 3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @field_validator("VECTOR_STORE_PATH", "DATA_DIR", mode='before')
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @model_validator(mode='after')
    def validate_llm_provider(self):
        """Validate LLM provider configuration."""
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY required when using Groq provider")
        if self.LLM_PROVIDER == "huggingface" and not self.HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY required when using HuggingFace provider")
        return self
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        (self.PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        (self.DATA_DIR / "temp").mkdir(exist_ok=True)
        (self.DATA_DIR / "cache").mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
settings.create_directories()
