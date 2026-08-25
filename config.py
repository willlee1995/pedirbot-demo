"""Configuration management for PedIR RAG Backend."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o"

    # Ollama Configuration
    ollama_api_base: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:9b"  # Default LLM (used for final answer generation)
    ollama_orchestrator_model: str = "qwen3.5:9b"  # Model for orchestration (tool calling)

    # LM Studio Configuration (OpenAI-compatible local API)
    lmstudio_api_base: str = "http://localhost:1234/v1"
    lmstudio_embedding_model: str = "text-embedding-qwen3-embedding-0.6b"  # Common embedding model
    lmstudio_chat_model: str = "qwen/qwen3.5-9b"  # MedGemma 1.5 for SickKids QnA curation (LMStudio)

    # Vector Database Configuration
    chroma_persist_directory: str = "./chroma_db"

    # OpenRouter Configuration (Free/Paid API)
    openrouter_api_key: str = ""
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    # Streamlit test default: OpenRouter :free open-weight stand-in for local GPU
    openrouter_chat_model: str = "nvidia/nemotron-3.5-lightning:free"
    openrouter_metadata_model: str = "nvidia/nemotron-3.5-lightning:free"
    collection_name: str = "pedir_knowledge_base"

    # Hugging Face Inference Endpoint Configuration
    hf_api_key: str = ""
    hf_api_base: str = "https://rmebyko6uyhlruyu.us-east-1.aws.endpoints.huggingface.cloud/v1"
    hf_chat_model: str = "medgemma-1-5-4b-it-dze"

    # LEANN Vector Index Configuration
    leann_persist_directory: str = ".leann/indexes"
    leann_backend: Literal["hnsw", "diskann"] = "hnsw"  # Backend: hnsw (default) or diskann (for large-scale)
    leann_graph_degree: int = 32  # Graph degree for HNSW
    leann_complexity: int = 64  # Build/search complexity
    leann_compact: bool = True  # Use compact storage (97% savings)
    leann_recompute: bool = True  # Enable recomputation (required for compact mode)

    # Document Database Configuration (SQLite for full document storage)
    document_db_path: str = "./document_db.sqlite"

    # Retrieval Configuration
    top_k_retrieval: int = 10
    top_k_reranker: int = 3  # Number of documents to return after reranking
    hybrid_alpha: float = 0.7  # Weight for semantic search (legacy, may be deprecated)

    # LangChain Agent Configuration
    agent_max_iterations: int = 5
    agent_verbose: bool = False
    agent_temperature: float = 0.1

    # LangSmith Configuration
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "pedir-bot"

    # LangChain Reranker Configuration
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default reranker model
    reranker_top_k: int = 10  # Number of documents to rerank (before selecting top_k_reranker)

    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_chunk_size: int = 1500  # Larger chunks for better context preservation
    chunk_overlap: int = 200  # More overlap to prevent context fragmentation
    min_relevance_score: float = 0.1  # Minimum similarity score for retrieval

    # Embedding Model Options
    embedding_provider: Literal["openai",
                                "sentence-transformer", "ollama", "lmstudio"] = "ollama"
    sentence_transformer_model: str = "jinaai/jina-embeddings-v2-base-en"
    ollama_embedding_model: str = "qwen3-embedding:latest"

    # LLM Provider
    llm_provider: Literal["openai", "ollama", "lmstudio", "openrouter", "huggingface"] = "ollama"




settings = Settings()
