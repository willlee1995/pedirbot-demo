"""Embedding generation for documents and queries."""
from typing import List, Union
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        default_headers: dict = None,
    ):
        """
        Initialize OpenAI-compatible embeddings.

        Args:
            model: Model name (default from settings)
            api_key: API key (default from settings)
            base_url: API base URL (default from settings)
            default_headers: Optional extra headers (OpenRouter referer, etc.)
        """
        self.model = model or settings.openai_embedding_model
        client_kwargs = {
            "api_key": api_key or settings.openai_api_key,
            "base_url": base_url or settings.openai_api_base,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)
        self._dimension = None
        logger.info(f"Initialized OpenAI-compatible embeddings with model: {self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with retry logic.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]

            # Cache dimension
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        if self._dimension is None:
            # Generate a test embedding to get dimension
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension


class SentenceTransformerEmbeddings(EmbeddingModel):
    """Sentence Transformer (local) embedding model implementation."""

    def __init__(self, model_name: str = None):
        """
        Initialize Sentence Transformer embeddings.

        Args:
            model_name: Model name (default from settings)
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for this embedding provider. "
                "Install the full local stack with: pip install -r requirements-full.txt"
            )
        self.model_name = model_name or settings.sentence_transformer_model
        logger.info(f"Loading Sentence Transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self._dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        return self._dimension


import requests # Ensure requests is imported if not globally available

class OllamaEmbeddings(EmbeddingModel):
    """Ollama embedding model implementation using direct HTTP API."""

    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize Ollama embeddings.

        Args:
            model: Model name (default from settings)
            base_url: Ollama API base URL (default from settings)
        """
        self.model = model or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_api_base

        # Ensure base_url doesn't end with trailing slash
        self.base_url = self.base_url.rstrip("/")
        self.api_endpoint = f"{self.base_url}/api/embed"

        # Set the Ollama host (for fallback/other tools if needed)
        if self.base_url:
            import os
            os.environ['OLLAMA_HOST'] = self.base_url

        # Check API and get dimension
        try:
            payload = {
                "model": self.model,
                "input": "test",
                "dimensions": 1024 # Instruct Olama to truncate locally generated embedding
            }
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()

            data = response.json()
            # The new API returns {"embeddings": [[...]]}
            test_embedding = data.get("embeddings", [])[0]

            self._dimension = len(test_embedding)
            logger.info(
                f"Initialized Ollama embeddings via API with model: {self.model}")
            logger.info(f"Embedding dimension set to: {self._dimension}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to initialize Ollama embeddings API connection: {e}")
            logger.error(f"Make sure Ollama is running at {self.base_url}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings API: {e}")
            logger.error(
                f"Make sure model '{self.model}' is available. Run: ollama pull {self.model}")
            raise

    def _truncate_text(self, text: str, max_length: int = 2048) -> str:
        """
        Truncate text to fit within model's context window.

        This should rarely happen if chunking is configured correctly.

        Args:
            text: Text to truncate
            max_length: Maximum number of characters (approximate tokens)

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Truncate and add indicator
        truncated = text[:max_length-3] + "..."
        logger.warning(f"⚠️ CHUNK TOO LARGE: Text truncated from {len(text)} to {max_length} chars! "
                    f"Consider reducing MAX_CHUNK_SIZE in .env to avoid data loss.")
        return truncated

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for text in texts:
            try:
                if len(text) > 2048:
                    truncated_text = self._truncate_text(text, max_length=2048)
                else:
                    truncated_text = text

                payload = {
                    "model": self.model,
                    "input": truncated_text,
                    "truncate": True,
                    "dimensions": 1024
                }

                response = requests.post(self.api_endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

                # 'embeddings' is a list of lists of floats
                embeddings.extend(data.get("embeddings", []))

            except Exception as e:
                error_msg = str(e).lower()
                if "context length" in error_msg or "input length" in error_msg:
                    logger.error(f"Context length error for text of {len(text)} chars: {e}")
                    # Try with aggressive truncation as last resort
                    try:
                        very_short_text = self._truncate_text(text, max_length=200)
                        payload = {
                            "model": self.model,
                            "input": very_short_text,
                            "truncate": True,
                            "dimensions": 1024
                        }
                        response = requests.post(self.api_endpoint, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        embeddings.extend(data.get("embeddings", []))

                        logger.warning(f"Recovered with aggressive truncation to 200 chars")
                    except Exception as e2:
                        logger.error(f"Failed even with aggressive truncation: {e2}")
                        raise
                else:
                    logger.error(f"Error generating embedding for text (length: {len(text)}): {e}")
                    raise

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            truncated_text = self._truncate_text(text, max_length=1024)
            payload = {
                "model": self.model,
                "input": truncated_text,
                "truncate": True,
                "dimensions": 1024
            }

            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            return data.get("embeddings", [])[0]

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        return self._dimension


class LMStudioEmbeddings(EmbeddingModel):
    """LM Studio embedding model using OpenAI-compatible API."""

    def __init__(self, model: str = None, base_url: str = None):
        """
        Initialize LM Studio embeddings.

        Args:
            model: Model name (default from settings)
            base_url: LM Studio API base URL (default from settings)
        """
        self.model = model or settings.lmstudio_embedding_model
        self.base_url = base_url or settings.lmstudio_api_base

        # LM Studio uses OpenAI-compatible API
        self.client = OpenAI(
            api_key="lm-studio",  # LM Studio doesn't require real key
            base_url=self.base_url
        )
        self._dimension = None
        logger.info(f"Initialized LM Studio embeddings with model: {self.model}")
        logger.info(f"LM Studio API base: {self.base_url}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]

            # Cache dimension
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings from LM Studio: {e}")
            logger.error(f"Make sure LM Studio is running with an embedding model loaded")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        if self._dimension is None:
            # Generate a test embedding to get dimension
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension


def get_embedding_model(provider: str = None) -> EmbeddingModel:
    """
    Factory function to get the appropriate embedding model.

    Args:
        provider: 'openai', 'openrouter', 'sentence-transformer', 'ollama', or 'lmstudio'

    Returns:
        EmbeddingModel instance
    """
    provider = provider or settings.embedding_provider

    if provider == "openai":
        return OpenAIEmbeddings()
    elif provider == "openrouter":
        return OpenAIEmbeddings(
            model=settings.openrouter_embedding_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_api_base,
            default_headers={
                "HTTP-Referer": "https://pedirbot-demo.streamlit.app",
                "X-Title": "PedIR Bot",
            },
        )
    elif provider == "sentence-transformer":
        return SentenceTransformerEmbeddings()
    elif provider == "ollama":
        return OllamaEmbeddings()
    elif provider == "lmstudio":
        return LMStudioEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
