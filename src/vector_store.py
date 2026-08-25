"""Vector database management using LangChain ChromaDB integration."""
from typing import List, Dict, Any, Optional
import os
import warnings
import sys
from contextlib import contextmanager

# Disable ChromaDB telemetry to suppress warnings - MUST be set before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Suppress all ChromaDB telemetry warnings
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*CollectionQueryEvent.*")
warnings.filterwarnings("ignore", message=".*capture.*")

# Context manager to suppress ChromaDB telemetry errors from stderr
@contextmanager
def suppress_telemetry_errors():
    """Suppress ChromaDB telemetry errors that are printed to stderr."""
    from io import StringIO
    import sys

    # Save original stderr
    original_stderr = sys.stderr

    # Create a filter that suppresses telemetry messages
    class TelemetryFilter:
        def __init__(self, original):
            self.original = original

        def write(self, message):
            if message and ('telemetry' in message.lower() or
                          'CollectionQueryEvent' in message or
                          'capture() takes' in message):
                return  # Suppress telemetry errors
            self.original.write(message)

        def flush(self):
            self.original.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

    # Replace stderr with filtered version
    sys.stderr = TelemetryFilter(original_stderr)
    try:
        yield
    finally:
        # Restore original stderr
        sys.stderr = original_stderr

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from loguru import logger

from src.document_processor import DocumentChunk
from src.embeddings import EmbeddingModel
from config import settings


class EmbeddingModelWrapper(Embeddings):
    """Wrapper to convert custom EmbeddingModel to LangChain Embeddings interface."""

    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize wrapper with custom embedding model."""
        self.embedding_model = embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embedding_model.embed_query(text)


class VectorStore:
    """Manages vector storage and retrieval using LangChain ChromaDB."""

    def __init__(self,
                 embedding_model: EmbeddingModel,
                 collection_name: str = None,
                 persist_directory: str = None):
        """
        Initialize the vector store.

        Args:
            embedding_model: Embedding model to use
            collection_name: Name of the collection (default from settings)
            persist_directory: Directory to persist the database (default from settings)
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory

        # Wrap embedding model for LangChain compatibility
        langchain_embeddings = EmbeddingModelWrapper(embedding_model)

        # Initialize LangChain Chroma vector store
        # Suppress telemetry errors during initialization
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*telemetry.*")
            warnings.filterwarnings("ignore", message=".*capture.*")
            with suppress_telemetry_errors():
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=langchain_embeddings,
                )

        logger.info(
            f"Initialized vector store with collection: {self.collection_name}")
        logger.info(f"Current collection size: {self.vectorstore._collection.count()}")

    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of documents to process at once
        """
        logger.info(f"Adding {len(chunks)} documents to vector store...")

        # Convert DocumentChunk to LangChain Document
        langchain_docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            langchain_docs.append(doc)

        # Add in batches
        for i in range(0, len(langchain_docs), batch_size):
            batch = langchain_docs[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(langchain_docs)-1)//batch_size + 1}")

            # Use LangChain's add_documents method with IDs
            ids = [chunk.chunk_id for chunk in batch_chunks]

            # Check for duplicates in this batch
            if len(ids) != len(set(ids)):
                logger.warning(f"Found duplicate IDs in batch, making them unique...")
                seen_ids = set()
                unique_ids = []
                for idx, chunk_id in enumerate(ids):
                    if chunk_id in seen_ids:
                        # Make it unique by appending batch index
                        unique_id = f"{chunk_id}_batch{i}_{idx}"
                        unique_ids.append(unique_id)
                    else:
                        unique_ids.append(chunk_id)
                        seen_ids.add(chunk_id)
                ids = unique_ids

            # Use ChromaDB upsert directly to handle duplicates gracefully
            # This will replace existing documents with same IDs
            collection = self.vectorstore._collection

            # Get embeddings for this batch
            texts = [doc.page_content for doc in batch]
            embeddings = self.embedding_model.embed_documents(texts)

            # Extract metadata
            metadatas = [doc.metadata for doc in batch]

            # Use upsert which handles duplicates (replaces existing)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.debug(f"Upserted batch {i//batch_size + 1} ({len(batch)} documents)")

        final_count = self.vectorstore._collection.count()
        logger.info(
            f"Successfully added {len(chunks)} documents. Total count: {final_count}")

    def similarity_search(self,
                          query: str,
                          k: int = None,
                          fetch_k: int = 50,
                          filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform MMR search on the vector store.

        Args:
            query: Query text
            k: Number of results to return (default from settings)
            filter_dict: Optional metadata filter

        Returns:
            List of results with content, metadata, and scores
        """
        k = k or settings.top_k_retrieval

        # Suppress telemetry errors during search
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*telemetry.*")
            warnings.filterwarnings("ignore", message=".*capture.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Also suppress stderr telemetry errors
            with suppress_telemetry_errors():
                # Use LangChain retriever interface
                try:
                    if filter_dict:
                        # Convert filter_dict to ChromaDB format with operators
                        # ChromaDB requires $eq operator for equality checks
                        # For multiple conditions, use $and
                        filter_items = []
                        for key, value in filter_dict.items():
                            filter_items.append({key: {"$eq": value}})

                        if len(filter_items) == 1:
                            # Single condition - use directly
                            where = filter_items[0]
                        elif len(filter_items) > 1:
                            # Multiple conditions - use $and
                            where = {"$and": filter_items}
                        else:
                            where = None

                        if where:
                            results = self.vectorstore.max_marginal_relevance_search(
                                query, k=k, fetch_k=fetch_k, filter=where
                            )
                        else:
                            results = self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
                    else:
                        results = self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
                except Exception as e:
                    # Catch telemetry-related errors (they're harmless)
                    if 'capture' in str(e).lower() or 'telemetry' in str(e).lower():
                        logger.debug(f"Suppressed telemetry error during search")
                        # Retry - the error is just in telemetry, not the actual search
                        if filter_dict:
                            filter_items = []
                            for key, value in filter_dict.items():
                                filter_items.append({key: {"$eq": value}})

                            if len(filter_items) == 1:
                                where = filter_items[0]
                            elif len(filter_items) > 1:
                                where = {"$and": filter_items}
                            else:
                                where = None

                            if where:
                                results = self.vectorstore.max_marginal_relevance_search(
                                    query, k=k, fetch_k=fetch_k, filter=where
                                )
                            else:
                                results = self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
                        else:
                            results = self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
                    else:
                        raise

        # Format results to match expected format
        formatted_results = []
        # max_marginal_relevance_search only returns Documents, no scores
        # We assign a default score of 1.0 (or simulated decay) since MMR handles diversity ranking
        for idx, doc in enumerate(results):
            similarity_score = 1.0 - (idx * 0.01) # Simulated slight decay

            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': similarity_score,
                'id': doc.metadata.get('chunk_id', doc.metadata.get('id', ''))
            })

        return formatted_results

    def as_retriever(self, **kwargs):
        """Get LangChain retriever from vector store."""
        return self.vectorstore.as_retriever(**kwargs)

    def delete_collection(self):
        """Delete the current collection."""
        self.vectorstore.delete_collection()
        logger.warning(f"Deleted collection: {self.collection_name}")

    def reset_collection(self):
        """Reset the collection (delete and recreate)."""
        try:
            # Try to delete the collection
            try:
                self.vectorstore.delete_collection()
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete collection (may not exist): {e}")

            # Get the client and ensure collection is deleted
            import chromadb
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*telemetry.*")
                warnings.filterwarnings("ignore", message=".*capture.*")
                client = chromadb.PersistentClient(path=self.persist_directory)
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"Confirmed deletion of collection: {self.collection_name}")
            except Exception as e:
                logger.debug(f"Collection may not exist: {e}")

            # Small delay to ensure deletion completes
            import time
            time.sleep(0.5)

            # Recreate vectorstore
            langchain_embeddings = EmbeddingModelWrapper(self.embedding_model)
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=langchain_embeddings,
            )

            # Verify it's empty
            count = self.vectorstore._collection.count()
            if count > 0:
                logger.warning(f"Collection still has {count} documents after reset, clearing...")
                # Force clear by deleting all
                self.vectorstore.delete_collection()
                time.sleep(0.5)
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=langchain_embeddings,
                )

            logger.info(f"Reset collection: {self.collection_name} (count: {self.vectorstore._collection.count()})")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.vectorstore._collection.count(),
            "persist_directory": self.persist_directory
        }

    @property
    def collection(self):
        """Access underlying ChromaDB collection (for backward compatibility)."""
        return self.vectorstore._collection
