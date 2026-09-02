"""LangChain-based retrieval system with BM25, SelfQueryRetriever and Reranker."""
from typing import List, Dict, Any, Optional
import functools
import re

from loguru import logger

# LangChain imports
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.openrouter_reranker import OpenRouterReranker, uses_openrouter_rerank
from src.vector_store import VectorStore
from src.source_allowlist import (
    HONG_KONG_LIVE_ORGS,
    canonicalize_live_org,
    document_is_live,
    filter_retrieval_hits,
    live_orgs_csv,
    plan_live_search,
)
from config import settings

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
    logger.debug("BM25 (rank_bm25) loaded successfully for hybrid search")
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available, BM25 hybrid search disabled")

# SelfQueryRetriever imports - try multiple paths
try:
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
except ImportError:
    try:
        from langchain.retrievers.self_query.base import SelfQueryRetriever
    except ImportError:
        try:
            from langchain_community.retrievers.self_query.base import SelfQueryRetriever
        except ImportError:
            logger.debug("SelfQueryRetriever not available. Using direct vector store search.")
            SelfQueryRetriever = None

try:
    from langchain_classic.chains.query_constructor.base import AttributeInfo
except ImportError:
    try:
        from langchain.chains.query_constructor.base import AttributeInfo
    except ImportError:
        AttributeInfo = None

# CrossEncoderReranker is deprecated in LangChain 1.0, we use Qwen3Reranker directly

# Import custom Qwen3 reranker
try:
    from src.qwen3_reranker import Qwen3Reranker
    QWEN3_RERANKER_AVAILABLE = True
except ImportError as e:
    QWEN3_RERANKER_AVAILABLE = False
    logger.debug(f"Qwen3Reranker not available: {e}")
except Exception as e:
    QWEN3_RERANKER_AVAILABLE = False
    logger.warning(f"Qwen3Reranker import error: {e}")


class BM25Retriever:
    """BM25-based keyword retriever for hybrid search."""

    def __init__(self, documents: List[Document] = None):
        """
        Initialize BM25 retriever.

        Args:
            documents: List of LangChain Documents to index
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []

        if documents:
            self.build_index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build_index(self, documents: List[Document]):
        """Build BM25 index from documents."""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, cannot build index")
            return

        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Built BM25 index with {len(documents)} documents")
        else:
            logger.warning("No documents to index for BM25")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using BM25.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of results with content, metadata, and BM25 scores
        """
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's a match
                doc = self.documents[idx]
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(scores[idx]),
                    'id': doc.metadata.get('chunk_id', doc.metadata.get('id', '')),
                    'retrieval_type': 'bm25'
                })

        logger.debug(f"BM25 returned {len(results)} results for query: {query[:50]}...")
        return filter_retrieval_hits(results)


class AdvancedRetriever:
    """Advanced retriever using BM25, SelfQueryRetriever and Reranker."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: Optional[BaseChatModel] = None,
        use_reranker: bool = None,
        reranker_model: str = None,
        use_hybrid_search: bool = True,
    ):
        """
        Initialize advanced retriever.

        Args:
            vector_store: VectorStore instance
            llm: LLM for SelfQueryRetriever (for query parsing)
            use_reranker: Whether to use reranker (default from settings)
            reranker_model: Reranker model name (default from settings)
            use_hybrid_search: Whether to use BM25 + semantic hybrid search (default: True)
        """
        self.vector_store = vector_store
        self.use_reranker = use_reranker if use_reranker is not None else settings.use_reranker
        self.reranker_model = reranker_model or settings.reranker_model
        self.use_hybrid_search = use_hybrid_search and BM25_AVAILABLE

        # Initialize BM25 retriever for hybrid search
        self.bm25_retriever = None
        if self.use_hybrid_search:
            try:
                # Get all documents from vector store for BM25 indexing
                collection = vector_store.vectorstore._collection
                all_docs_data = collection.get()

                if all_docs_data and all_docs_data.get('documents'):
                    # Convert to LangChain Documents (live allowlist only)
                    docs = []
                    for i, content in enumerate(all_docs_data['documents']):
                        metadata = all_docs_data['metadatas'][i] if all_docs_data.get('metadatas') else {}
                        if not document_is_live(metadata):
                            continue
                        docs.append(Document(page_content=content, metadata=metadata))

                    if docs:
                        self.bm25_retriever = BM25Retriever(docs)
                        logger.info(f"BM25 hybrid search enabled with {len(docs)} documents")
                    else:
                        logger.warning("No allowlisted documents found for BM25 indexing")
                        self.use_hybrid_search = False
                else:
                    logger.warning("No documents found for BM25 indexing")
                    self.use_hybrid_search = False
            except Exception as e:
                logger.warning(f"Failed to initialize BM25: {e}")
                self.use_hybrid_search = False

        # Get base retriever from vector store
        # In LangChain 1.0+, retrievers are created via as_retriever()
        try:
            base_retriever = vector_store.as_retriever(
                search_kwargs={"k": settings.reranker_top_k if self.use_reranker else settings.top_k_retrieval}
            )
        except Exception as e:
            logger.warning(f"Error creating retriever: {e}, will use direct vector store search")
            base_retriever = None

        # Setup SelfQueryRetriever if LLM is provided
        if llm:
            # Define metadata fields for filtering with enhanced structure
            if AttributeInfo is not None:
                metadata_field_info = [
                    AttributeInfo(
                        name="source_org",
                        description=(
                            "The source organization "
                            f"({live_orgs_csv()})"
                        ),
                        type="string",
                    ),
                    AttributeInfo(
                        name="filename",
                        description="The name of the source file",
                        type="string",
                    ),
                    AttributeInfo(
                        name="region",
                        description="The region: 'Hong Kong' (for HKCH and HKSIR sources) or 'Non-Hong Kong' (for other sources)",
                        type="string",
                    ),
                    AttributeInfo(
                        name="procedure_category",
                        description="The procedure category: 'Venous Access', 'Angiogram Related', 'Embolization Related', 'Biopsy Related', 'Pain Injection Relief Related', or 'Other'",
                        type="string",
                    ),
                    AttributeInfo(
                        name="procedure_type",
                        description="The type of procedure (PICC, angioplasty, stent, biopsy, drainage, ablation, etc.)",
                        type="string",
                    ),
                    AttributeInfo(
                        name="related_procedures",
                        description="List of related procedures mentioned in the document (e.g., ['PICC insertion', 'catheter placement'])",
                        type="list[string]",
                    ),
                    AttributeInfo(
                        name="category",
                        description="The phase of care: 'pre-operative' (before procedure), 'perioperative' (during procedure), 'post-operative' (after procedure), or 'general' (overview/general information)",
                        type="string",
                    ),
                    AttributeInfo(
                        name="keywords",
                        description="Important medical keywords or terms extracted from the document (e.g., ['insertion', 'removal', 'complications', 'care'])",
                        type="list[string]",
                    ),
                ]
            else:
                metadata_field_info = []

            document_contents = "Information about pediatric interventional radiology procedures, including procedures, care instructions, complications, and patient education materials."

            if (
                settings.llm_provider == "openrouter"
                or SelfQueryRetriever is None
                or AttributeInfo is None
            ):
                if settings.llm_provider == "openrouter":
                    logger.info("OpenRouter demo: using base retriever (no SelfQuery)")
                else:
                    logger.debug(
                        "SelfQueryRetriever or AttributeInfo not available. "
                        "Using base retriever (LangChain 1.0 pattern)."
                    )
                self.retriever = (
                    base_retriever
                    if base_retriever is not None
                    else vector_store.vectorstore.as_retriever()
                )
            else:
                try:
                    # Optional: Use ChromaTranslator if necessary
                    try:
                        from langchain_community.query_constructors.chroma import ChromaTranslator
                        translator = ChromaTranslator()
                    except ImportError:
                        translator = None

                    kwargs = {
                        "llm": llm,
                        "vectorstore": vector_store.vectorstore,
                        "document_contents": document_contents,
                        "metadata_field_info": metadata_field_info,
                        "verbose": True
                    }
                    if translator:
                        kwargs["structured_query_translator"] = translator

                    self.retriever = SelfQueryRetriever.from_llm(**kwargs)
                    logger.info("Initialized SelfQueryRetriever with enhanced metadata fields (related_procedures, category, keywords)")

                    # Store reference to SelfQueryRetriever before it might be wrapped
                    self_query_retriever = self.retriever

                    # Wrap the _get_relevant_documents method to log structured query
                    original_method = self_query_retriever._get_relevant_documents

                    @functools.wraps(original_method)
                    def wrapped_get_relevant_documents(query: str, *, run_manager=None):
                        """Wrapper to intercept and log structured query during retrieval."""
                        try:
                            # Access the query_constructor (which is a RunnableBinding)
                            query_constructor = self_query_retriever.query_constructor

                            # Prepare input for query_constructor (SelfQueryRetriever passes a dict with 'query')
                            input_data = {"query": query}

                            # Invoke query_constructor to get structured query
                            try:
                                structured_query = query_constructor.invoke(input_data)

                                # Log the structured query
                                logger.info("=" * 80)
                                logger.info("STRUCTURED QUERY OUTPUT:")
                                logger.info(f"Original Query: {query}")
                                logger.info(f"✅ Parsed Query Text: {structured_query.query if hasattr(structured_query, 'query') else 'N/A'}")
                                logger.info(f"✅ Metadata Filters: {structured_query.filter if hasattr(structured_query, 'filter') else 'N/A'}")
                                logger.info(f"✅ Limit: {structured_query.limit if hasattr(structured_query, 'limit') else 'N/A'}")

                                # Show filter details if available
                                if hasattr(structured_query, 'filter') and structured_query.filter:
                                    logger.info("Filter Details:")
                                    import json
                                    try:
                                        if hasattr(structured_query.filter, 'dict'):
                                            filter_dict = structured_query.filter.dict()
                                        elif hasattr(structured_query.filter, 'model_dump'):
                                            filter_dict = structured_query.filter.model_dump()
                                        elif hasattr(structured_query.filter, '__dict__'):
                                            filter_dict = structured_query.filter.__dict__
                                        else:
                                            filter_dict = structured_query.filter

                                        filter_str = json.dumps(filter_dict, indent=2, default=str)
                                        logger.info(filter_str)
                                    except Exception:
                                        logger.info(f"Filter (as string): {str(structured_query.filter)}")
                                else:
                                    logger.info("⚠️  No metadata filters extracted")
                                logger.info("=" * 80)
                            except Exception as e:
                                # If query construction fails, log warning but continue
                                logger.warning(f"Failed to log structured query: {e}")
                                logger.debug(f"Error type: {type(e).__name__}")

                            # Call the original method to perform actual retrieval
                            return original_method(query, run_manager=run_manager)
                        except Exception as e:
                            logger.error(f"Error in wrapped_get_relevant_documents: {e}")
                            logger.exception(e)
                            # Re-raise to let the original error handling work
                            raise

                    # Replace the method
                    self_query_retriever._get_relevant_documents = wrapped_get_relevant_documents

                    # Store reference to SelfQueryRetriever for use when wrapped in ContextualCompressionRetriever
                    self._self_query_retriever = self_query_retriever

                except Exception as e:
                    logger.warning(
                        f"Failed to initialize SelfQueryRetriever: {e}. "
                        "Using base retriever."
                    )
                    self.retriever = (
                        base_retriever
                        if base_retriever is not None
                        else vector_store.vectorstore.as_retriever()
                    )
        else:
            logger.info("No LLM provided, using base retriever without SelfQuery")
            self.retriever = base_retriever if base_retriever is not None else vector_store.vectorstore.as_retriever()

        # Setup reranker if enabled (apply directly, not via ContextualCompressionRetriever)
        self.reranker = None
        if self.use_reranker:
            if uses_openrouter_rerank(self.reranker_model):
                try:
                    self.reranker = OpenRouterReranker(
                        model=self.reranker_model,
                        api_key=settings.openrouter_api_key,
                        base_url=settings.openrouter_api_base,
                        top_n=settings.top_k_reranker,
                    )
                    logger.info(
                        f"OpenRouter reranker {self.reranker_model} "
                        f"(top_n: {settings.top_k_reranker})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenRouter reranker: {e}")
                    self.use_reranker = False
            elif QWEN3_RERANKER_AVAILABLE:
                try:
                    self.reranker = Qwen3Reranker(
                        model_name="Qwen/Qwen3-Reranker-0.6B",
                        top_n=settings.top_k_reranker,
                    )
                    logger.info(f"Qwen3 Reranker initialized (top_n: {settings.top_k_reranker})")
                except Exception as e:
                    logger.warning(f"Failed to initialize Qwen3 Reranker: {e}")
                    self.use_reranker = False
            else:
                logger.warning("No reranker available, disabling reranking")
                self.use_reranker = False

        logger.info(f"Initialized AdvancedRetriever (reranker: {self.use_reranker}, hybrid_bm25: {self.use_hybrid_search})")

    def retrieve(
        self,
                 query: str,
                 k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval with optional reranking.

        Args:
            query: Query text
            k: Number of results to return (default from settings)
            filter_dict: Optional metadata filter (used if SelfQueryRetriever not available)

        Returns:
            List of results with content, metadata, and scores
        """
        k = k or settings.top_k_retrieval

        search_plan = plan_live_search(filter_dict)
        if not search_plan.allowed:
            logger.warning(
                "Rejected retrieval for a source org outside the public live allowlist"
            )
            return []
        filter_dict = search_plan.filter_dict

        logger.debug(f"Retrieving documents for query: {query[:50]}...")

        # Hybrid search: combine BM25 and semantic results
        if self.use_hybrid_search and self.bm25_retriever:
            logger.info("🔍 Using hybrid search (BM25 + semantic)")

            # Get BM25 results
            bm25_results = self.bm25_retriever.search(query, k=k * 2)
            logger.debug(f"BM25 returned {len(bm25_results)} results")

            # Get semantic results
            semantic_results = self.vector_store.similarity_search(
                query=query,
                k=k * 2,
                filter_dict=filter_dict
            )
            logger.debug(f"Semantic search returned {len(semantic_results)} results")

            # Reciprocal Rank Fusion (RRF) to combine results
            rrf_k = 60  # RRF constant
            doc_scores = {}  # id -> (score, result)

            def calculate_boosted_rrf(rank, metadata):
                """Calculate RRF score with a 2.0x boost for local HKCH/HKSIR resources."""
                base_score = 1.0 / (rrf_k + rank + 1)
                org = canonicalize_live_org(metadata.get('source_org'))
                is_local = org in HONG_KONG_LIVE_ORGS or metadata.get('region') == 'Hong Kong'
                return base_score * 2.0 if is_local else base_score

            # Score BM25 results
            for rank, result in enumerate(bm25_results):
                doc_id = result.get('id') or result['content'][:100]
                rrf_score = calculate_boosted_rrf(rank, result.get('metadata', {}))

                if doc_id in doc_scores:
                    doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, result)
                else:
                    result['retrieval_type'] = 'bm25'
                    doc_scores[doc_id] = (rrf_score, result)

            # Score semantic results
            for rank, result in enumerate(semantic_results):
                doc_id = result.get('id') or result['content'][:100]
                rrf_score = calculate_boosted_rrf(rank, result.get('metadata', {}))

                if doc_id in doc_scores:
                    doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, result)
                    doc_scores[doc_id][1]['retrieval_type'] = 'hybrid'  # Mark as found in both
                else:
                    result['retrieval_type'] = 'semantic'
                    doc_scores[doc_id] = (rrf_score, result)

            # Sort by combined RRF score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)

            # Take top k and normalize scores
            results = []
            for doc_id, (rrf_score, result) in sorted_docs[:k]:
                result['score'] = min(1.0, rrf_score * 30)  # Normalize to 0-1 range
                results.append(result)

            logger.info(f"Hybrid search combined into {len(results)} results")

            # Apply reranker if enabled
            if self.use_reranker and self.reranker is not None:
                docs = [Document(page_content=r['content'], metadata=r['metadata']) for r in results]
                logger.info(f"Reranking {len(docs)} documents...")
                reranked_docs = self.reranker.compress_documents(docs, query)
                logger.info(f"After reranking: {len(reranked_docs)} documents")

                # Convert back to result format
                results = []
                for doc in reranked_docs[:k]:
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': 0.9,  # Reranked docs are high quality
                        'id': doc.metadata.get('chunk_id', doc.metadata.get('id', '')),
                        'retrieval_type': 'reranked'
                    })

            logger.info(f"Retrieved {len(results)} documents (hybrid search)")
            return filter_retrieval_hits(results)

        # Fallback: Use LangChain retriever (semantic only)
        if filter_dict and (SelfQueryRetriever is None or not isinstance(self.retriever, SelfQueryRetriever)):
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter_dict=filter_dict
            )
        else:
            try:
                if hasattr(self.retriever, 'get_relevant_documents'):
                    docs = self.retriever.get_relevant_documents(query)
                elif hasattr(self.retriever, 'invoke'):
                    docs = self.retriever.invoke(query)
                else:
                    logger.warning("Retriever doesn't have get_relevant_documents or invoke, using vector store directly")
                    results = self.vector_store.similarity_search(query=query, k=k, filter_dict=filter_dict)
                    return filter_retrieval_hits(results)
            except Exception as e:
                logger.warning(f"Error using retriever: {e}, falling back to direct vector store search")
                logger.exception(e)
                results = self.vector_store.similarity_search(query=query, k=k, filter_dict=filter_dict)
                return filter_retrieval_hits(results)

            # Apply reranker if enabled
            if self.use_reranker and self.reranker is not None:
                logger.info(f"Reranking {len(docs)} documents...")
                docs = self.reranker.compress_documents(docs, query)
                logger.info(f"After reranking: {len(docs)} documents")

            # Convert LangChain Documents to our format
            results = []
            for doc in docs[:k]:
                score = getattr(doc, 'score', None)
                if score is None:
                    score = 0.8

                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score,
                    'id': doc.metadata.get('chunk_id', doc.metadata.get('id', ''))
                })

        logger.info(f"Retrieved {len(results)} documents")
        return filter_retrieval_hits(results)

    def rebuild_index(self):
        """Rebuild the retriever index (placeholder for future implementation)."""
        logger.info("Rebuilding retriever index...")
        # Note: LangChain retrievers automatically use the updated vector store
        pass
