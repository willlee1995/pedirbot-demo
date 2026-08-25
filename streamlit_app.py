"""
Streamlit Chat Interface for PedIR RAG Chatbot.

Enhanced version with:
- Streaming responses
- Step-by-step progress indicators
- Query decomposition for multi-part questions

Run with: streamlit run streamlit_app.py
"""
from pathlib import Path
import sys
import time

import streamlit as st

# Page config must be the first Streamlit command.
st.set_page_config(
    page_title="PedIR-Bot | HKCH Radiology",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cloud_bootstrap import (
    apply_streamlit_secrets_and_cloud_defaults,
    check_demo_query,
    day_queries_used,
    demo_access_code,
    demo_quota_limits,
    ensure_demo_knowledge_base,
    is_cloud_demo,
    missing_cloud_credentials,
    record_demo_query,
)

apply_streamlit_secrets_and_cloud_defaults()

from src.agentic_rag import create_agentic_rag_graph
from src.rag_pipeline import RAGPipeline
from src.llm import get_langchain_llm
from src.retriever import AdvancedRetriever
from src.vector_store import VectorStore
from src.embeddings import get_embedding_model
from src.conversation_memory import ConversationMemory
from src.openrouter_demo_models import (
    AA_INDEX_NOTE,
    OPENROUTER_DEMO_MODELS,
    default_demo_model_id,
    demo_model_label,
    get_demo_model,
)
from config import settings

# Custom CSS for chat styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.75rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    .stats-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2d5a87;
        margin-bottom: 0.5rem;
    }
    .safety-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .safety-critical {
        background: #f8d7da;
        border: 1px solid #dc3545;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .progress-step {
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        color: #e0e0e0;
    }
    .progress-active {
        background: rgba(33, 150, 243, 0.2);
        border-left: 3px solid #2196f3;
        color: #90caf9;
    }
    .progress-done {
        background: rgba(76, 175, 80, 0.2);
        border-left: 3px solid #4caf50;
        color: #a5d6a7;
    }
    .decomposed-badge {
        background: #e1f5fe;
        color: #01579b;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Processing steps for visual feedback
PROCESSING_STEPS = [
    ("🔍", "Analyzing your question..."),
    ("📚", "Searching knowledge base..."),
    ("📊", "Evaluating document relevance..."),
    ("🔒", "Running safety check..."),
    ("✍️", "Generating response (free models may queue up to a minute)..."),
]


@st.cache_resource
def init_vector_store():
    """Load embeddings and the demo index once (not per chat model)."""
    embedding_model = get_embedding_model()
    vector_store = VectorStore(embedding_model)
    demo_chunks = ensure_demo_knowledge_base(
        vector_store,
        chunk_size=settings.max_chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    stats = vector_store.get_stats()
    stats["demo_chunks_added"] = demo_chunks
    return vector_store, stats


@st.cache_resource
def init_rag_system(chat_model: str):
    """Initialize Agentic RAG for one OpenRouter (or configured) chat model."""
    vector_store, stats = init_vector_store()
    llm_kwargs = {}
    if settings.llm_provider == "openrouter":
        llm_kwargs["model"] = chat_model
    llm = get_langchain_llm(**llm_kwargs)
    retriever = AdvancedRetriever(vector_store, llm=llm)
    graph = create_agentic_rag_graph(
        vector_store,
        orchestrator_llm=llm,
        answer_llm=llm,
    )
    rag_pipeline = RAGPipeline(vector_store, retriever=retriever, graph=graph)
    return rag_pipeline, stats


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=10)
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "show_steps" not in st.session_state:
        st.session_state.show_steps = True
    if "demo_queries_used" not in st.session_state:
        st.session_state.demo_queries_used = 0
    if "demo_unlocked" not in st.session_state:
        st.session_state.demo_unlocked = False


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 PedIR-Bot</h1>
        <p>Hong Kong Children's Hospital - Interventional Radiology Information Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    if is_cloud_demo():
        st.caption(
            "Cloud test: the same Agentic RAG graph as the local stack, with "
            "OpenRouter **free** chat and embedding models as a stand-in for a "
            "hospital GPU box. Answers come from the bundled demo leaflets, "
            "not the full local knowledge base."
        )


def _configured_chat_model() -> str:
    """Return the configured chat model id for the active LLM provider."""
    provider = settings.llm_provider
    if provider == "openrouter":
        return settings.openrouter_chat_model
    if provider == "openai":
        return settings.openai_chat_model
    if provider == "huggingface":
        return settings.hf_chat_model
    if provider == "lmstudio":
        return settings.lmstudio_chat_model
    if provider == "ollama":
        return settings.ollama_chat_model
    raise ValueError(f"Unknown LLM provider: {provider}")


def render_model_picker() -> str:
    """Let testers pick an OpenRouter free open-weight model before RAG init."""
    if settings.llm_provider != "openrouter":
        return _configured_chat_model()

    options = [model.id for model in OPENROUTER_DEMO_MODELS]
    default_id = default_demo_model_id(settings.openrouter_chat_model)
    default_index = options.index(default_id) if default_id in options else 0

    with st.sidebar:
        st.header("🧪 Test model")
        st.caption(
            "Open-weight models on OpenRouter’s **free** endpoint. "
            "Same Agentic RAG graph as a local Ollama/LM Studio run — "
            "Streamlit Cloud just cannot host the GPU."
        )
        selected = st.selectbox(
            "Chat model",
            options=options,
            index=default_index,
            format_func=demo_model_label,
            key="openrouter_demo_chat_model",
            help="Switching model rebuilds the agent graph only. The leaflet index stays cached.",
        )
        model = get_demo_model(selected)
        if model:
            st.markdown(
                f"**{model.short_name}** ({model.lab})  \n"
                f"AA Intelligence Index **{model.aa_index}** · {model.params}  \n"
                f"{model.local_fit}"
            )
            st.caption(model.why)
            st.markdown(
                f"[Artificial Analysis]({model.aa_url}) · "
                f"[OpenRouter free endpoint]({model.openrouter_url})"
            )
        with st.expander("Why these three models?"):
            st.markdown(
                """
PedIR-Bot’s production-facing path is **Query → urgency screen → hybrid
retrieve → grade/rewrite → generate**. The Cloud demo keeps that graph and
swaps only the LLM.

We picked **open-weight** models that are free on OpenRouter today, using
[Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index)
as the ranking source — not paid hosted chat models.

| Role | Model | AA |
| --- | --- | --- |
| Local GPU stand-in (default) | Nemotron 3.5 Lightning (30B-A3B) | 24 |
| Workstation-class | Gemma 4 31B dense | 29 |
| Open-weight ceiling | Nemotron 3 Ultra (550B-A55B) | 38 |

Lightning is the default because it is the size a department GPU box could
host, and it is only two Index points behind Nemotron 3 Super (AA 26) at
about one-quarter the parameters. Gemma is the quality step that still
fits one workstation and has native tool use plus multilingual coverage.
Ultra shows the high end of US open weights; it is not a realistic
on-prem serve for HKCH.

Free endpoints can rate-limit. If one model fails, switch and retry.
Chat is $0 on these slugs; **embeddings still use the OpenAI key**.
                """
            )
            st.caption(AA_INDEX_NOTE)
        st.divider()
    return selected


def render_sidebar(stats, chat_model: str):
    """Render the sidebar with settings and stats."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Toggles
        st.session_state.show_sources = st.toggle(
            "Show source documents",
            value=st.session_state.show_sources
        )
        st.session_state.show_steps = st.toggle(
            "Show processing steps",
            value=st.session_state.show_steps
        )
        
        st.divider()
        
        model = get_demo_model(chat_model)
        model_line = model.short_name if model else chat_model
        # Stats
        st.header("📊 System Status")
        st.markdown(f"""
        <div class="stats-card">
            <strong>Documents:</strong> {stats['total_documents']}<br>
            <strong>Embedding:</strong> {settings.embedding_provider}
            ({settings.openrouter_embedding_model if settings.embedding_provider == "openrouter" else settings.openai_embedding_model})<br>
            <strong>LLM:</strong> {settings.llm_provider}<br>
            <strong>Chat model:</strong> {model_line}
        </div>
        """, unsafe_allow_html=True)
        
        # Memory stats
        mem_stats = st.session_state.memory.get_stats()
        st.markdown(f"""
        <div class="stats-card">
            <strong>Session:</strong> {mem_stats['session_id']}<br>
            <strong>Turns:</strong> {mem_stats['total_turns']}
        </div>
        """, unsafe_allow_html=True)
        
        if is_cloud_demo():
            limits = demo_quota_limits()
            st.markdown(f"""
            <div class="stats-card">
                <strong>This tester:</strong>
                {st.session_state.demo_queries_used}/{limits['session']} questions<br>
                <strong>Shared today:</strong>
                {day_queries_used()}/{limits['day']} questions<br>
                <strong>Max length:</strong> {limits['chars']} characters
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # Actions
        st.header("🔧 Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
        
        st.divider()
        
        # Disclaimer
        st.caption("""
        ⚠️ **Disclaimer**: This chatbot provides educational information only. 
        It is not a substitute for professional medical advice. 
        Always consult your doctor or nurse for medical questions.
        """)


def render_message(message):
    """Render a chat message with optional sources."""
    with st.chat_message(message["role"]):
        # Show decomposition badge if applicable
        if message.get("was_decomposed"):
            st.markdown(f"""
            <span class="decomposed-badge">
                🔀 Combined answer from {message.get('sub_query_count', 2)} sub-queries
            </span>
            """, unsafe_allow_html=True)
        
        st.markdown(message["content"])
        
        # Show safety assessment if available
        if message.get("safety_assessment"):
            risk = message["safety_assessment"].get("risk_level", "none")
            if risk in ["high", "critical"]:
                st.markdown(f"""
                <div class="safety-critical">
                    ⚠️ <strong>Safety Note:</strong> This query was flagged as {risk} risk.
                </div>
                """, unsafe_allow_html=True)
            elif risk == "medium":
                st.markdown(f"""
                <div class="safety-warning">
                    ⚠️ <strong>Note:</strong> Please consult your medical team for personalized guidance.
                </div>
                """, unsafe_allow_html=True)
        
        # Show sources if enabled
        if message.get("sources") and st.session_state.show_sources:
            with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                for i, source in enumerate(message["sources"][:5], 1):
                    score = source.get("score", 0)
                    score_color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                    st.markdown(f"""
                    **{i}. {source.get('filename', 'Unknown')}**  
                    *{source.get('source_org', 'Unknown source')}* | 
                    Score: :{score_color}[{score:.3f}]
                    """)


class ProgressTracker:
    """Tracks and displays processing progress."""
    
    def __init__(self, container, show_steps: bool = True):
        self.container = container
        self.show_steps = show_steps
        self.current_step = 0
        self.step_placeholders = []
        
        if show_steps:
            for i, (icon, text) in enumerate(PROCESSING_STEPS):
                self.step_placeholders.append(
                    self.container.empty()
                )
    
    def update(self, step_index: int, status: str = "active"):
        """Update progress to a specific step."""
        if not self.show_steps:
            return
            
        self.current_step = step_index
        
        for i, placeholder in enumerate(self.step_placeholders):
            icon, text = PROCESSING_STEPS[i]
            if i < step_index:
                # Completed
                placeholder.markdown(f"""
                <div class="progress-step progress-done">
                    ✅ {text.replace('...', '')} ✓
                </div>
                """, unsafe_allow_html=True)
            elif i == step_index:
                # Active
                placeholder.markdown(f"""
                <div class="progress-step progress-active">
                    {icon} {text}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Pending - show empty
                placeholder.empty()
    
    def complete(self):
        """Mark all steps as complete and clear."""
        if self.show_steps:
            time.sleep(0.3)
            for placeholder in self.step_placeholders:
                placeholder.empty()


def process_query(rag_pipeline, prompt, progress_tracker):
    """Process a query with progress tracking."""
    
    # Step 0: Analyze query
    progress_tracker.update(0)
    time.sleep(0.5)
    
    # Single query path with staggered progress updates
    progress_tracker.update(1)  # Searching
    time.sleep(0.5)
    
    progress_tracker.update(2)  # Evaluating
    time.sleep(0.3)
    
    progress_tracker.update(3)  # Safety check
    time.sleep(0.3)
    
    progress_tracker.update(4)  # Generating
    
    result = rag_pipeline.generate_response(
        query=prompt,
        include_sources=True
    )
    return result


def gate_demo_access() -> bool:
    """Ask for the demo passphrase before any API work starts."""
    code = demo_access_code()
    if not code or not is_cloud_demo():
        return True
    if st.session_state.demo_unlocked:
        return True

    render_header()
    st.info(
        "This demo is passphrase-locked so casual visitors cannot spend API credit. "
        "Ask the presenter for the phrase."
    )
    entered = st.text_input("Passphrase", type="password")
    if st.button("Enter demo", type="primary"):
        if entered == code:
            st.session_state.demo_unlocked = True
            st.rerun()
        st.error("That passphrase does not match.")
    return False


def main():
    """Main Streamlit app."""
    init_session_state()

    if not gate_demo_access():
        st.stop()

    missing = missing_cloud_credentials()
    if missing:
        st.error(
            "Streamlit Cloud demo is missing API secrets: "
            + ", ".join(missing)
        )
        st.info(
            "Open the app menu → Settings → Secrets and paste the keys from "
            "`.streamlit/secrets.toml.example`."
        )
        st.stop()

    render_header()
    chat_model = render_model_picker()

    # Initialize RAG system
    try:
        with st.spinner("Initializing RAG system..."):
            rag_pipeline, stats = init_rag_system(chat_model)
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.stop()

    render_sidebar(stats, chat_model)
    
    # Display chat history
    for message in st.session_state.messages:
        render_message(message)
    
    # Chat input
    quota_error = ""
    if is_cloud_demo():
        quota_error = check_demo_query("ok", st.session_state.demo_queries_used)
        if quota_error:
            st.warning(quota_error)

    chat_disabled = bool(quota_error)
    if prompt := st.chat_input(
        "Ask about interventional radiology procedures...",
        disabled=chat_disabled,
    ):
        blocked = check_demo_query(prompt, st.session_state.demo_queries_used)
        if blocked:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant",
                "content": blocked,
            })
            st.rerun()

        if is_cloud_demo():
            st.session_state.demo_queries_used += 1
            record_demo_query()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.memory.add_user_message(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check for follow-up and enhance query
        enhanced_prompt = prompt
        if st.session_state.memory.is_follow_up_question(prompt):
            enhanced_prompt = st.session_state.memory.enhance_query_with_context(prompt)
        
        # Generate response with progress tracking
        with st.chat_message("assistant"):
            # Create progress container
            progress_container = st.container()
            response_container = st.empty()
            
            # Initialize progress tracker
            progress_tracker = ProgressTracker(
                progress_container, 
                show_steps=st.session_state.show_steps
            )
            
            try:
                # Process query with progress
                result = process_query(
                    rag_pipeline, 
                    enhanced_prompt, 
                    progress_tracker
                )
                
                # Clear progress indicators
                progress_tracker.complete()
                
                response = result["response"]
                sources = result.get("sources", [])
                safety = result.get("safety_assessment")
                was_decomposed = result.get("was_decomposed", False)
                sub_query_count = result.get("sub_query_count", 0)
                
                # Show decomposition badge if applicable
                if was_decomposed:
                    st.markdown(f"""
                    <span class="decomposed-badge">
                        🔀 Combined answer from {sub_query_count} sub-queries
                    </span>
                    """, unsafe_allow_html=True)
                
                # Display response
                st.markdown(response)
                
                # Show safety warning if needed
                if safety:
                    risk = safety.get("risk_level", "none")
                    if risk in ["high", "critical"]:
                        st.markdown(f"""
                        <div class="safety-critical">
                            ⚠️ <strong>Safety Note:</strong> This query was flagged as {risk} risk.
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show sources
                if sources and st.session_state.show_sources:
                    with st.expander(f"📚 Sources ({len(sources)} documents)"):
                        for i, source in enumerate(sources[:5], 1):
                            score = source.get("score", 0)
                            score_color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                            st.markdown(f"""
                            **{i}. {source.get('filename', 'Unknown')}**  
                            *{source.get('source_org', 'Unknown source')}* | 
                            Score: :{score_color}[{score:.3f}]
                            """)
                
                # Store in memory and session
                st.session_state.memory.add_assistant_message(
                    response, sources=sources, safety_info=safety
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                    "safety_assessment": safety,
                    "was_decomposed": was_decomposed,
                    "sub_query_count": sub_query_count
                })
                
            except Exception as e:
                progress_tracker.complete()
                st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
