"""LangGraph-based Agentic RAG implementation for Ollama."""
from typing import List, Dict, Any, Optional, Literal, Union
from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional
import logging
import re
import time

from src.safety_guard import SafetyGuard, RiskLevel, SafetyAssessment
from src.data_models import RAGResponse
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.llm import get_langchain_llm
from src.tools import get_knowledge_base_tools
from src.vector_store import VectorStore
from src.guardrails import EmergencyGuardrailMiddleware, SafetyCheckGuardrail, EMERGENCY_RESPONSE
from config import settings
from src import prompts

try:
    from src.sql_tools import get_sql_tools
except ImportError:
    get_sql_tools = None

import json
from langchain_core.tools import render_text_description

def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON object from text."""
    try:
        # Strip thinking tokens first if present
        text = re.sub(r'<unused94>thought.*?<unused95>', '', text, flags=re.DOTALL)
        text = text.replace('<unused94>', '').replace('<unused95>', '')
        text = text.strip()

        # Try to find JSON block explicitly
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Try without code block, find any valid JSON object
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Local models sometimes use single quotes instead of double quotes
                json_str_fixed = json_str.replace("'", '"')
                return json.loads(json_str_fixed)

        return None
    except Exception:
        return None

def extract_text_from_content(content) -> str:
    """
    Extract text from content that can be a string or list of content blocks.

    Args:
        content: Message content (string or list of dicts)

    Returns:
        Extracted text string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle structured content blocks (LangChain 1.0+)
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                # Extract text from content blocks
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif 'text' in block:
                    text_parts.append(str(block['text']))
            elif isinstance(block, str):
                text_parts.append(block)
        return ' '.join(text_parts)
    else:
        # Fallback: convert to string
        return str(content)

# Import LangSmith traceable decorator
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Create a no-op decorator if LangSmith is not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# Prompts removed and moved to src/prompts.py


def create_agentic_rag_graph(
    vector_store: VectorStore,
    llm: Optional[BaseChatModel] = None,
    orchestrator_llm: Optional[BaseChatModel] = None,
    answer_llm: Optional[BaseChatModel] = None,
    grader_llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
) -> StateGraph:
    """
    Create a LangGraph-based agentic RAG graph following the LangGraph tutorial pattern.

    Args:
        vector_store: VectorStore instance for knowledge base access

    Note:
        LLM invoke calls use _invoke_with_retry() to auto-retry on empty responses
        (up to 20 attempts, 3 seconds apart).
        llm: LangChain LLM instance (legacy, for backward compatibility)
        orchestrator_llm: LLM for orchestration (tool calling, query generation, rewriting) - default: qwen2.5:8b
        answer_llm: LLM for final answer generation (medical domain) - default: medgemma
        grader_llm: LLM for document grading (default: same as orchestrator_llm)
        tools: List of tools for the agent (default: knowledge base tools)

    Returns:
        Compiled StateGraph instance
    """
    # Get orchestrator LLM (for tool calling, query generation, rewriting)
    logger.info(f"DEBUG: create_agentic_rag_graph called. Settings provider: {settings.llm_provider}")
    if orchestrator_llm is None:
        # Use configured LLM provider
        orchestrator_llm = get_langchain_llm()
        logger.info(f"Using orchestrator LLM from provider: {settings.llm_provider}")

    # Get answer LLM (for final medical answer generation)
    if answer_llm is None:
        # Use configured LLM provider
        answer_llm = get_langchain_llm()
        logger.info(f"Using answer LLM from provider: {settings.llm_provider}")

    # Use orchestrator_llm for grading if not specified
    if grader_llm is None:
        grader_llm = orchestrator_llm

    # Legacy support: if llm is provided, use it as orchestrator
    if llm is not None and orchestrator_llm is None:
        orchestrator_llm = llm

    # Get tools if not provided
    # Follow LangChain 1.0 agentic RAG pattern: simple retriever tool + SQL tools
    # https://docs.langchain.com/oss/python/langgraph/agentic-rag
    # https://docs.langchain.com/oss/python/langgraph/sql-agent
    if tools is None:
        sql_tools = []
        if get_sql_tools is not None:
            try:
                sql_tools = get_sql_tools()
            except Exception as exc:
                logger.warning(f"SQL document tools unavailable: {exc}")

        kb_tools = get_knowledge_base_tools(vector_store, retriever=None)
        tools = sql_tools + kb_tools
        if sql_tools:
            logger.info(
                f"Initialized {len(sql_tools)} SQL tools + {len(kb_tools)} "
                f"vector search tools = {len(tools)} total"
            )
        else:
            logger.info(
                f"SQL tools skipped (not in this deploy). "
                f"Using {len(kb_tools)} vector search tools"
            )

    # Convert tools to retriever tool format if needed
    retriever_tool = tools[0] if tools else None

    # Initialize guardrails
    emergency_guardrail = EmergencyGuardrailMiddleware()
    safety_check = SafetyCheckGuardrail(llm=grader_llm)  # Use grader_llm for safety checks

    # ── Auto-retry helper for empty LLM responses ──
    MAX_RETRY_ATTEMPTS = 1
    RETRY_DELAY_SECONDS = 3

    def _invoke_with_retry(llm, messages, *, label="LLM", structured=False, structured_schema=None, max_retries=None):
        """
        Invoke an LLM and retry up to max_retries times if the response
        content is empty, waiting RETRY_DELAY_SECONDS between each attempt.

        Args:
            llm: The LangChain LLM instance to invoke.
            messages: The messages to send.
            label: A human-readable label for logging.
            structured: If True, use with_structured_output and return the
                        dict {"raw", "parsed", "parsing_error"}.
            structured_schema: The Pydantic model for structured output.
            max_retries: Override default MAX_RETRY_ATTEMPTS.

        Returns:
            The LLM response (AIMessage) or structured result dict.
        """
        attempts = max_retries if max_retries is not None else MAX_RETRY_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                if structured and structured_schema:
                    structured_llm = llm.with_structured_output(structured_schema, include_raw=True)
                    result = structured_llm.invoke(messages)
                    raw_msg = result.get("raw")
                    parsed = result.get("parsed")
                    raw_content = raw_msg.content if raw_msg and hasattr(raw_msg, 'content') else ""
                    # Consider it non-empty if we got parsed output with an answer, or raw content
                    if parsed and getattr(parsed, 'answer', None):
                        if attempt > 1:
                            logger.info(f"✅ {label}: got non-empty structured response on attempt {attempt}")
                        return result
                    if raw_content:
                        if attempt > 1:
                            logger.info(f"✅ {label}: got raw content on attempt {attempt} (parsed failed)")
                        return result
                    # Empty — retry
                    logger.warning(f"⚠️ {label}: empty structured response on attempt {attempt}/{attempts}, retrying in {RETRY_DELAY_SECONDS}s...")
                else:
                    response = llm.invoke(messages)
                    content = response.content if hasattr(response, 'content') else ""
                    content = extract_text_from_content(content) if content else ""
                    if content.strip():
                        if attempt > 1:
                            logger.info(f"✅ {label}: got non-empty response on attempt {attempt}")
                        return response
                    # Empty — retry
                    logger.warning(f"⚠️ {label}: empty response on attempt {attempt}/{attempts}, retrying in {RETRY_DELAY_SECONDS}s...")
            except Exception as e:
                logger.warning(f"⚠️ {label}: error on attempt {attempt}/{attempts}: {e}, retrying in {RETRY_DELAY_SECONDS}s...")

            if attempt < attempts:
                time.sleep(RETRY_DELAY_SECONDS)

        # Exhausted retries — return whatever we got last
        logger.error(f"❌ {label}: exhausted all {attempts} retry attempts, returning last result")
        if structured and structured_schema:
            structured_llm = llm.with_structured_output(structured_schema, include_raw=True)
            return structured_llm.invoke(messages)
        return llm.invoke(messages)

    # Node 0: Emergency check (before agent processing)
    @traceable(name="emergency_check", run_type="chain", metadata={"node": "guardrail"})
    def check_emergency_node(state: MessagesState):
        """Check for emergency keywords before processing."""
        try:
            logger.info("=== Node: emergency_check (DISABLED by user request) ===")
            # Emergency guardrail bypassed to rely strictly on LLM safety check
            return {"messages": state["messages"]}
        except Exception as e:
            logger.error(f"Error in emergency_check: {e}")
            logger.exception(e)
            # On error, continue processing
            return {"messages": state["messages"]}

    # Conditional routing function for emergency check
    def route_emergency(state: MessagesState) -> Literal["handle_emergency", "generate_query_or_respond"]:
        """Route based on emergency detection."""
        messages = state.get("messages", [])

        # Check if emergency marker is present
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                # Handle both string and structured content
                content = msg.content
                if isinstance(content, str) and content == "__EMERGENCY_DETECTED__":
                    return "handle_emergency"
                elif isinstance(content, list):
                    # Check if any block contains the marker
                    for block in content:
                        if isinstance(block, dict) and block.get('text') == "__EMERGENCY_DETECTED__":
                            return "handle_emergency"

        return "generate_query_or_respond"

    # Node 0.5: Emergency response handler
    @traceable(name="handle_emergency", run_type="chain", metadata={"node": "guardrail"})
    def handle_emergency(state: MessagesState):
        """Return emergency response when emergency keywords detected."""
        messages = state["messages"]

        # Remove the emergency marker message (handle both string and structured content)
        def is_emergency_marker(msg):
            if not (isinstance(msg, AIMessage) and hasattr(msg, 'content')):
                return False
            content = msg.content
            if isinstance(content, str):
                return content == "__EMERGENCY_DETECTED__"
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('text') == "__EMERGENCY_DETECTED__":
                        return True
            return False

        filtered_messages = [m for m in messages if not is_emergency_marker(m)]

        # Extract user query (latest version) to detect language
        query = ""
        for msg in reversed(filtered_messages):
            if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                query = extract_text_from_content(content)
                break

        # Use emergency guardrail to get appropriate response
        emergency_response = emergency_guardrail.check_emergency(query)
        if emergency_response:
            logger.info("Returning emergency response")
            return {"messages": [AIMessage(content=emergency_response)]}
        else:
            # Fallback emergency response
            return {"messages": [AIMessage(content=EMERGENCY_RESPONSE)]}

    # Node 1: Generate query or respond (uses orchestrator_llm with tool calling)
    @traceable(name="generate_query_or_respond", run_type="chain", metadata={"node": "orchestrator"})
    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response. It will decide to retrieve using tools or respond directly."""
        try:
            logger.info("=== Node: generate_query_or_respond ===")
            logger.info(f"State messages count: {len(state.get('messages', []))}")

            # Add system instruction to prefer semantic search
            messages = state["messages"]
            system_instruction = prompts.RETRIEVAL_STRATEGY_INSTRUCTION

            # Add system instruction as first message if not already present
            has_system_instruction = False
            for msg in messages:
                if isinstance(msg, SystemMessage) or (isinstance(msg, dict) and msg.get('role') == 'system'):
                    has_system_instruction = True
                    break

            if not has_system_instruction:
                messages_with_instruction = [SystemMessage(content=system_instruction)] + messages
            else:
                messages_with_instruction = messages

            # Bind tools to orchestrator LLM if supported
            # NOTE: For local models like MedGemma via LM Studio, verify if bind_tools works reliably.
            # If not, fall back to manual tool use.
            # Forcing manual tool use for now as LM Studio/MedGemma seems to hallucinate tool schemas.
            use_bind_tools = False # settings.llm_provider != "lmstudio"

            if use_bind_tools and hasattr(orchestrator_llm, 'bind_tools'):
                try:
                    response = orchestrator_llm.bind_tools(tools).invoke(messages_with_instruction)
                    logger.info(f"Generated response (with tools): {type(response).__name__}")
                    if hasattr(response, 'content'):
                        logger.info(f"Response content preview: {response.content[:200]}...")
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        logger.info(f"Tool calls made: {len(response.tool_calls)}")
                        for tc in response.tool_calls:
                            logger.info(f"  - Tool: {tc.get('name', 'unknown')}, Args: {tc.get('args', {})}")
                    return {"messages": [response]}
                except Exception as e:
                    logger.warning(f"bind_tools failed: {e}, falling back to manual tool use")

                    # Render tools description
                    tools_description = render_text_description(tools)

                    tool_system_prompt = prompts.TOOL_SYSTEM_PROMPT_TEMPLATE.format(tools_description=tools_description)


                    # Invoke with manual prompt
                    # Add system instruction as last message to ensure it's seen
                    context_messages = messages_with_instruction + [SystemMessage(content=tool_system_prompt)]
                    response = orchestrator_llm.invoke(context_messages)
                    content = extract_text_from_content(response.content)

                    # Parse for JSON
                    tool_call_json = _extract_json(content)
                    # Handle both formats: {"tool": "..."} and {"name": "...", "arguments": {...}}
                    tool_name = tool_call_json.get("tool") or tool_call_json.get("name") if tool_call_json else None
                    if tool_name:
                        # Create tool call ID
                        import uuid
                        call_id = f"call_{uuid.uuid4().hex[:8]}"

                        logger.info(f"Manual tool call detected: {tool_name}")

                        # Preprocess query to correct typos before search
                        original_args = tool_call_json.get("arguments", tool_call_json.get("args", {}))
                        if tool_name == "search_kb" and "query" in original_args:
                            original_query = original_args["query"]
                            try:
                                clean_prompt = prompts.QUERY_CLEAN_PROMPT.format(query=original_query)
                                clean_response = orchestrator_llm.invoke([HumanMessage(content=clean_prompt)])
                                cleaned_query = clean_response.content.strip() if hasattr(clean_response, 'content') else original_query
                                # Only use cleaned query if it's reasonable length and not empty
                                if cleaned_query and len(cleaned_query) < 200:
                                    if cleaned_query != original_query:
                                        logger.info(f"🔄 Query cleaned: '{original_query}' → '{cleaned_query}'")
                                    original_args["query"] = cleaned_query
                            except Exception as e:
                                logger.warning(f"Query cleaning failed: {e}, using original query")

                        # Create AIMessage with tool_calls for LangGraph compatibility
                        ai_msg = AIMessage(
                            content="",
                            tool_calls=[{
                                "name": tool_name,
                                "args": original_args,
                                "id": call_id
                            }]
                        )
                        return {"messages": [ai_msg]}

                    # No tool call found, return as normal response
                    return {"messages": [response]}
            else:
                # For models without bind_tools (or when disabled), use the same manual fallback logic
                # Render tools description
                tools_description = render_text_description(tools)

                tool_system_prompt = prompts.TOOL_SYSTEM_PROMPT_TEMPLATE.format(tools_description=tools_description)

                context_messages = messages_with_instruction + [SystemMessage(content=tool_system_prompt)]
                response = _invoke_with_retry(
                    orchestrator_llm, context_messages,
                    label="generate_query_or_respond"
                )
                content = extract_text_from_content(response.content)

                # Parse for JSON
                tool_call_json = _extract_json(content)
                # Handle both formats: {"tool": "..."} and {"name": "...", "arguments": {...}}
                tool_name = tool_call_json.get("tool") or tool_call_json.get("name") if tool_call_json else None
                if tool_name:
                    # Create tool call ID
                    import uuid
                    call_id = f"call_{uuid.uuid4().hex[:8]}"

                    logger.info(f"Manual tool call detected: {tool_name}")

                    # Preprocess query to correct typos before search
                    original_args = tool_call_json.get("arguments", tool_call_json.get("args", {}))
                    if tool_name == "search_kb" and "query" in original_args:
                        original_query = original_args["query"]
                        try:
                            clean_prompt = prompts.QUERY_CLEAN_PROMPT.format(query=original_query)
                            clean_response = orchestrator_llm.invoke([HumanMessage(content=clean_prompt)])
                            cleaned_query = clean_response.content.strip() if hasattr(clean_response, 'content') else original_query
                            # Only use cleaned query if it's reasonable length and not empty
                            if cleaned_query and len(cleaned_query) < 200:
                                if cleaned_query != original_query:
                                    logger.info(f"🔄 Query cleaned: '{original_query}' → '{cleaned_query}'")
                                original_args["query"] = cleaned_query
                        except Exception as e:
                            logger.warning(f"Query cleaning failed: {e}, using original query")

                    # Create AIMessage with tool_calls for LangGraph compatibility
                    ai_msg = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": tool_name,
                            "args": original_args,
                            "id": call_id
                        }]
                    )
                    return {"messages": [ai_msg]}

                # No tool call found — check if response is also empty
                response_content = extract_text_from_content(response.content) if hasattr(response, 'content') else ""
                response_tool_calls = getattr(response, 'tool_calls', []) or []

                if not response_content and not response_tool_calls:
                    # CRITICAL: empty content + no tool calls = graph will skip generate_answer → "I'm sorry"
                    # Force a search_kb call with the latest question
                    logger.warning("⚠️ generate_query_or_respond: LLM returned EMPTY content and NO tool calls. Forcing search_kb fallback.")
                    question = ""
                    for msg in reversed(messages):
                        if isinstance(msg, HumanMessage):
                            question = extract_text_from_content(msg.content)
                            break
                    if question:
                        import uuid
                        call_id = f"call_{uuid.uuid4().hex[:8]}"
                        ai_msg = AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "search_kb",
                                "args": {"query": question[:200]},
                                "id": call_id
                            }]
                        )
                        return {"messages": [ai_msg]}

                # Return as normal response
                return {"messages": [response]}


        except Exception as e:
            logger.error(f"Error in generate_query_or_respond: {e}")
            logger.exception(e)
            # Fallback: just respond without tools
            messages = state["messages"]
            # Add system instruction if not present
            has_system = any(isinstance(msg, SystemMessage) or (isinstance(msg, dict) and msg.get('role') == 'system') for msg in messages)
            if not has_system:
                system_instruction = "PREFER using SQL tools (search_documents_sql) over semantic search (search_kb) for full document context."
                messages = [SystemMessage(content=system_instruction)] + messages
            response = orchestrator_llm.invoke(messages)
            return {"messages": [response]}

    # Node 2: Grade documents
    @traceable(name="grade_documents", run_type="chain", metadata={"node": "grader"})
    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        try:
            logger.info("=== Node: grade_documents ===")
            messages = state["messages"]
            logger.info(f"State messages count: {len(messages)}")

            # Extract question from LATEST human message (could be a rewrite)
            question = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    question = extract_text_from_content(content)
                    logger.info(f"Extracted question for grading: {question[:100]}...")
                    break

            # Get the last tool message (retrieved content)
            context = ""
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    context = msg.content if hasattr(msg, 'content') else str(msg)
                    logger.info(f"Found ToolMessage context (length: {len(context)} chars)")
                    logger.info(f"ToolMessage name: {msg.name if hasattr(msg, 'name') else 'unknown'}")
                    logger.info(f"ToolMessage content preview: {context[:300]}...")
                    break
                elif isinstance(msg, dict) and msg.get('role') == 'tool':
                    context = msg.get('content', '')
                    logger.info(f"Found tool dict context (length: {len(context)} chars)")
                    break
                elif hasattr(msg, 'name') and msg.name:
                    context = msg.content if hasattr(msg, 'content') else str(msg)
                    logger.info(f"Found named message context (length: {len(context)} chars)")
                    break

            # Check rewrite count to prevent infinite loops
            rewrite_count = sum(1 for msg in messages if isinstance(msg, HumanMessage)) - 1
            if rewrite_count >= 2:
                logger.warning(f"Maximum rewrite attempts reached ({rewrite_count}), forcing generate_answer")
                return "generate_answer"

            if not context:
                logger.warning("No context found for grading, defaulting to generate_answer")
                return "generate_answer"

            # Enhance context with metadata for better grading
            # Extract filename and source info from context if available
            enhanced_context = context[:1500]  # Limit context length
            # Try to extract document title/filename from context string
            # Context format from tools.py: "[Document X] Source: {org} - {filename}..."
            filename_match = re.search(r'\[Document \d+\] Source: [^-]+ - ([^\n\(]+)', context)
            if filename_match:
                filename = filename_match.group(1).strip()
                enhanced_context = f"Document Title/Filename: {filename}\n\n" + enhanced_context
                logger.info(f"Extracted filename for grading: {filename}")

            prompt = prompts.GRADE_PROMPT.format(question=question, context=enhanced_context)
            logger.debug(f"Grading prompt length: {len(prompt)} chars")

            # Use raw invoke for simple yes/no to avoid JSON schema conflicts with local models
            response = grader_llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)

            # Strip thinking tokens
            content = re.sub(r'<unused94>thought.*?<unused95>', '', content, flags=re.DOTALL)
            content = content.replace('<unused94>', '').replace('<unused95>', '').strip().lower()

            logger.info(f"Grader raw response: {content[:200]}...")

            # Simple heuristic checking for 'yes' or 'no'
            score = "yes" if "yes" in content else "no"

            logger.info(f"Document grade: {score}")
            logger.info(f"Routing to: {'generate_answer' if score == 'yes' else 'rewrite_question'}")
            return "generate_answer" if score == "yes" else "rewrite_question"
        except Exception as e:
            logger.error(f"Error in grade_documents: {e}")
            logger.exception(e)
            # Default to generate_answer on error
            return "generate_answer"

    # Node 3: Rewrite question
    @traceable(name="rewrite_question", run_type="chain", metadata={"node": "rewriter"})
    def rewrite_question(state: MessagesState):
        """Rewrite the original user question for better retrieval."""
        try:
            logger.info("=== Node: rewrite_question ===")
            messages = state["messages"]
            logger.info(f"State messages count: {len(messages)}")

            # Check if we've already rewritten too many times (prevent infinite loops)
            rewrite_count = sum(1 for msg in messages if isinstance(msg, HumanMessage)) - 1  # Subtract original question
            if rewrite_count >= 2:  # Max 2 rewrites
                logger.warning(f"Maximum rewrite attempts reached ({rewrite_count}), proceeding to generate answer")
                # Extract original question and go straight to answer generation
                original_question = ""
                for msg in messages:
                    if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                        content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                        original_question = extract_text_from_content(content)
                        break
                # Return original question to proceed to answer generation (will be routed to generate_answer)
                return {"messages": [HumanMessage(content=original_question)]}

            # Extract original question
            question = ""
            for msg in messages:
                if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    question = extract_text_from_content(content)
                    logger.info(f"Original question: {question}")
                    break

            prompt = prompts.REWRITE_PROMPT.format(question=question)
            logger.debug(f"Rewrite prompt: {prompt[:200]}...")

            response = orchestrator_llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"Rewrite response type: {type(response).__name__}")

            rewritten = response.content if hasattr(response, 'content') else str(response)

            # Strip thinking tokens
            rewritten = re.sub(r'<unused94>thought.*?<unused95>', '', rewritten, flags=re.DOTALL)
            rewritten = rewritten.replace('<unused94>', '').replace('<unused95>', '').strip()

            logger.info(f"Raw rewritten response: {rewritten[:200]}...")

            # Clean up the response - extract just the question text
            # Handle the new REWRITTEN_QUESTION: prefix
            if "REWRITTEN_QUESTION:" in rewritten:
                rewritten = rewritten.split("REWRITTEN_QUESTION:")[1].strip()

            # Additional cleanup for safety
            # Remove markdown bold and other formatting
            rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
            rewritten = re.sub(r'#+\s*', '', rewritten)
            rewritten = re.sub(r'^\s*[-*]\s*', '', rewritten, flags=re.MULTILINE)

            # Split by common headers if LLM included both thought and result
            for header in ["REWRITTEN QUESTION:", "OPTIMIZED QUESTION:", "Improved Question:", "Question:"]:
                if header.lower() in rewritten.lower():
                    rewritten = re.split(header, rewritten, flags=re.IGNORECASE)[-1].strip()

            # Final sanity check: if it's still multi-line, take the first substantial line
            if "\n" in rewritten:
                lines = [l.strip() for l in rewritten.split('\n') if l.strip()]
                if lines:
                    rewritten = lines[0]

            logger.info(f"Rewritten question (final): {rewritten}")

            # If still too long or contains explanations, try to extract just the question part
            if len(rewritten) > 200 or 'rationale' in rewritten.lower() or 'improved' in rewritten.lower():
                # Look for question marks - take the sentence with the question mark
                sentences = re.split(r'[.!?]', rewritten)
                for sent in sentences:
                    if '?' in sent:
                        rewritten = sent.strip()
                        break

            logger.info(f"Rewritten question: {rewritten}")

            # Log the rewritten question prominently
            logger.info("=" * 80)
            logger.info("🔄 QUESTION REWRITTEN:")
            logger.info(f"Original Question: {question}")
            logger.info(f"Rewritten Question: {rewritten}")
            logger.info("=" * 80)

            return {"messages": [HumanMessage(content=rewritten)]}
        except Exception as e:
            logger.error(f"Error in rewrite_question: {e}")
            # Return original question on error
            return {"messages": [m for m in state["messages"] if isinstance(m, HumanMessage)][:1]}

    # Node 4: Generate answer
    @traceable(name="generate_answer", run_type="chain", metadata={"node": "answer_generator"})
    def generate_answer(state: MessagesState):
        """Generate answer using RAG context and Pydantic structured output."""
        logger.info("=== Node: generate_answer ===")
        messages = state["messages"]

        # Extract question (latest version)
        question = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                question = extract_text_from_content(content)
                break

        # Extract context from tool messages
        context_parts = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                context_parts.append(content)
            elif isinstance(msg, dict) and msg.get('role') == 'tool':
                context_parts.append(msg.get('content', ''))

        context = "\n\n".join(context_parts) if context_parts else ""
        logger.info(f"Total context length: {len(context)} chars")
        logger.info(f"Context preview: {context[:300]}...")
        logger.info(f"Generating answer for question: {question[:50]}...")
        logger.debug(f"Context length: {len(context)} chars")

        if not context:
            logger.warning("No context found, generating answer without context")
            context = "No specific context was retrieved."

        # Extract target language from SystemMessage if present
        language_instruction = prompts.LANGUAGE_INSTRUCTION_DEFAULT
        target_lang = "English"  # Default
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            if isinstance(content, str) and content.startswith("TARGET_LANGUAGE:"):
                target_lang = content.replace("TARGET_LANGUAGE:", "").strip()
                language_instruction = prompts.LANGUAGE_INSTRUCTION_TARGET_TEMPLATE.format(target_lang=target_lang)
                break

        prompt = prompts.GENERATE_PROMPT.format(
            question=question,
            context=context[:65000],
            language_instruction=language_instruction
        )
        logger.debug(f"Generate prompt length: {len(prompt)} chars")

        # Use structured output with include_raw=True for debugging.
        # OpenRouter :free models often queue, then hang on JSON-schema tool
        # calling; skip that path and generate plain text.
        try:
            if settings.llm_provider == "openrouter":
                raise ValueError("skip structured output on OpenRouter")
            result = _invoke_with_retry(
                answer_llm, [HumanMessage(content=prompt)],
                label="generate_answer",
                structured=True,
                structured_schema=RAGResponse,
                max_retries=1
            )

            # include_raw=True returns dict: {"raw": AIMessage, "parsed": RAGResponse|None, "parsing_error": Exception|None}
            raw_msg = result.get("raw")
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")

            # Debug: log raw message details
            raw_content = raw_msg.content if raw_msg and hasattr(raw_msg, 'content') else ""
            raw_tool_calls = raw_msg.tool_calls if raw_msg and hasattr(raw_msg, 'tool_calls') else []
            logger.info(f"⚠️ DEBUG structured: raw content length={len(raw_content)}, tool_calls={len(raw_tool_calls)}, parsed={parsed is not None}, parsing_error={parsing_error}")
            if raw_content:
                logger.debug(f"⚠️ DEBUG structured: raw content preview: {raw_content[:300]}")

            if parsing_error:
                logger.warning(f"⚠️ DEBUG structured: parsing_error={parsing_error}")

            # Guard: parsed can be None when parsing fails
            if parsed is None:
                logger.warning("⚠️ DEBUG: structured output parsed is None, falling back to raw generation")
                # Try to use the raw content instead
                if raw_content:
                    logger.info(f"⚠️ DEBUG: using raw_content as fallback (length={len(raw_content)})")
                    raise ValueError(f"Structured parsing failed but raw content available: {parsing_error}")
                raise ValueError(f"Structured output returned None: {parsing_error}")

            # Check if the answer field is empty
            if not parsed.answer:
                logger.warning(f"⚠️ DEBUG: Structured output parsed but answer is EMPTY. confidence={parsed.confidence}, sources={parsed.sources}")
                if raw_content:
                    raise ValueError("Structured output has empty answer field, raw content available")
                raise ValueError("Structured output has empty answer field")

            # Convert Pydantic model to string for compatibility with existing LangGraph format
            content = parsed.model_dump_json()
            logger.info(f"✅ generate_answer: structured output OK, content length={len(content)}")
            return {"messages": [AIMessage(content=content, response_metadata={'safety_blocked': False})]}

        except Exception as e:
            logger.error(f"Structured output failed, falling back to raw generation: {e}")
            # Fallback: try to use raw content from structured call first, else make a new raw call
            fallback_content = ""
            if 'raw_msg' in dir() and raw_msg and hasattr(raw_msg, 'content') and raw_msg.content:
                fallback_content = raw_msg.content
                logger.info(f"⚠️ DEBUG fallback: reusing raw_msg content, length={len(fallback_content)}")
            else:
                response = _invoke_with_retry(
                    answer_llm, [HumanMessage(content=prompt)],
                    label="generate_answer_fallback"
                )
                fallback_content = response.content if hasattr(response, 'content') else str(response)
                logger.info(f"⚠️ DEBUG fallback: raw LLM call (with retry), response_content length={len(fallback_content)}")

            response_content = fallback_content
            if not response_content:
                logger.warning(f"⚠️ DEBUG fallback: response content is EMPTY even after fallback")

            # Try to force it into our JSON format if the LLM didn't return JSON naturally
            parsed_json = _extract_json(response_content)

            # Strip thinking tags out of the raw response in case we have to fall back to it
            clean_response = re.sub(r'<unused94>thought.*?<unused95>', '', response_content, flags=re.DOTALL)
            clean_response = clean_response.replace('<unused94>', '').replace('<unused95>', '').strip()

            if parsed_json and "answer" in parsed_json:
                final_answer = parsed_json["answer"]
                logger.info(f"⚠️ DEBUG fallback: extracted answer from JSON, length={len(final_answer)}")
            else:
                final_answer = clean_response
                logger.info(f"⚠️ DEBUG fallback: using clean_response as answer, length={len(final_answer)}")

            if parsed_json and "answer" in parsed_json:
                parsed_json["answer"] = final_answer
                content = json.dumps(parsed_json, ensure_ascii=False)
            else:
                # Wrap the raw clean string in the expected JSON schema so the frontend doesn't crash
                fallback_dict = {
                    "answer": final_answer,
                    "confidence": 0.8,
                    "sources": [],
                    "reasoning": "Fallback generation"
                }
                content = json.dumps(fallback_dict, ensure_ascii=False)

            logger.info(f"⚠️ DEBUG fallback: final content length={len(content)}")
            return {"messages": [AIMessage(content=content, response_metadata={'safety_blocked': False})]}

    # Node: Radiologist Review Subagent
    @traceable(name="radiologist_review", run_type="chain", metadata={"node": "reviewer"})
    def radiologist_review(state: MessagesState):
        """Review and potentially rewrite the generated answer for clinical accuracy and completeness."""
        try:
            logger.info("=== Node: radiologist_review ===")
            messages = state["messages"]

            # Extract user query (latest version)
            question = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    question = extract_text_from_content(content)
                    break

            # Extract draft answer from the last AIMessage
            draft_answer_json_str = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    draft_answer_json_str = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    break

            # The draft answer is currently a JSON string containing "answer", "confidence", etc.
            draft_parsed = _extract_json(draft_answer_json_str)
            draft_text = draft_parsed.get("answer", draft_answer_json_str) if draft_parsed else draft_answer_json_str

            # If it was blocked by safety, don't review it, just pass it through
            if "blocked by safety" in draft_text.lower() or "無法提供此回覆" in draft_text:
                logger.info("Skipping review due to safety block.")
                return {"messages": messages[-1:]} # Just return the last message

            # Extract context from context messages
            context_texts = []
            for msg in state.get("messages", []):
                if isinstance(msg, SystemMessage) and "CONTEXT:" in str(msg.content):
                    context_texts.append(str(msg.content))
            context = "\n".join(context_texts)

            # Determine language instruction
            target_lang = "English"
            if "chinese" in question.lower() or "cantonese" in question.lower() or any(ord(c) > 127 for c in question):
                target_lang = "Traditional Chinese (Cantonese style)"
            lang_inst = prompts.RADIOLOGIST_LANGUAGE_INSTRUCTION_TEMPLATE.format(target_lang=target_lang)

            prompt = prompts.RADIOLOGIST_REVIEW_PROMPT.format(
                question=question,
                context=context,
                draft_answer=draft_text,
                language_instruction=lang_inst
            )

            # Use orchestrator/grader LLM for the review
            logger.info("Sending draft to Radiologist Subagent for review...")
            response = grader_llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content if hasattr(response, 'content') else str(response)

            if "__EMERGENCY_DETECTED__" in response_content:
                logger.warning("🚨 Radiologist Subagent detected an emergency!")
                return {"messages": [AIMessage(content="__EMERGENCY_DETECTED__", response_metadata={'reviewed': True})]}

            # Clean output (remove thoughts if any)
            clean_response = re.sub(r'<thought>.*?</thought>', '', response_content, flags=re.DOTALL | re.IGNORECASE)
            clean_response = re.sub(r'<unused94>thought.*?<unused95>', '', clean_response, flags=re.DOTALL)
            clean_response = clean_response.replace('<unused94>', '').replace('<unused95>', '').strip()

            # Programmatically append disclaimer
            disclaimer_en = prompts.DISCLAIMER_EN
            disclaimer_zh = prompts.DISCLAIMER_ZH
            disclaimer = disclaimer_zh if ("chinese" in target_lang.lower() or "cantonese" in target_lang.lower()) else disclaimer_en
            clean_response += disclaimer

            # Post-agent safety check
            logger.info("=== Running post-agent safety check (Radiologist) ===")
            is_safe, error_message = safety_check.check_safety(clean_response, llm=grader_llm)

            if not is_safe:
                logger.warning("⚠️ Safety check failed, returning safety error message")
                safe_msg = prompts.SAFETY_ERROR_EN
                if "chinese" in target_lang.lower() or "cantonese" in target_lang.lower():
                    safe_msg = prompts.SAFETY_ERROR_ZH

                # Wrap safety message in JSON
                safe_json = json.dumps({
                     "answer": safe_msg,
                     "confidence": 1.0,
                     "sources": [],
                     "reasoning": "Blocked by safety guardrail"
                }, ensure_ascii=False)
                return {"messages": [AIMessage(content=safe_json, response_metadata={'safety_blocked': True, 'original_answer': clean_response})]}

            logger.info("✅ Safety check passed (Radiologist)")

            # Package back into JSON format
            if draft_parsed:
                draft_parsed["answer"] = clean_response
                draft_parsed["reasoning"] = draft_parsed.get("reasoning", "") + " | Reviewed by Radiologist Subagent"
                final_content = json.dumps(draft_parsed, ensure_ascii=False)
            else:
                fallback_dict = {
                    "answer": clean_response,
                    "confidence": 0.9,
                    "sources": [],
                    "reasoning": "Reviewed by Radiologist Subagent"
                }
                final_content = json.dumps(fallback_dict, ensure_ascii=False)

            logger.info("✅ Radiologist review complete")
            return {"messages": [AIMessage(content=final_content, response_metadata={'reviewed': True})]}

        except Exception as e:
            logger.error(f"Radiologist review failed: {e}")
            logger.exception(e)
            # On error, just pass the original draft through
            return {"messages": state["messages"][-1:] if state.get("messages") else []}

    def route_radiologist_review(state: MessagesState) -> str:
        """Route to emergency if the radiologist detected one."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                if "__EMERGENCY_DETECTED__" in str(content):
                    return "handle_emergency"
                break
        return END

    # Build the graph
    workflow = StateGraph(MessagesState)

    # Create ToolNode instance once
    tool_node = ToolNode(tools)

    # Node 1.5: Generate multiple queries
    @traceable(name="generate_queries", run_type="chain", metadata={"node": "query_generator"})
    def generate_queries(state: MessagesState):
        """Generate multiple search queries for better recall."""
        try:
            logger.info("=== Node: generate_queries ===")
            messages = state["messages"]

            # Extract original question (use FIRST human message as base for multi-query)
            question = ""
            for msg in messages:
                if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('role') == 'user'):
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    question = extract_text_from_content(content)
                    break

            prompt = prompts.MULTI_QUERY_PROMPT.format(question=question)
            response = orchestrator_llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON array
            try:
                # Clean markdown code blocks if present
                content = content.replace("```json", "").replace("```", "").strip()
                queries = json.loads(content)
                if not isinstance(queries, list):
                    queries = [question]
            except Exception:
                # Fallback: simple line splitting or just use original
                logger.warning(f"Failed to parse multi-query JSON: {content[:100]}...")
                queries = [question]

            logger.info(f"Generated {len(queries)} queries: {queries}")

            # Store queries in a special ToolMessage for the retriever to find
            query_msg = ToolMessage(
                content=json.dumps(queries),
                tool_call_id="multi_query",
                name="multi_query_generator"
            )

            return {"messages": [query_msg]}
        except Exception as e:
            logger.error(f"Error in generate_queries: {e}")
            # Continue with original question implies no extra queries
            return {"messages": []}

    # Node wrapper for tool execution with validation/timing/multi-query support
    def retrieve_with_timing(state: MessagesState):
        """Wrapper around ToolNode that logs tool execution details and timing."""
        import time

        # Get messages that need tool execution
        messages = state.get("messages", [])

        # Check for multi-query input
        queries = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "multi_query_generator":
                try:
                    queries = json.loads(msg.content)
                    logger.info(f"Using {len(queries)} generated queries for retrieval")
                except:
                    pass
                break

        # If we have multiple queries, we need to run retrieval for EACH
        if queries and len(queries) > 1:
            all_tool_calls = []

            # We assume we are using the 'search_kb' tool (vector search) for these
            # Find the search_kb tool definition to get correct name
            search_tool_name = "search_kb"

            for q in queries:
                 all_tool_calls.append({
                    "name": search_tool_name,
                    "args": {"query": q},
                    "id": f"call_{hash(q)}"
                 })

            # Create a temporary state with AIMessage requesting these tool calls
            # This is a bit of a hack to use ToolNode with multiple calls
            temp_ai_msg = AIMessage(content="", tool_calls=all_tool_calls)

            # Invoke ToolNode
            start_time = time.time()
            tool_result = tool_node.invoke({"messages": [temp_ai_msg]})
            execution_time = time.time() - start_time
            logger.info(f"⏱️  Multi-query Retrieval Time: {execution_time:.2f} seconds")

            return tool_result

        # Find the last AIMessage with tool_calls (Original Logic)
        tool_calls_info = []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    tool_calls_info.append((tool_name, tool_args))
                break  # Only process the most recent AIMessage with tool_calls

        # Log tool execution details before execution
        if tool_calls_info:
            logger.info("=" * 80)
            for tool_name, tool_args in tool_calls_info:
                logger.info(f"🔧 TOOL CALLED: {tool_name}")
                logger.info(f"📝 Query/Arguments:")
                for key, value in tool_args.items():
                    logger.info(f"   {key}: {value}")
                logger.info("-" * 80)

            # Time the tool execution
            start_time = time.time()

            # Execute the tool using ToolNode
            tool_result_state = tool_node.invoke(state)

            execution_time = time.time() - start_time

            logger.info(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            logger.info("=" * 80)

            return tool_result_state

        # If no tool calls found, use standard ToolNode behavior
        return tool_node.invoke(state)

    # Add nodes
    workflow.add_node("check_emergency", check_emergency_node)
    workflow.add_node("handle_emergency", handle_emergency)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", retrieve_with_timing)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("radiologist_review", radiologist_review)

    # Add edges - start with emergency check
    workflow.add_edge(START, "check_emergency")

    # Conditional edge: route to emergency handler or continue
    workflow.add_conditional_edges(
        "check_emergency",
        route_emergency,
        {
            "handle_emergency": "handle_emergency",
            "generate_query_or_respond": "generate_query_or_respond",
        },
    )

    # Emergency handler goes to END
    workflow.add_edge("handle_emergency", END)

    # Conditional edge: decide whether to run multi-query or direct retrieval
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # For now, we'll insert multi-query in the semantic search path
    # But complicating the graph too much might break existing flows.
    # Let's simple insert "generate_queries" before any retrieval if it's semantic.

    # Actually, the original design has "generate_query_or_respond" -> "retrieve".
    # We should intercept this.
    # But since tools_condition is a prebuilt router, validation is complex.
    # For this implementation, we will keep it simple:
    # IF tools_condition says "tools", we go to "retrieve" (which now supports multi-query inside)
    # The generation of multi-queries should ideally happen inside "generate_query_or_respond" or between.

    # Let's add the node but for safety in this iteration, we make it optional/parallel.
    # Or better: "retrieve" node now checks if it should expand the query.


    # Conditional edge: grade documents and route accordingly
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    # Route generate_answer directly to END (radiologist_review temporarily disabled)
    workflow.add_edge("generate_answer", END)
    # # Route generate_answer to radiologist_review (DISABLED)
    # workflow.add_edge("generate_answer", "radiologist_review")
    # # Route radiologist_review to END or emergency
    # workflow.add_conditional_edges(
    #     "radiologist_review",
    #     route_radiologist_review,
    #     {
    #         "handle_emergency": "handle_emergency",
    #         END: END
    #     }
    # )

    # Add edge from rewrite_question back to generate_query_or_respond, but limit loops
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile without checkpointer (not needed for simple RAG)
    graph = workflow.compile()
    logger.info("Agentic RAG graph compiled successfully")

    return graph

