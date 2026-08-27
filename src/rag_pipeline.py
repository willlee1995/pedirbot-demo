"""RAG pipeline implementation using LangGraph Agentic RAG."""
import json
import re
import time
from typing import List, Dict, Any, Optional

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from loguru import logger

from src.agentic_rag import create_agentic_rag_graph
from src.vector_store import VectorStore
from src.retriever import AdvancedRetriever
from src.safety_guard import SafetyGuard, SafetyAssessment, RiskLevel
from src.llm import get_llm_provider
from src.source_allowlist import filter_citation_sources
from config import settings


def format_provider_error(exc: BaseException) -> str:
    """Turn an OpenRouter / OpenAI API error into text the tester can read."""
    body = None
    status = None
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        body = getattr(current, "body", None) or body
        status = getattr(current, "status_code", None) or status
        current = current.__cause__ or getattr(current, "__context__", None)

    if isinstance(body, dict):
        err = body.get("error", body)
        if not isinstance(err, dict):
            err = {"message": str(err)}
        meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
        code = err.get("code") or status or "error"
        raw = meta.get("raw") or err.get("message") or str(exc)
        lines = [f"Error code: {code}", "", str(raw)]
        provider = meta.get("provider_name")
        if provider:
            lines.extend(["", f"Provider: {provider}"])
        hint = meta.get("remedy_hint")
        if hint:
            lines.extend(["", str(hint)])
        return "\n".join(lines)

    text = str(exc).strip() or repr(exc)
    if status:
        return f"Error code: {status}\n\n{text}"
    return text

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


class RAGPipeline:
    """RAG pipeline using LangGraph Agentic RAG for generating responses."""

    # Emergency keywords that trigger canned responses
    # Note: For PedIR context, we need specific qualifiers to avoid false positives
    # when discussing procedures like embolization that involve bleeding management
    EMERGENCY_KEYWORDS = [
        # Breathing emergencies (always urgent)
        "can't breathe", "cannot breathe", "not breathing", "stopped breathing",
        # Cardiac (always urgent)
        'chest pain', 'heart attack',
        # Severe allergic reaction
        'anaphylaxis', 'severe allergic reaction', 'throat swelling',
        # True emergencies
        'emergency room', 'call 999', 'call ambulance', '999', 'unconscious', 'passed out',
        # Chinese emergency terms
        '不能呼吸', '胸痛', '過敏反應', '緊急', '昏迷', '急症室',
        # Post-procedure true emergencies (require "heavy/uncontrolled" qualifier)
        'uncontrolled bleeding', 'heavy bleeding', "bleeding won't stop", "bleeding will not stop",
        '大量出血', '無法止血'
    ]

    EMERGENCY_RESPONSE = """This sounds like it could be an emergency. Please do not rely on this chatbot.

**Call 999 or go to the nearest Accident & Emergency department immediately.**

If you have urgent questions about your procedure, please contact the HKCH IR nurse coordinator at [phone number].
"""

    EMERGENCY_RESPONSE_ZH = """這聽起來可能是緊急情況。請不要依賴此聊天機器人。

**請立即致電999或前往最近的急症室。**

如果您對手術有緊急疑問，請致電[電話號碼]聯絡香港兒童醫院介入放射科護士協調員。
"""

    def __init__(
        self,
        vector_store: VectorStore,
        retriever: Optional[AdvancedRetriever] = None,
        graph: Optional[StateGraph] = None,
        use_safety_guard: bool = True,
    ):
        """
        Initialize the RAG pipeline using LangGraph.

        Args:
            vector_store: VectorStore instance
            retriever: AdvancedRetriever instance (optional, kept for compatibility)
            graph: Compiled LangGraph StateGraph (optional, will be created if not provided)
            use_safety_guard: Whether to use the safety guardrail agent
        """
        self.vector_store = vector_store
        self.retriever = retriever
        self.use_safety_guard = use_safety_guard

        # Initialize SafetyGuard if enabled
        if use_safety_guard:
            try:
                llm_provider = get_llm_provider()
                self.safety_guard = SafetyGuard(llm_provider, use_llm_check=False)  # Pattern-based for speed
                logger.info("Initialized RAG pipeline with SafetyGuard")
            except Exception as e:
                logger.warning(f"Failed to initialize SafetyGuard: {e}")
                self.safety_guard = None
                self.use_safety_guard = False
        else:
            self.safety_guard = None

        # Create LangGraph if not provided
        if graph is None:
            self.graph = create_agentic_rag_graph(vector_store)
        else:
            self.graph = graph

        logger.info("Initialized RAG pipeline with LangGraph Agentic RAG")

    def _check_emergency(self, query: str) -> Optional[str]:
        """
        Check if query contains emergency keywords.

        Args:
            query: User query

        Returns:
            Emergency response if triggered, None otherwise
        """
        query_lower = query.lower()

        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in query_lower:
                logger.warning(f"Emergency keyword detected: {keyword}")
                # Detect language and return appropriate response
                if any(ord(char) > 127 for char in query):  # Contains non-ASCII (likely Chinese)
                    return self.EMERGENCY_RESPONSE_ZH
                else:
                    return self.EMERGENCY_RESPONSE

        return None

    @traceable(name="rag_generate_response", metadata={"component": "rag_pipeline"})
    def generate_response(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        include_sources: bool = True,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using LangGraph Agentic RAG.

        Args:
            query: User query
            k: Number of documents to retrieve (not used, kept for compatibility)
            filter_dict: Optional metadata filter (not directly used, kept for compatibility)
            temperature: LLM temperature (not directly used, kept for compatibility)
            include_sources: Whether to include source documents in response
            language: Optional target language (e.g., 'Cantonese', 'English')

        Returns:
            Dict with 'response', 'sources', 'is_emergency', and 'total_time' keys
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info(f"🔄 STARTING QUERY PROCESSING")
        logger.info(f"📝 Query: {query}")
        logger.info(f"⏰ Start Time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        logger.info("=" * 80)

        # Safety Guard: Pre-query assessment
        safety_assessment = None
        if self.use_safety_guard and self.safety_guard:
            safety_assessment = self.safety_guard.assess_query(query)
            logger.info(f"Safety assessment: {safety_assessment.risk_level.value} (emergency: {safety_assessment.is_emergency})")

            # Handle critical emergencies from SafetyGuard
            if safety_assessment.is_emergency or safety_assessment.risk_level == RiskLevel.CRITICAL:
                end_time = time.time()
                total_time = end_time - start_time
                logger.warning(f"SafetyGuard detected emergency: {safety_assessment.clinical_concerns}")
                return {
                    'response': self.safety_guard.get_emergency_response(query),
                    'sources': [],
                    'is_emergency': True,
                    'total_time': total_time,
                    'safety_assessment': {
                        'risk_level': safety_assessment.risk_level.value,
                        'concerns': safety_assessment.clinical_concerns
                    }
                }

        # Check for emergency keywords (fallback if SafetyGuard disabled)
        emergency_response = self._check_emergency(query)
        if emergency_response:
            end_time = time.time()
            total_time = end_time - start_time
            logger.info("=" * 80)
            logger.info(f"✅ QUERY COMPLETED (Emergency Response)")
            logger.info(f"⏱️  Total Time: {total_time:.2f} seconds")
            logger.info("=" * 80)
            return {
                'response': emergency_response,
                'sources': [],
                'is_emergency': True,
                'total_time': total_time,
            }

        greeting = re.compile(
            r"^(hi|hello|hey|yo|thanks|thank you|ok|okay|"
            r"good morning|good afternoon|good evening|"
            r"你好|早晨|午安|晚安|多謝|谢谢)[\s!.?？！]*$",
            re.IGNORECASE,
        )
        if greeting.match((query or "").strip()):
            end_time = time.time()
            return {
                "response": (
                    "Hello — I am PedIR-Bot, an educational assistant for "
                    "paediatric interventional radiology. Ask about a procedure, "
                    "fasting, a PICC line, or aftercare. I am not a substitute "
                    "for your clinical team."
                ),
                "sources": [],
                "is_emergency": False,
                "total_time": end_time - start_time,
            }

        # Use LangGraph to generate response
        try:
            logger.info("Invoking LangGraph agentic RAG...")

            messages = []
            if language:
                messages.append(SystemMessage(content=f"TARGET_LANGUAGE: {language}"))
            messages.append(HumanMessage(content=query))

            # Run the graph
            result = self.graph.invoke({
                "messages": messages
            })

            # Extract final response from messages
            messages = result.get("messages", [])
            response_text = ""
            sources = []

            # Log all messages in full detail for human observers
            logger.info("=" * 80)
            logger.info(f"📨 CONVERSATION MESSAGES ({len(messages)} total)")
            logger.info("=" * 80)

            for i, msg in enumerate(messages):
                logger.info(f"\n--- Message {i+1} ---")
                msg_type = type(msg).__name__

                if isinstance(msg, AIMessage):
                    logger.info(f"Type: {msg_type} (AI Response)")
                    # Show full content
                    if hasattr(msg, 'content') and msg.content:
                        logger.info(f"Content:\n{msg.content}")
                    else:
                        logger.info(f"Content: (empty)")

                    # Show tool calls with full details
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        logger.info(f"\nTool Calls ({len(msg.tool_calls)}):")
                        for j, tc in enumerate(msg.tool_calls, 1):
                            tool_name = tc.get('name', 'unknown')
                            tool_args = tc.get('args', {})
                            logger.info(f"  {j}. Tool: {tool_name}")
                            logger.info(f"     Arguments:")
                            for key, value in tool_args.items():
                                logger.info(f"       - {key}: {value}")

                elif isinstance(msg, ToolMessage):
                    tool_name = msg.name if hasattr(msg, 'name') else 'Unknown'
                    logger.info(f"Type: {msg_type} (Tool Result)")
                    logger.info(f"Tool: {tool_name}")
                    # Show full content
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content
                        # Show first 1000 chars if very long, otherwise full
                        if len(content) > 1000:
                            logger.info(f"Content (first 1000 chars):\n{content[:1000]}...")
                            logger.info(f"... ({len(content) - 1000} more characters)")
                        else:
                            logger.info(f"Content:\n{content}")
                    else:
                        logger.info(f"Content: (empty)")

                elif isinstance(msg, HumanMessage):
                    logger.info(f"Type: {msg_type} (User Query)")
                    # Show full content
                    if hasattr(msg, 'content') and msg.content:
                        logger.info(f"Content:\n{msg.content}")
                    else:
                        logger.info(f"Content: (empty)")

                elif isinstance(msg, dict):
                    msg_role = msg.get('role', 'unknown')
                    msg_type_name = msg.get('type', 'unknown')
                    logger.info(f"Type: dict (role={msg_role}, type={msg_type_name})")
                    content = str(msg.get('content', ''))
                    if len(content) > 1000:
                        logger.info(f"Content (first 1000 chars):\n{content[:1000]}...")
                    else:
                        logger.info(f"Content:\n{content}")
                else:
                    logger.info(f"Type: {msg_type}")
                    msg_str = str(msg)
                    if len(msg_str) > 500:
                        logger.info(f"Content (first 500 chars):\n{msg_str[:500]}...")
                    else:
                        logger.info(f"Content:\n{msg_str}")

            logger.info("=" * 80)

            # Find the final assistant message (AIMessage from LangChain)
            logger.debug(f"Total messages in result: {len(messages)}")
            safety_blocked = False
            original_answer = ""
            for msg in reversed(messages):
                # Check for AIMessage instance (LangChain message object)
                if isinstance(msg, AIMessage):
                    response_text = msg.content if hasattr(msg, 'content') else str(msg)
                    if hasattr(msg, 'response_metadata'):
                        safety_blocked = msg.response_metadata.get('safety_blocked', False)
                        original_answer = msg.response_metadata.get('original_answer', '')
                    logger.debug(f"Found AIMessage with content length: {len(response_text) if response_text else 0}")
                    if not response_text:
                        logger.warning(f"⚠️ DEBUG: AIMessage found but content is EMPTY. tool_calls={getattr(msg, 'tool_calls', None)}")
                    break
                # Fallback: check for dict format
                elif isinstance(msg, dict):
                    if msg.get('role') == 'assistant' or 'type' in msg and msg.get('type') == 'ai':
                        response_text = msg.get('content', '')
                        logger.debug(f"Found assistant message in dict format")
                        break
                # Fallback: check for content attribute
                elif hasattr(msg, 'content'):
                    # Make sure it's not a HumanMessage or ToolMessage
                    if not isinstance(msg, (HumanMessage, ToolMessage)):
                        response_text = msg.content if hasattr(msg, 'content') else str(msg)
                        logger.debug(f"Found message with content attribute")
                        break
            else:
                logger.warning(f"⚠️ DEBUG: No AIMessage found in {len(messages)} messages. Types: {[type(m).__name__ for m in messages]}")

            # Strip thinking tokens if present
            if response_text:
                original_len = len(response_text)
                # Pattern to match <unused94>thought...<unused95> or similar thinking blocks
                # We use DOTALL so the `.` matches newlines as well
                response_text = re.sub(r'<unused94>thought.*?<unused95>', '', response_text, flags=re.DOTALL)
                # Also strip just <unused94> and <unused95> in case they appear isolated
                response_text = response_text.replace('<unused94>', '').replace('<unused95>', '')
                response_text = response_text.strip()
                if not response_text and original_len > 0:
                    logger.warning(f"⚠️ DEBUG: response_text became EMPTY after stripping think tokens (was {original_len} chars)")


            # Extract sources from tool messages
            if include_sources:
                for msg in messages:
                    if isinstance(msg, ToolMessage) or (hasattr(msg, 'name') and msg.name):
                        tool_name = msg.name if hasattr(msg, 'name') else 'Unknown'
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        if not isinstance(content, str):
                            content = str(content)

                        # Parse document info from formatted tool output
                        # Format: [Document N] Source: ORG | Region: X | Category: Y | filename (Relevance: 0.XXX)
                        doc_pattern = r'\[Document \d+\] Source: ([^|]+) \| Region: ([^|]+) \| Category: ([^|]+) \| ([^(]+) \(Relevance: ([\d.]+)\)'
                        matches = re.findall(doc_pattern, content)

                        if matches:
                            for match in matches:
                                source_org, region, category, filename, score = match
                                sources.append({
                                    'filename': filename.strip(),
                                    'source_org': source_org.strip(),
                                    'region': region.strip(),
                                    'category': category.strip(),
                                    'score': float(score),
                                    'tool': tool_name,
                                    'content': content[:200] + '...' if len(content) > 200 else content
                                })
                        else:
                            # Format 2: SQL Tool
                            # [N] Document ID: ...
                            #     Filename: ...
                            sql_pattern = r'Document ID: (.*?)\n\s+Filename: (.*?)\n\s+Source: (.*?)\n\s+Region: (.*?)\n\s+Category: (.*?)\n'
                            matches_sql = re.findall(sql_pattern, content, re.DOTALL)

                            if matches_sql:
                                for match in matches_sql:
                                    doc_id, filename, source_org, region, category = match
                                    sources.append({
                                        'filename': filename.strip(),
                                        'source_org': source_org.strip(),
                                        'region': region.strip(),
                                        'category': category.strip(),
                                        'score': 1.0,  # SQL search implies exact match relevance
                                        'tool': tool_name,
                                        'content': content[:200] + '...' if len(content) > 200 else content
                                    })
                            else:
                                # Fallback: just include tool and content
                                sources.append({
                                    'tool': tool_name,
                                    'filename': 'Unknown',
                                    'source_org': 'Unknown',
                                    'score': 0.0,
                                    'content': content[:200] + '...' if len(content) > 200 else content
                                })

                sources = filter_citation_sources(sources)

            # Parse structured output if available
            try:
                # Try to parse response_text as JSON
                structured_data = json.loads(response_text)

                # If successful, use the 'answer' field as the main response
                if isinstance(structured_data, dict) and 'answer' in structured_data:
                    parsed_answer = structured_data['answer']
                    if not parsed_answer:
                        logger.warning(f"⚠️ DEBUG: JSON parsed OK but 'answer' field is EMPTY. Full JSON keys: {list(structured_data.keys())}")
                        logger.warning(f"⚠️ DEBUG: confidence={structured_data.get('confidence')}, sources={structured_data.get('sources')}, reasoning={str(structured_data.get('reasoning', ''))[:200]}")
                    response_text = parsed_answer
                    logger.info("Successfully parsed structured output from LLM")

                    # If sources are provided in the structured output, we can use them
                    # But often the LLM just cites filenames. We should verify against our retrieved sources.
                    if 'sources' in structured_data and structured_data['sources']:
                        llm_sources = structured_data['sources']
                        logger.info(f"LLM cited sources: {llm_sources}")
                        # We could optionally filter 'sources' list to only include those cited by LLM
                        # For now, we prefer the tool outputs as they contain metadata
                else:
                    logger.warning(f"⚠️ DEBUG: JSON parsed but no 'answer' key. Keys: {list(structured_data.keys()) if isinstance(structured_data, dict) else type(structured_data).__name__}")
            except json.JSONDecodeError:
                # Not JSON, treat as raw text
                logger.debug(f"Response is not JSON, treating as raw text (length: {len(response_text)})")
                pass
            except Exception as e:
                logger.warning(f"Error parsing structured output: {e}")

            if not response_text:
                logger.warning(f"⚠️ DEBUG: FINAL response_text is EMPTY after all extraction steps")
                logger.warning(f"⚠️ DEBUG: Total messages: {len(messages)}, types: {[type(msg).__name__ for msg in messages]}")
                # Log the last few message contents for debugging
                for idx, msg in enumerate(messages[-3:]):
                    msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                    logger.warning(f"⚠️ DEBUG: Message[-{len(messages)-idx}] ({type(msg).__name__}): {msg_content[:300]}...")
                response_text = "I'm sorry, I couldn't generate a response."

            # Calculate total time
            end_time = time.time()
            total_time = end_time - start_time

            logger.info("=" * 80)
            logger.info(f"✅ QUERY COMPLETED")
            logger.info(f"📝 Original Query: {query}")
            logger.info(f"📤 Response Length: {len(response_text)} characters")
            logger.info(f"📚 Sources Used: {len(sources)}")
            logger.info(f"⏱️  Total Round Trip Time: {total_time:.2f} seconds")
            logger.info(f"⏰ End Time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
            logger.info("=" * 80)

            return {
                'response': response_text,
                'sources': sources,
                'source_documents': sources,  # Alias for compatibility
                'is_emergency': False,
                'safety_blocked': safety_blocked,
                'original_answer': original_answer,
                'total_time': total_time,
                'messages': messages,  # Raw graph messages for eval/debugging
            }

        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time if 'start_time' in locals() else 0
            logger.error(f"Error generating response: {e}")
            logger.exception(e)
            logger.info("=" * 80)
            logger.info(f"❌ QUERY FAILED")
            logger.info(f"⏱️  Time Before Error: {total_time:.2f} seconds")
            logger.info("=" * 80)
            error_text = format_provider_error(e)
            return {
                'response': error_text,
                'error': error_text,
                'sources': [],
                'is_emergency': False,
                'total_time': total_time,
            }

    def stream_response(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
    ):
        """
        Generate a streaming response using LangGraph Agentic RAG.

        Args:
            query: User query
            k: Number of documents to retrieve (not used)
            filter_dict: Optional metadata filter
            temperature: LLM temperature (not directly used)

        Yields:
            Response chunks and metadata
        """
        logger.info(f"Processing streaming query: {query[:100]}...")

        # Check for emergency keywords
        emergency_response = self._check_emergency(query)
        if emergency_response:
            yield {'type': 'emergency', 'content': emergency_response}
            return

        # Use LangGraph streaming
        try:
            logger.info("Streaming LangGraph response...")
            for chunk in self.graph.stream({
                "messages": [HumanMessage(content=query)]
            }):
                # LangGraph returns chunks per node
                for node_name, node_output in chunk.items():
                    if node_name == "generate_answer":
                        # Final answer node
                        messages = node_output.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, 'content'):
                                yield {'type': 'response', 'content': msg.content}
                    elif node_name == "retrieve":
                        # Retrieval node
                        yield {'type': 'tool_execution', 'content': f"Retrieving documents from {node_name}..."}
                    elif node_name == "generate_query_or_respond":
                        # Query generation node
                        yield {'type': 'agent_thinking', 'content': f"Thinking about query: {query[:50]}..."}
                    elif node_name == "rewrite_question":
                        # Question rewriting node
                        yield {'type': 'agent_thinking', 'content': "Rewriting question for better retrieval..."}

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield {'type': 'error', 'content': f"Error: {str(e)}"}

        logger.info("Streaming completed")