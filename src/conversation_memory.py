"""Conversation memory management for multi-turn chat."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from loguru import logger


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Manages conversation history for multi-turn RAG interactions.

    Features:
    - Maintains rolling window of conversation turns
    - Summarizes older context when window is full
    - Provides context for follow-up questions
    """

    def __init__(self, max_turns: int = 10, max_tokens_estimate: int = 2000):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to keep in full detail
            max_tokens_estimate: Approximate token limit for context
        """
        self.max_turns = max_turns
        self.max_tokens_estimate = max_tokens_estimate
        self.history: deque[ConversationTurn] = deque(maxlen=max_turns * 2)
        self.summary: str = ""
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Initialized ConversationMemory (max_turns={max_turns}, session={self.session_id})")

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a conversation turn to memory.

        Args:
            role: "user" or "assistant"
            content: The message content
            metadata: Optional metadata (sources, safety assessment, etc.)
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.history.append(turn)
        logger.debug(f"Added {role} turn to memory (total: {len(self.history)} turns)")

    def add_user_message(self, content: str) -> None:
        """Add a user message to memory."""
        self.add_turn("user", content)

    def add_assistant_message(self, content: str, sources: List[Dict] = None,
                               safety_info: Dict = None) -> None:
        """Add an assistant message to memory with optional metadata."""
        metadata = {}
        if sources:
            # Store only source filenames, not full content
            metadata["source_files"] = [s.get("filename", "unknown") for s in sources[:5]]
        if safety_info:
            metadata["safety_assessment"] = safety_info

        self.add_turn("assistant", content, metadata)

    def get_context_for_query(self, current_query: str, num_turns: int = 4) -> str:
        """
        Get conversation context formatted for inclusion in RAG query.

        Args:
            current_query: The current user query
            num_turns: Number of recent turns to include

        Returns:
            Formatted context string
        """
        if len(self.history) == 0:
            return ""

        # Get recent turns
        recent_turns = list(self.history)[-num_turns:]

        if not recent_turns:
            return ""

        context_parts = ["[Previous Conversation]"]
        for turn in recent_turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            # Truncate long messages
            content = turn.content[:500] + "..." if len(turn.content) > 500 else turn.content
            context_parts.append(f"{role_label}: {content}")

        return "\n".join(context_parts)

    def get_messages_for_llm(self, system_prompt: str, current_query: str,
                             include_history: bool = True,
                             max_history_turns: int = 4) -> List[Dict[str, str]]:
        """
        Build message list for LLM including conversation history.

        Args:
            system_prompt: The system prompt
            current_query: Current user query
            include_history: Whether to include conversation history
            max_history_turns: Maximum history turns to include

        Returns:
            List of message dicts for LLM
        """
        messages = [{"role": "system", "content": system_prompt}]

        if include_history and len(self.history) > 0:
            # Add recent conversation history
            recent_turns = list(self.history)[-max_history_turns:]
            for turn in recent_turns:
                messages.append({
                    "role": turn.role,
                    "content": turn.content
                })

        # Add current query
        messages.append({"role": "user", "content": current_query})

        return messages

    def get_follow_up_context(self) -> Optional[str]:
        """
        Get context hint if this appears to be a follow-up question.

        Returns:
            Context hint or None
        """
        if len(self.history) < 2:
            return None

        # Get last assistant response
        for turn in reversed(list(self.history)):
            if turn.role == "assistant":
                # Check if sources were used
                sources = turn.metadata.get("source_files", [])
                if sources:
                    return f"(Follow-up to previous discussion about: {', '.join(sources[:3])})"
                break

        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self.history.clear()
        self.summary = ""
        logger.info("Conversation memory cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "session_id": self.session_id,
            "total_turns": len(self.history),
            "user_turns": sum(1 for t in self.history if t.role == "user"),
            "assistant_turns": sum(1 for t in self.history if t.role == "assistant"),
            "has_summary": bool(self.summary)
        }

    def is_follow_up_question(self, query: str) -> bool:
        """
        Detect if a query is likely a follow-up question.

        Args:
            query: The user query

        Returns:
            True if likely a follow-up
        """
        if len(self.history) == 0:
            return False

        # Common follow-up indicators
        follow_up_patterns = [
            "what about", "how about", "and ", "also ", "more about",
            "tell me more", "explain more", "what else", "anything else",
            "why", "how long", "when", "is it", "can i", "can you", "could you",
            "should i", "would you", "further", "explain",
            "那", "還有", "另外", "為什麼", "怎麼", "可以", "需要"
        ]

        query_lower = query.lower()
        for pattern in follow_up_patterns:
            if query_lower.startswith(pattern) or f" {pattern}" in query_lower:
                return True

        # Short queries after conversation likely follow-ups
        if len(query.split()) <= 5 and len(self.history) >= 2:
            return True

        return False

    def enhance_query_with_context(self, query: str) -> str:
        """
        Enhance a follow-up query with conversation context.

        Args:
            query: Original user query

        Returns:
            Enhanced query with context if needed
        """
        if not self.is_follow_up_question(query):
            return query

        # Get last topic discussed
        context = self.get_follow_up_context()
        if context:
            logger.info(f"Enhancing follow-up query with context: {context}")
            return f"{query} {context}"

        return query
