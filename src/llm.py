"""LLM integration for OpenAI-compatible and Ollama APIs with LangChain support."""
from typing import List, Dict, Any, Optional
import os

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import settings


def get_langchain_llm(provider: str = None, **kwargs) -> BaseChatModel:
    """
    Get a LangChain LLM instance.

    Args:
        provider: 'openai' or 'ollama' (default from settings)
        **kwargs: Additional parameters for LLM initialization

    Returns:
        LangChain BaseChatModel instance
    """
    provider = provider or settings.llm_provider

    if provider == "openai":
        return ChatOpenAI(
            model=kwargs.get('model', settings.openai_chat_model),
            api_key=kwargs.get('api_key', settings.openai_api_key),
            base_url=kwargs.get('base_url', settings.openai_api_base),
            temperature=kwargs.get('temperature', settings.agent_temperature),
            max_tokens=kwargs.get('max_tokens', 1024),
            streaming=kwargs.get('streaming', False),
        )
    elif provider == "ollama":
        if ChatOllama is None:
            raise ImportError(
                "langchain-ollama is required for the Ollama provider. "
                "Install the full local stack with: pip install -r requirements-full.txt"
            )
        base_url = kwargs.get('base_url', settings.ollama_api_base)
        # Set Ollama host if needed
        if base_url and base_url != "http://localhost:11434":
            os.environ['OLLAMA_HOST'] = base_url.replace('http://', '').replace('https://', '')

        return ChatOllama(
            model=kwargs.get('model', settings.ollama_chat_model),
            base_url=base_url,
            temperature=kwargs.get('temperature', settings.agent_temperature),
            num_ctx=kwargs.get('num_ctx', 4096),
        )
    elif provider == "lmstudio":
        # LM Studio uses OpenAI-compatible API
        return ChatOpenAI(
            model=kwargs.get('model') or settings.lmstudio_chat_model,
            api_key="lm-studio",  # LM Studio doesn't require real key
            base_url=kwargs.get('base_url') or settings.lmstudio_api_base,
            temperature=kwargs.get('temperature', settings.agent_temperature),
            max_tokens=kwargs.get('max_tokens', 1024),
            streaming=kwargs.get('streaming', False),
        )
    elif provider == "openrouter":
        # OpenRouter uses OpenAI-compatible API
        openrouter_key = (
            kwargs.get("api_key")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or settings.openrouter_api_key
        ).strip()
        if not openrouter_key:
            raise ValueError(
                "No OpenRouter API key. In Streamlit Secrets set OPENROUTER_API_KEY "
                "to the full key from https://openrouter.ai/keys (starts with sk-or-v1-)."
            )
        return ChatOpenAI(
            model=kwargs.get('model', settings.openrouter_chat_model),
            api_key=openrouter_key,
            base_url=kwargs.get('base_url', settings.openrouter_api_base),
            temperature=kwargs.get('temperature', settings.agent_temperature),
            max_tokens=kwargs.get('max_tokens', 1024),
            streaming=kwargs.get('streaming', False),
            request_timeout=kwargs.get('request_timeout', 60),
            default_headers={"HTTP-Referer": "https://pedir-bot.local", "X-Title": "PedIR Bot"},
            extra_body={
                "transforms": ["middle-out"],
            },
        )
    elif provider == "huggingface":
        # Hugging Face Inference Endpoints (TGI) are often OpenAI-compatible
        return ChatOpenAI(
            model=kwargs.get('model', settings.hf_chat_model),
            api_key=kwargs.get('api_key', settings.hf_api_key),
            base_url=kwargs.get('base_url', settings.hf_api_base),
            temperature=kwargs.get('temperature', settings.agent_temperature),
            max_tokens=kwargs.get('max_tokens', 4096),
            streaming=kwargs.get('streaming', False),
            request_timeout=kwargs.get('request_timeout', 120),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Legacy compatibility classes (kept for backward compatibility)
class LLMProvider:
    """Abstract base class for LLM providers (legacy compatibility)."""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError("Use get_langchain_llm instead")

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Generate a streaming response from the LLM."""
        raise NotImplementedError("Use get_langchain_llm instead")


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider (legacy compatibility)."""

    def __init__(self, **kwargs):
        """Initialize OpenAI provider."""
        self.langchain_llm = get_langchain_llm(provider="openai", **kwargs)
        logger.info(f"Initialized OpenAI provider (legacy mode)")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from OpenAI."""
        try:
            # Convert messages to LangChain format
            langchain_messages = _convert_messages_to_langchain(messages)
            response = self.langchain_llm.invoke(langchain_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Generate a streaming response from OpenAI."""
        try:
            langchain_messages = _convert_messages_to_langchain(messages)
            for chunk in self.langchain_llm.stream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Ollama local API provider (legacy compatibility)."""

    def __init__(self, **kwargs):
        """Initialize Ollama provider."""
        self.langchain_llm = get_langchain_llm(provider="ollama", **kwargs)
        logger.info(f"Initialized Ollama provider (legacy mode)")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from Ollama."""
        try:
            langchain_messages = _convert_messages_to_langchain(messages)
            response = self.langchain_llm.invoke(langchain_messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Generate a streaming response from Ollama."""
        try:
            langchain_messages = _convert_messages_to_langchain(messages)
            for chunk in self.langchain_llm.stream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            raise

def _convert_messages_to_langchain(messages: List[Dict[str, str]]) -> List:
    """Convert message dicts to LangChain message objects."""
    langchain_messages = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            langchain_messages.append(SystemMessage(content=content))
        elif role == 'assistant':
            langchain_messages.append(AIMessage(content=content))
        else:  # user or default
            langchain_messages.append(HumanMessage(content=content))

    return langchain_messages


class LMStudioProvider(LLMProvider):
    """LM Studio local API provider using OpenAI-compatible API."""

    def __init__(self,
                 model: str = None,
                 base_url: str = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1024):
        """
        Initialize LM Studio provider.

        Args:
            model: Model name (default from settings)
            base_url: LM Studio API base URL (default from settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        from openai import OpenAI

        self.model = model or settings.lmstudio_chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or settings.lmstudio_api_base

        self.client = OpenAI(
            api_key="lm-studio",  # LM Studio doesn't require real key
            base_url=self.base_url
        )

        logger.info(f"Initialized LM Studio provider with model: {self.model}")
        logger.info(f"LM Studio API base: {self.base_url}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from LM Studio.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text
        """
        try:
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "stream": False,
            }
            if 'stop' in kwargs:
                api_kwargs['stop'] = kwargs['stop']
            response = self.client.chat.completions.create(**api_kwargs)

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from LM Studio: {e}")
            logger.error(f"Make sure LM Studio is running with a chat model loaded")
            raise

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """
        Generate a streaming response from LM Studio.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Yields:
            Response chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming from LM Studio: {e}")
            raise

class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider using OpenAI-compatible API."""

    def __init__(self,
                 model: str = None,
                 base_url: str = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1024):
        """
        Initialize OpenRouter provider.

        Args:
            model: Model name (default from settings)
            base_url: OpenRouter API base URL (default from settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        from openai import OpenAI

        self.model = model or settings.openrouter_chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or settings.openrouter_api_base
        self.api_key = settings.openrouter_api_key

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"HTTP-Referer": "https://pedir-bot.local", "X-Title": "PedIR Bot"}
        )

        logger.info(f"Initialized OpenRouter provider with model: {self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {e}")
            raise

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Generate a streaming response from OpenRouter."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming from OpenRouter: {e}")
            raise


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference Endpoint provider using OpenAI-compatible API."""

    def __init__(self,
                 model: str = None,
                 base_url: str = None,
                 api_key: str = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1024):
        """Initialize Hugging Face provider."""
        from openai import OpenAI

        self.model = model or settings.hf_chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or settings.hf_api_base
        self.api_key = api_key or settings.hf_api_key

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(f"Initialized Hugging Face provider with model: {self.model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from Hugging Face."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Hugging Face: {e}")
            raise

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Generate a streaming response from Hugging Face."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming from Hugging Face: {e}")
            raise


def get_llm_provider(provider: str = None, **kwargs) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider (legacy compatibility).

    Args:
        provider: 'openai', 'ollama', or 'lmstudio' (default from settings)
        **kwargs: Additional parameters for provider initialization

    Returns:
        LLMProvider instance
    """
    provider = provider or settings.llm_provider

    if provider == "openai":
        return OpenAIProvider(**kwargs)
    elif provider == "ollama":
        return OllamaProvider(**kwargs)
    elif provider == "lmstudio":
        return LMStudioProvider(**kwargs)
    elif provider == "openrouter":
        return OpenRouterProvider(**kwargs)
    elif provider == "huggingface":
        return HuggingFaceProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
