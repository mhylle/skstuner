"""LLM provider abstraction layer for supporting multiple LLM backends"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import requests
from anthropic import Anthropic, APIError

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Generate text using the LLM

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name being used"""
        pass


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude provider

        Args:
            api_key: Anthropic API key
            model: Claude model to use

        Raises:
            ValueError: If API key is empty
        """
        if not api_key or not api_key.strip():
            raise ValueError("Anthropic API key cannot be empty")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Claude provider with model: {model}")

    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Generate text using Claude API

        Args:
            prompt: The prompt to send to Claude
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            APIError: If Claude API call fails
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            if not response.content or not hasattr(response.content[0], "text"):
                raise APIError(f"Unexpected API response format. Response: {response}")

            return response.content[0].text

        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def get_model_name(self) -> str:
        """Return the Claude model name"""
        return self.model


class OllamaProvider(LLMProvider):
    """Ollama server provider for local LLM inference"""

    def __init__(
        self,
        base_url: str = "https://ollama:11434",
        model: str = "llama3.3",
        timeout: int = 120,
    ):
        """
        Initialize Ollama provider

        Args:
            base_url: Base URL of Ollama server (e.g., https://ollama:11434)
            model: Model name to use (e.g., llama3.3)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If parameters are invalid
        """
        if not base_url or not base_url.strip():
            raise ValueError("Ollama base_url cannot be empty")
        if not model or not model.strip():
            raise ValueError("Ollama model cannot be empty")

        # Normalize URL (remove trailing slash)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"

        logger.info(f"Initialized Ollama provider with model: {model} at {base_url}")

    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Generate text using Ollama API

        Args:
            prompt: The prompt to send to Ollama
            max_tokens: Maximum tokens to generate (mapped to num_predict)

        Returns:
            Generated text

        Raises:
            requests.exceptions.RequestException: If Ollama API call fails
            ValueError: If response is invalid
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
            },
        }

        try:
            logger.debug(f"Calling Ollama API at {self.api_url}")
            response = requests.post(
                self.api_url, json=payload, timeout=self.timeout, verify=False
            )
            response.raise_for_status()

            result = response.json()

            if "response" not in result:
                raise ValueError(
                    f"Invalid Ollama response format. Expected 'response' field. Got: {result}"
                )

            generated_text = result["response"]
            logger.debug(f"Generated {len(generated_text)} characters from Ollama")

            return generated_text

        except requests.exceptions.Timeout:
            error_msg = f"Ollama request timed out after {self.timeout}s"
            logger.error(error_msg)
            raise requests.exceptions.RequestException(error_msg)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to Ollama at {self.base_url}: {e}"
            logger.error(error_msg)
            raise requests.exceptions.RequestException(error_msg)
        except requests.exceptions.HTTPError as e:
            error_msg = f"Ollama API returned error: {e}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error calling Ollama: {e}"
            logger.error(error_msg)
            raise

    def get_model_name(self) -> str:
        """Return the Ollama model name"""
        return self.model


def create_provider(
    provider_type: str,
    anthropic_api_key: Optional[str] = None,
    claude_model: str = "claude-sonnet-4-20250514",
    ollama_base_url: str = "https://ollama:11434",
    ollama_model: str = "llama3.3",
    ollama_timeout: int = 120,
) -> LLMProvider:
    """
    Factory function to create LLM providers

    Args:
        provider_type: Type of provider ("claude" or "ollama")
        anthropic_api_key: Anthropic API key (required for Claude)
        claude_model: Claude model name
        ollama_base_url: Ollama server URL
        ollama_model: Ollama model name
        ollama_timeout: Ollama request timeout

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider_type is invalid or required parameters are missing
    """
    provider_type = provider_type.lower()

    if provider_type == "claude":
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key is required for Claude provider")
        return ClaudeProvider(api_key=anthropic_api_key, model=claude_model)

    elif provider_type == "ollama":
        return OllamaProvider(
            base_url=ollama_base_url, model=ollama_model, timeout=ollama_timeout
        )

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. Must be 'claude' or 'ollama'"
        )
