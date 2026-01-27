"""
Base LLM provider interface for OKT-RAG.

Defines the protocol for model-agnostic LLM operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Optional, Literal, runtime_checkable


@dataclass
class Message:
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class CompletionResult:
    """Result from LLM completion."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    finish_reason: str = "stop"
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for LLM providers.

    Enables model-agnostic generation operations.
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def cost_per_1k_input_tokens(self) -> float:
        """Return cost per 1K input tokens."""
        ...

    @property
    def cost_per_1k_output_tokens(self) -> float:
        """Return cost per 1K output tokens."""
        ...

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> CompletionResult:
        """
        Generate completion from messages.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters.

        Returns:
            CompletionResult with generated content and metadata.
        """
        ...

    async def complete_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> CompletionResult:
        """
        Generate completion from text prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters.

        Returns:
            CompletionResult with generated content.
        """
        ...


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides common functionality and enforces interface.
    """

    def __init__(
        self,
        model: str,
        cost_input: float,
        cost_output: float,
    ):
        self._model = model
        self._cost_input = cost_input
        self._cost_output = cost_output

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cost_per_1k_input_tokens(self) -> float:
        return self._cost_input

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return self._cost_output

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        input_cost = (input_tokens / 1000) * self._cost_input
        output_cost = (output_tokens / 1000) * self._cost_output
        return input_cost + output_cost

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> CompletionResult:
        """Generate completion from messages."""
        pass

    async def complete_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> CompletionResult:
        """Generate completion from text prompt."""
        messages = []

        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        messages.append(Message(role="user", content=prompt))

        return await self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
