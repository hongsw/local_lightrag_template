"""
OpenAI LLM provider for OKT-RAG.

Supports GPT-4, GPT-4 Turbo, GPT-4o, and GPT-3.5 models.
"""

import os
from typing import Optional

import openai

from .base import BaseLLMProvider, CompletionResult, Message


# Pricing per 1K tokens (as of 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI LLM provider.

    Supports all OpenAI chat completion models with cost tracking.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI LLM provider.

        Args:
            model: Model identifier.
            api_key: OpenAI API key (uses env var if not provided).
        """
        pricing = OPENAI_PRICING.get(model, {"input": 0.01, "output": 0.03})

        super().__init__(
            model=model,
            cost_input=pricing["input"],
            cost_output=pricing["output"],
        )

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        self._client = openai.AsyncOpenAI(api_key=self._api_key)

    async def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> CompletionResult:
        """
        Generate completion using OpenAI API.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional OpenAI parameters.

        Returns:
            CompletionResult with generated content.
        """
        # Convert Message objects to dict format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage

        cost = self.calculate_cost(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

        return CompletionResult(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost,
            finish_reason=choice.finish_reason or "stop",
            metadata={
                "id": response.id,
                "created": response.created,
            },
        )

    def __repr__(self) -> str:
        return f"OpenAILLMProvider(model={self._model})"
