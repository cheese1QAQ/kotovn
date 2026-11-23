"""Protocol for local quantized LLM inference backends (llama.cpp, mock, etc.)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMGenerationConfig(BaseModel):
    """Per-request generation controls."""

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=16, le=2048)
    stop: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Generation result with fine-grained timing telemetry."""

    text: str
    output_tokens: int = Field(default=0, ge=0)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    total_ms: float = Field(default=0.0, ge=0.0)

    @property
    def decode_tps(self) -> float:
        """Compute decode throughput in tokens per second."""
        if self.total_ms <= 0.0 or self.output_tokens <= 0:
            return 0.0
        return round(self.output_tokens / (self.total_ms / 1000.0), 2)


@runtime_checkable
class LLMBackend(Protocol):
    """Interface implemented by local quantized model runtimes and test doubles."""

    @property
    def model_name(self) -> str: ...

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse: ...

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMGenerationConfig | None = None,
    ) -> AsyncIterator[str]: ...
