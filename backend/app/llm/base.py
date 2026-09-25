from abc import ABC, abstractmethod
from typing import Any


class ProviderFailure(Exception):
    def __init__(self, provider: str, category: str, detail: str) -> None:
        super().__init__(detail)
        self.provider = provider
        self.category = category
        self.detail = detail


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str | dict[str, Any]:
        """Return provider output without trusting or executing it."""
