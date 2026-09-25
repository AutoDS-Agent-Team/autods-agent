from typing import Any

import httpx

from app.llm.base import LLMProvider, ProviderFailure


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "format": response_schema,
                    "options": {"temperature": 0},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            if not content:
                raise ProviderFailure(self.name, "malformed_output", "Ollama returned no plan.")
            return content
        except ProviderFailure:
            raise
        except httpx.TimeoutException as error:
            raise ProviderFailure(self.name, "timeout", "Ollama request timed out.") from error
        except httpx.HTTPStatusError as error:
            category = "rate_limited" if error.response.status_code == 429 else "provider_error"
            raise ProviderFailure(self.name, category, f"Ollama request failed: {category}.") from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderFailure(self.name, "unavailable", "Ollama is unavailable.") from error

    def generate_text(self, prompt: str) -> str:
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False, "options": {"temperature": 0}}, timeout=self.timeout_seconds)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            if not content:
                raise ProviderFailure(self.name, "malformed_output", "Ollama returned no answer.")
            return content[:4000]
        except ProviderFailure:
            raise
        except httpx.TimeoutException as error:
            raise ProviderFailure(self.name, "timeout", "Ollama request timed out.") from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderFailure(self.name, "unavailable", "Ollama is unavailable.") from error
