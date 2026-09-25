from typing import Any

from app.llm.base import LLMProvider, ProviderFailure


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        if not self.api_key:
            raise ProviderFailure(self.name, "unavailable", "Gemini API key is not configured.")

        try:
            from google import genai
            from google.genai import types

            with genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=response_schema,
                    ),
                )
            if not response.text:
                raise ProviderFailure(self.name, "malformed_output", "Gemini returned no plan.")
            return response.text
        except ProviderFailure:
            raise
        except TimeoutError as error:
            raise ProviderFailure(self.name, "timeout", "Gemini request timed out.") from error
        except Exception as error:
            category = self._failure_category(error)
            raise ProviderFailure(self.name, category, f"Gemini request failed: {category}.") from error

    def generate_text(self, prompt: str) -> str:
        if not self.api_key:
            raise ProviderFailure(self.name, "unavailable", "Gemini API key is not configured.")
        try:
            from google import genai
            from google.genai import types
            with genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000))) as client:
                response = client.models.generate_content(model=self.model, contents=prompt)
            if not response.text:
                raise ProviderFailure(self.name, "malformed_output", "Gemini returned no answer.")
            return response.text[:4000]
        except ProviderFailure:
            raise
        except TimeoutError as error:
            raise ProviderFailure(self.name, "timeout", "Gemini request timed out.") from error
        except Exception as error:
            raise ProviderFailure(self.name, self._failure_category(error), "Gemini request failed.") from error

    @staticmethod
    def _failure_category(error: Exception) -> str:
        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        message = str(error).lower()
        if status_code == 429 or "429" in message or "rate limit" in message:
            return "rate_limited"
        if "timeout" in message or "timed out" in message:
            return "timeout"
        if status_code in {500, 502, 503, 504}:
            return "unavailable"
        return "provider_error"
