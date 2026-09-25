import logging
from dataclasses import dataclass

from pydantic import ValidationError

from app.core.exceptions import PlanningFailureError
from app.llm.base import LLMProvider, ProviderFailure
from app.schemas.pipeline_plan import PipelinePlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedPlan:
    plan: PipelinePlan
    provider_used: str


class LLMRouter:
    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        max_retries: int,
    ) -> None:
        self.providers = (primary, fallback)
        self.max_retries = max_retries

    def generate_plan(
        self,
        prompt: str,
        *,
        confirmed_task_type: str,
        confirmed_target_column: str,
    ) -> RoutedPlan:
        schema = PipelinePlan.model_json_schema()
        failures: list[str] = []

        for provider_index, provider in enumerate(self.providers):
            if provider_index:
                logger.warning("LLM fallback attempted provider=%s", provider.name)
            for attempt in range(self.max_retries + 1):
                logger.info("LLM provider attempted provider=%s attempt=%s", provider.name, attempt + 1)
                try:
                    output = provider.generate_structured(prompt, schema)
                    plan = PipelinePlan.validate_provider_output(
                        output,
                        confirmed_task_type=confirmed_task_type,
                        confirmed_target_column=confirmed_target_column,
                    )
                    logger.info("LLM plan accepted final_provider=%s", provider.name)
                    return RoutedPlan(plan=plan, provider_used=provider.name)
                except ProviderFailure as error:
                    category = error.category
                except ValidationError:
                    category = "contract_validation"
                failures.append(f"{provider.name}:{category}")
                logger.warning(
                    "LLM provider failed provider=%s category=%s attempt=%s",
                    provider.name,
                    category,
                    attempt + 1,
                )

        logger.error("All LLM providers failed categories=%s", ",".join(failures))
        raise PlanningFailureError(
            "Pipeline planning failed because no provider returned a valid plan."
        )
