from datetime import datetime, timedelta, timezone

from app.core.config import Settings
from app.llm.base import ProviderFailure
from app.services.assistant_service import answer_question
from app.services.experiment_memory_service import bounded_context, retrieve_similar_experiments
from app.services.assistant_service import AssistantResult
from app.services.security import get_current_user
from test_training import classification_csv, prepare_experiment
from test_ownership import request


MEMORY = [{"experiment_id": "owned", "dataset": {"filename": "safe.csv", "rows": 10, "columns": 3}, "objective": "predict target", "status": "COMPLETED", "task_type": "binary_classification", "target_column": "target", "pipeline_plan": {"primary_metric": "accuracy"}, "trained_models": ["logistic_regression", "random_forest", "xgboost"], "selected_model": "random_forest", "selected_model_validation_metrics": {"accuracy": 0.826, "roc_auc": 0.88}, "evaluation": {"primary_metric": "accuracy", "validation_comparison": [{"model_run_id": "hidden-one", "model_name": "logistic_regression", "metrics": {"accuracy": 0.80}}, {"model_run_id": "hidden-two", "model_name": "random_forest", "metrics": {"accuracy": 0.826, "roc_auc": 0.88}}, {"model_run_id": "hidden-three", "model_name": "xgboost", "metrics": {"accuracy": 0.81}}], "final_test_metrics": {"accuracy": 0.8156, "precision": 0.8, "recall": 0.79, "f1": 0.79, "roc_auc": 0.91}, "confusion_matrix": None}, "optimization": None, "prediction_count": 0, "report_count": 0, "updated_at": "2026-01-01T00:00:00+00:00"}]


class Provider:
    def __init__(self, name: str, failure: bool = False) -> None:
        self.name, self.failure, self.calls, self.prompts = name, failure, 0, []
    def generate_text(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.failure:
            raise ProviderFailure(self.name, "unavailable", "unavailable")
        return "The selected model is appropriate for this classification task."


def test_verified_answer_uses_primary_and_includes_persisted_metrics() -> None:
    primary, fallback = Provider("gemini"), Provider("ollama")
    result = answer_question("Which model performed best?", MEMORY, primary, fallback, 1000)
    assert result.provider_used == "gemini" and primary.calls == 1 and fallback.calls == 0
    assert "Random Forest" in result.answer and "validation Accuracy of **82.60%**" in result.answer
    assert "final test Accuracy was **81.56%**" in result.answer


def test_provider_failure_falls_back_but_missing_memory_does_not() -> None:
    primary, fallback = Provider("gemini", True), Provider("ollama")
    assert answer_question("Summarize", MEMORY, primary, fallback, 1000).provider_used == "ollama"
    primary, fallback = Provider("gemini"), Provider("ollama")
    result = answer_question("Unknown", [], primary, fallback, 1000)
    assert result.provider_used == "verified_retrieval" and primary.calls == fallback.calls == 0


def test_bounded_context_excludes_paths_and_truncates() -> None:
    context = bounded_context(MEMORY, 80)
    assert len(context) == 80
    assert "C:\\" not in context and "secret" not in context.lower()


def test_question_aware_metric_summary_and_no_internal_leakage() -> None:
    primary, fallback = Provider("gemini"), Provider("ollama")
    metric = answer_question("What was the test ROC-AUC?", MEMORY, primary, fallback, 1000)
    summary = answer_question("Summarize this experiment.", MEMORY, primary, fallback, 1000)
    assert metric.answer == "The final test ROC-AUC was **91.00%**."
    assert "### Experiment Summary" in summary.answer and "| Accuracy | 81.56% |" in summary.answer
    assert "owned" not in summary.answer and "binary_classification" not in summary.answer


def test_accuracy_keeps_validation_and_final_test_values_separate() -> None:
    answer = answer_question("What was the accuracy?", MEMORY, Provider("gemini"), Provider("ollama"), 1000).answer
    assert "**Validation accuracy:** 82.60%." in answer
    assert "**Final test accuracy:** 81.56%." in answer


def test_regression_accuracy_question_returns_verified_regression_guidance() -> None:
    regression_memory = [{**MEMORY[0], "task_type": "regression", "evaluation": {**MEMORY[0]["evaluation"], "final_test_metrics": {"mae": 12000.0, "rmse": 18000.0, "r2": 0.81}}}]
    answer = answer_question("What is the accuracy?", regression_memory, Provider("gemini"), Provider("ollama"), 1000).answer
    assert "Accuracy applies to classification" in answer
    assert "MAE" in answer and "RMSE" in answer and "R²" in answer


def test_why_selected_and_model_comparison_use_verified_values() -> None:
    why = answer_question("Why was Random Forest selected?", MEMORY, Provider("gemini"), Provider("ollama"), 1000).answer
    comparison = answer_question("Compare baseline models", MEMORY, Provider("gemini"), Provider("ollama"), 1000).answer
    assert "Logistic Regression: **80.00%**" in why and "XGBoost: **81.00%**" in why
    assert "| Random Forest | 82.60% |" in comparison


def test_missing_metric_is_not_fabricated_and_provider_evidence_is_equivalent() -> None:
    incomplete = [{**MEMORY[0], "selected_model_validation_metrics": {}, "evaluation": {**MEMORY[0]["evaluation"], "final_test_metrics": {}}}]
    primary, fallback = Provider("gemini", True), Provider("ollama")
    answer_question("Which model was selected?", incomplete, primary, fallback, 1000)
    assert "The verified validation value is not recorded." in answer_question("Which model was selected?", incomplete, Provider("gemini"), Provider("ollama"), 1000).answer
    assert primary.prompts[0] == fallback.prompts[0]
    assert "hidden-one" not in fallback.prompts[0] and "binary_classification" not in fallback.prompts[0]


def test_unsupported_question_is_not_filled_with_unverified_content() -> None:
    result = answer_question("What is the passenger's home address?", MEMORY, Provider("gemini"), Provider("ollama"), 1000)

    assert result.answer == "I don't have verified experiment data to answer that."


def test_recent_comparison_deduplicates_identical_experiments() -> None:
    duplicate = {**MEMORY[0], "experiment_id": "different"}
    result = answer_question("Compare my recent experiments.", [MEMORY[0], duplicate], Provider("gemini"), Provider("ollama"), 1000)
    assert "Recent Experiment Comparison" not in result.answer


def test_cross_experiment_comparison_uses_only_compatible_persisted_metric():
    previous = {"experiment_id": "prior", "dataset_filename": "other.csv", "objective": "predict target", "task_type": "binary_classification", "target_column": "target", "selected_model": "xgboost", "primary_metric": "accuracy", "validation_score": 0.8, "test_score": 0.78, "match_reason": "Same task and target"}
    provider = Provider("gemini")
    answer = answer_question("Compare this with my previous experiments", MEMORY, provider, Provider("ollama"), 1000, [previous]).answer
    assert "### Current and related experiment results" in answer
    assert "| Previous | other.csv | XGBoost | 80.00% | 78.00% |" in answer
    assert "prior" not in provider.prompts[0]
    incompatible = {**previous, "primary_metric": "roc_auc"}
    answer = answer_question("Compare this with my previous experiments", MEMORY, Provider("gemini"), Provider("ollama"), 1000, [incompatible]).answer
    assert "different primary metrics" in answer


def test_similar_experiment_retrieval_is_bounded_and_uses_dataset_identity(monkeypatch):
    from types import SimpleNamespace

    candidates = [SimpleNamespace(id="same", user_objective="predict target survival rate", dataset_id="dataset-1"), SimpleNamespace(id="other", user_objective="predict target survival rate", dataset_id="dataset-2")]
    class Result:
        def all(self): return candidates
    class Database:
        def scalars(self, _statement): return Result()
    memories = {
        "same": {"experiment_id": "same", "dataset": {"id": "dataset-1", "filename": "renamed.csv"}, "objective": "predict target survival", "task_type": "binary_classification", "target_column": "survived", "evaluation": {"primary_metric": "accuracy", "final_test_metrics": {"accuracy": 0.8}}, "selected_model_validation_metrics": {"accuracy": 0.82}, "selected_model": "random_forest"},
        "other": {"experiment_id": "other", "dataset": {"id": "dataset-2", "filename": "same-name.csv"}, "objective": "predict target survival rate", "task_type": "binary_classification", "target_column": "survived", "evaluation": {"primary_metric": "accuracy", "final_test_metrics": {"accuracy": 0.75}}, "selected_model_validation_metrics": {"accuracy": 0.78}, "selected_model": "xgboost"},
    }
    monkeypatch.setattr("app.services.experiment_memory_service._memory_for_experiment", lambda _db, experiment: memories[experiment.id])
    current = {**MEMORY[0], "experiment_id": "current", "dataset": {"id": "dataset-1", "filename": "same-name.csv"}, "objective": "predict survival rate"}
    results = retrieve_similar_experiments(Database(), "user", current, limit=1)
    assert len(results) == 1 and results[0]["experiment_id"] == "same"
    assert "Same saved dataset" in results[0]["match_reason"]


def test_assistant_requires_auth_and_scopes_experiment_memory(test_app, monkeypatch) -> None:
    experiment_id = prepare_experiment(test_app, classification_csv(["yes", "no"]), target="target", task_type="binary_classification", models=["logistic_regression"])
    captured = []
    def fake_answer(question, memories, *_args):
        captured.extend(memories)
        return AssistantResult("VERIFIED RESULT: safe", "gemini")
    monkeypatch.setattr("app.api.routes.assistant.answer_question", fake_answer)
    response = request(test_app, "POST", "/api/v1/assistant/query", json={"question": "What dataset did I use?", "experiment_id": experiment_id})
    assert response.status_code == 200 and response.json()["provider_used"] == "gemini"
    assert captured and captured[0]["experiment_id"] == experiment_id
    test_app.dependency_overrides.pop(get_current_user, None)
    assert request(test_app, "POST", "/api/v1/assistant/query", json={"question": "What dataset did I use?"}).status_code == 401
