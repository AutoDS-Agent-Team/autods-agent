from app.llm.base import ProviderFailure
from app.services import reflection_service


MEMORY = {"experiment_id": "hidden", "dataset": {"filename": "safe.csv", "rows": 10, "columns": 3}, "objective": "predict", "task_type": "binary_classification", "target_column": "target", "selected_model": "random_forest", "selected_model_validation_metrics": {"accuracy": 0.82}, "evaluation": {"primary_metric": "accuracy", "validation_comparison": [], "final_test_metrics": {"accuracy": 0.81, "precision": 0.86, "recall": 0.62}, "confusion_matrix": None}, "optimization": None}


class Provider:
    def __init__(self, name, output=None, failure=False): self.name, self.output, self.failure, self.calls = name, output, failure, 0
    def generate_structured(self, *_args):
        self.calls += 1
        if self.failure: raise ProviderFailure(self.name, "unavailable", "unavailable")
        return self.output


def setup(monkeypatch):
    monkeypatch.setattr(reflection_service, "retrieve_verified_memory", lambda *_args: [MEMORY])
    monkeypatch.setattr(reflection_service, "get_explainability", lambda *_args: type("E", (), {"global_feature_importance": [type("F", (), {"feature_name": "Age"})(), type("F", (), {"feature_name": "Fare"})()]})())


def test_reflection_uses_verified_evidence_and_does_not_mutate(monkeypatch, test_app):
    setup(monkeypatch)
    original = dict(MEMORY["evaluation"]["final_test_metrics"])
    provider = Provider("gemini", {"executive_summary": "review", "model_assessment": "review", "strengths": [], "weaknesses": [], "important_features": [], "recommendations": []})
    result = reflection_service.create_reflection("experiment", "user", None, test_app.state.testing_settings, (provider, Provider("ollama", failure=True)))
    assert result.provider_used == "gemini" and "82.00%" in result.model_assessment
    assert "1.00% lower than validation" in result.model_assessment
    assert len(result.recommendations) >= 2
    assert result.research_status in {"EVIDENCE_FOUND", "NO_RELEVANT_EVIDENCE"}
    assert all(item.source_url.startswith("https://") for item in result.research_evidence)
    assert result.decision_explanations
    assert MEMORY["evaluation"]["final_test_metrics"] == original


def test_reflection_fallback_and_hallucinated_advice_are_sanitized(monkeypatch, test_app):
    setup(monkeypatch)
    bad = Provider("gemini", {"executive_summary": "999 accuracy", "model_assessment": "x", "strengths": [], "weaknesses": [], "important_features": [{"feature": "Invented", "explanation": "x"}], "recommendations": [{"title": "Use magic", "reason": "999", "suggested_action": "shell command", "requires_retraining": True}]})
    fallback = Provider("ollama", failure=True)
    result = reflection_service.create_reflection("experiment", "user", None, test_app.state.testing_settings, (bad, fallback))
    assert result.provider_used == "gemini"
    assert all(item.feature != "Invented" for item in result.important_features)
    assert "999" not in result.executive_summary


def test_both_unavailable_returns_verified_fallback(monkeypatch, test_app):
    setup(monkeypatch)
    result = reflection_service.create_reflection("experiment", "user", None, test_app.state.testing_settings, (Provider("gemini", failure=True), Provider("ollama", failure=True)))
    assert result.provider_used == "verified_reflection" and result.status == "FALLBACK"
    assert result.research_status in {"EVIDENCE_FOUND", "NO_RELEVANT_EVIDENCE"}


def test_recommendation_citations_are_local_and_optional():
    recommendation = reflection_service._verified_review(MEMORY, []).recommendations[0]
    linked, evidence = reflection_service._link_research_to_recommendations([recommendation])
    assert linked[0].research_sources
    assert all(source.source_url in {item.source_url for item in evidence} for source in linked[0].research_sources)
    assert all(source.source_url.startswith("https://") for source in linked[0].research_sources)
    unsupported = type(recommendation)(title="Use a quasar flux capacitor", reason="Resolve qzvplmxx", suggested_action="Apply frobnicator telemetry to a quantum turbine.", requires_retraining=False)
    unlinked, no_evidence = reflection_service._link_research_to_recommendations([unsupported])
    assert not unlinked[0].research_sources and not no_evidence


def test_decision_explanations_trace_selection_preprocessing_and_profile():
    from types import SimpleNamespace

    memory = {**MEMORY, "pipeline_plan": {"models": ["random_forest", "xgboost"], "primary_metric": "accuracy", "numeric_imputation": "median", "categorical_imputation": "most_frequent", "categorical_encoding": "one_hot", "numeric_scaling": "standard", "excluded_columns": ["record_id"]}, "evaluation": {**MEMORY["evaluation"], "validation_comparison": [{"model_name": "random_forest", "metrics": {"accuracy": 0.82}}, {"model_name": "xgboost", "metrics": {"accuracy": 0.80}}]}}
    profile = SimpleNamespace(summary=SimpleNamespace(total_missing_values=4, duplicate_row_count=2), columns=[SimpleNamespace(name="Age", logical_type="numerical", missing_count=3), SimpleNamespace(name="target", logical_type="categorical", missing_count=1)])
    explanations = reflection_service._decision_explanations(memory, profile)
    stages = {item["stage"] for item in explanations}
    assert {"Baseline selection", "Preprocessing", "Feature exclusions", "Dataset quality", "Missing-value preparation context"} <= stages
    selected = next(item for item in explanations if item["stage"] == "Baseline selection")
    assert "matched the best validation value" in selected["rationale"]
    assert selected["evidence_source"] == "persisted validation comparison"
