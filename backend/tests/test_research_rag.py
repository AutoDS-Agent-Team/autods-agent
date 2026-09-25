from app.services.research_rag_service import retrieve_research
from app.services.optimization_service import _cost_profile
import optuna


def test_offline_retrieval_returns_source_backed_automl_evidence():
    results = retrieve_research("Automated machine learning pipeline optimization with meta-learning under compute time budget")
    assert results
    assert any("Auto-Sklearn 2.0" in item["title"] or "Automated Machine Learning" in item["title"] for item in results)
    assert all(item["source_url"].startswith(("https://arxiv.org/", "https://proceedings.")) for item in results)
    assert all(item["evidence_summary"] and item["relevance_score"] > 0 for item in results)


def test_unrelated_query_returns_no_citations():
    assert retrieve_research("qzvplmxx 8127 frobnicator") == []


def test_optimization_cost_profile_reports_fast_near_best_without_changing_best():
    study = optuna.create_study(direction="minimize")
    for score, seconds in ((100.0, 8.0), (100.5, 2.0), (125.0, 1.0)):
        trial = optuna.trial.create_trial(value=score, user_attrs={"duration_seconds": seconds})
        study.add_trial(trial)
    profile = _cost_profile(study, "rmse", 120)
    assert profile["fastest_near_optimal"]["duration_seconds"] == 2.0
    assert profile["budget_seconds"] == 120
    assert study.best_value == 100.0
