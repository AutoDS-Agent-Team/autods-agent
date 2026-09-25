from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.llm.base import ProviderFailure
from app.services.experiment_memory_service import bounded_context


@dataclass(frozen=True)
class AssistantResult:
    answer: str
    provider_used: str


PERCENT_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "r2"}


def _label(value: str | None, labels: dict[str, str] | None = None) -> str:
    if not value:
        return "Not recorded"
    rendered = (labels or {}).get(value, value.replace("_", " ").title())
    return "XGBoost" if rendered.lower() == "xgboost" else rendered


def _metric_label(metric: str) -> str:
    return {"f1": "F1 Score", "roc_auc": "ROC-AUC", "mae": "MAE", "rmse": "RMSE", "r2": "R²"}.get(metric, metric.replace("_", " ").title())


def _format_metric(metric: str, value: Any) -> str:
    if value is None:
        return "Not applicable"
    number = float(value)
    return f"{number * 100:.2f}%" if metric in PERCENT_METRICS else f"{number:.3f}"


def _deduplicate(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique, seen = [], set()
    for memory in memories:
        key = memory.get("experiment_id")
        if key not in seen:
            seen.add(key)
            unique.append(memory)
    return unique


def _evaluation(memory: dict[str, Any]) -> dict[str, Any]:
    return memory.get("evaluation") or {}


def _primary_metric(memory: dict[str, Any]) -> str | None:
    return _evaluation(memory).get("primary_metric") or (memory.get("pipeline_plan") or {}).get("primary_metric")


def _validation_rows(memory: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in _evaluation(memory).get("validation_comparison") or [] if isinstance(row, dict)]


def _selected_validation_metrics(memory: dict[str, Any]) -> dict[str, Any]:
    direct = memory.get("selected_model_validation_metrics")
    if isinstance(direct, dict):
        return direct
    selected = memory.get("selected_model")
    row = next((item for item in _validation_rows(memory) if item.get("model_name") == selected), None)
    return row.get("metrics", {}) if row and isinstance(row.get("metrics"), dict) else {}


def _selection_answer(memory: dict[str, Any]) -> str:
    model, metric = memory.get("selected_model"), _primary_metric(memory)
    if not model:
        return "I don't have verified experiment data to answer that."
    validation = _selected_validation_metrics(memory)
    if metric and validation.get(metric) is not None:
        first = f"**{_label(model)}** was selected as the best baseline model with a validation {_metric_label(metric)} of **{_format_metric(metric, validation[metric])}**."
    elif metric:
        first = f"**{_label(model)}** was selected as the best baseline model based on validation {_metric_label(metric)}. The verified validation value is not recorded."
    else:
        first = f"**{_label(model)}** was selected as the best baseline model."
    final = _evaluation(memory).get("final_test_metrics") or {}
    if metric and final.get(metric) is not None:
        return first + f"\n\nIts final test {_metric_label(metric)} was **{_format_metric(metric, final[metric])}**."
    return first


def _validation_table(memory: dict[str, Any]) -> str | None:
    metric, rows = _primary_metric(memory), _validation_rows(memory)
    if not metric or not rows:
        return None
    rows = [row for row in rows if isinstance(row.get("metrics"), dict) and row["metrics"].get(metric) is not None]
    if not rows:
        return None
    lines = ["### Validation Results", "", f"| Model | {_metric_label(metric)} |", "|---|---:|"]
    lines += [f"| {_label(row.get('model_name'))} | {_format_metric(metric, row['metrics'][metric])} |" for row in rows]
    return "\n".join(lines)


def _why_selected_answer(memory: dict[str, Any]) -> str:
    selection = _selection_answer(memory)
    metric, model = _primary_metric(memory), memory.get("selected_model")
    if not metric or not model:
        return selection
    alternatives = []
    for row in _validation_rows(memory):
        value = (row.get("metrics") or {}).get(metric)
        if row.get("model_name") != model and value is not None:
            alternatives.append(f"{_label(row.get('model_name'))}: **{_format_metric(metric, value)}**")
    return selection + ("\n\nOther baseline validation " + _metric_label(metric) + " scores were " + "; ".join(alternatives) + "." if alternatives else "")


def _accuracy_answer(memory: dict[str, Any]) -> str:
    validation, final = _selected_validation_metrics(memory), _evaluation(memory).get("final_test_metrics") or {}
    if memory.get("task_type") == "regression":
        metrics = [f"**{_metric_label(name)}:** {_format_metric(name, value)}" for name, value in final.items() if name in {"mae", "rmse", "r2"}]
        return "Accuracy applies to classification, not this regression experiment. " + ("Verified final-test regression metrics: " + " · ".join(metrics) + "." if metrics else "Run evaluation to produce verified MAE, RMSE, and R² values.")
    lines = []
    if validation.get("accuracy") is not None:
        lines.append(f"**Validation accuracy:** {_format_metric('accuracy', validation['accuracy'])}.")
    if final.get("accuracy") is not None:
        lines.append(f"**Final test accuracy:** {_format_metric('accuracy', final['accuracy'])}.")
    return "\n\n".join(lines) if lines else "I don't have verified experiment data to answer that."


def _summary(memory: dict[str, Any]) -> str:
    metric, model = _primary_metric(memory), memory.get("selected_model")
    lines = ["### Experiment Summary", "", f"**Dataset:** {memory['dataset'].get('filename') or 'Not recorded'}", f"**Task:** {_label(memory.get('task_type'))}", f"**Target:** {memory.get('target_column') or 'Not recorded'}", f"**Selected model:** {_label(model)}", f"**Selection metric:** {_metric_label(metric) if metric else 'Not recorded'}"]
    validation_table = _validation_table(memory)
    if validation_table:
        lines += ["", validation_table]
    final = _evaluation(memory).get("final_test_metrics") or {}
    if final:
        lines += ["", "### Final Test Performance", "", "| Metric | Score |", "|---|---:|"]
        lines += [f"| {_metric_label(name)} | {_format_metric(name, value)} |" for name, value in final.items()]
    interpretation, validation = [], _selected_validation_metrics(memory)
    if model and metric and validation.get(metric) is not None:
        interpretation.append(f"{_label(model)} was selected using validation {_metric_label(metric)} of {_format_metric(metric, validation[metric])}.")
    if metric and final.get(metric) is not None:
        interpretation.append(f"Its final test {_metric_label(metric)} was {_format_metric(metric, final[metric])}; this is reported separately from validation selection.")
    optimization = memory.get("optimization")
    if optimization:
        interpretation.append(f"Optimization was {_label(optimization.get('status'))} for {_label(optimization.get('model_name'))} after {optimization.get('trial_count', 0)} trials.")
    if interpretation:
        lines += ["", "### Interpretation", "", *interpretation[:3]]
    return "\n".join(lines + ["", "*Based on verified AutoDS experiment results.*"])


def _answer_for_question(question: str, memories: list[dict[str, Any]]) -> str:
    selected, lower = memories[0], question.lower()
    why_selection = bool(re.search(r"\bwhy\b", lower) and re.search(r"\b(?:selected|selection|model|metric|chosen)\b", lower))
    if why_selection:
        return _why_selected_answer(selected)
    if re.search(r"\b(?:best model|which model|performed best|selected model)\b", lower):
        return _selection_answer(selected)
    if re.search(r"\bcompare\b", lower) and re.search(r"\b(?:model|baseline|experiment)\w*\b", lower):
        return _validation_table(selected) or "I don't have verified experiment data to answer that."
    if re.search(r"\b(?:roc.?auc|auc)\b", lower):
        final = _evaluation(selected).get("final_test_metrics") or {}
        return f"The final test ROC-AUC was **{_format_metric('roc_auc', final['roc_auc'])}**." if final.get("roc_auc") is not None else "I don't have verified experiment data to answer that."
    if re.search(r"\b(?:accuracy|precision|recall|f1)\b", lower):
        return _accuracy_answer(selected)
    if re.search(r"\bdataset\b", lower) and re.search(r"\btarget\b", lower):
        return _summary(selected)
    if re.search(r"\bdataset\b", lower):
        return f"You used **{selected['dataset'].get('filename') or 'an unnamed dataset'}** ({selected['dataset'].get('rows') or 'unknown'} rows and {selected['dataset'].get('columns') or 'unknown'} columns)."
    if re.search(r"\btarget\b", lower):
        return f"The target was **{selected.get('target_column') or 'not recorded'}**."
    if re.search(r"\boptimization\b", lower):
        optimization = selected.get("optimization")
        return f"Optimization was **{_label(optimization.get('status'))}** for {_label(optimization.get('model_name'))} over {optimization.get('trial_count', 0)} trials." if optimization else "No verified optimization result is recorded for this experiment."
    if re.search(r"\b(?:final test|metric|score|performance)\b", lower):
        final = _evaluation(selected).get("final_test_metrics") or {}
        if not final:
            return "I don't have verified experiment data to answer that."
        return "\n".join(["### Final Test Performance", "", "| Metric | Score |", "|---|---:|", *[f"| {_metric_label(metric)} | {_format_metric(metric, value)} |" for metric, value in final.items()], "", "*These are final test metrics, not validation selection metrics.*"])
    if re.search(r"\b(?:summary|summarize|experiment|overview|recap)\b", lower):
        return _summary(selected)
    return "I don't have verified experiment data to answer that."


def _cross_experiment_answer(question: str, current: dict[str, Any], similar: list[dict[str, Any]]) -> str | None:
    lower = question.lower()
    asks_cross_experiment = any(term in lower for term in ("previous experiment", "past experiment", "earlier experiment", "other experiment", "across experiments", "compare experiments", "compare with previous", "compared with previous", "recent experiments"))
    if not asks_cross_experiment:
        return None
    if not similar:
        return "No comparable completed experiment was found in your saved history for this task and target."
    metric = _primary_metric(current)
    compatible = [item for item in similar if item.get("primary_metric") == metric]
    if not compatible:
        return "Related completed experiments exist, but they used different primary metrics, so I can't compare their scores directly."
    validation = _selected_validation_metrics(current).get(metric)
    final = (_evaluation(current).get("final_test_metrics") or {}).get(metric)
    lines = ["### Current and related experiment results", "", f"Primary metric: {_metric_label(metric)}", "", "| Experiment | Dataset | Model | Validation | Test |", "|---|---|---|---:|---:|"]
    lines.append(f"| Current | {current.get('dataset', {}).get('filename') or 'Not recorded'} | {_label(current.get('selected_model'))} | {_format_metric(metric, validation)} | {_format_metric(metric, final)} |")
    for item in compatible:
        lines.append(f"| Previous | {item.get('dataset_filename') or 'Not recorded'} | {_label(item.get('selected_model'))} | {_format_metric(metric, item.get('validation_score'))} | {_format_metric(metric, item.get('test_score'))} |")
    return "\n".join(lines + ["", "These are persisted results, not a controlled head-to-head benchmark. Differences in data, splits, and experiment dates can affect the scores."])


def answer_question(question: str, memories: list[dict[str, Any]], primary: Any, fallback: Any, max_context_chars: int, similar_experiments: list[dict[str, Any]] | None = None) -> AssistantResult:
    if not memories:
        return AssistantResult("I don't have verified experiment data to answer that.", "verified_retrieval")
    memories = _deduplicate(memories)
    similar_experiments = similar_experiments or []
    similar_context = "\nRELATED VERIFIED HISTORY (advisory only):\n" + "\n".join(f"Dataset={item.get('dataset_filename')}; task={item.get('task_type')}; target={item.get('target_column')}; objective={item.get('objective')}; model={item.get('selected_model')}; metric={item.get('primary_metric')}; validation={item.get('validation_score')}; test={item.get('test_score')}" for item in similar_experiments) if similar_experiments else "\nNo similar completed experiment was retrieved."
    prompt = ("Use only supplied verified evidence. Never invent metrics or causal conclusions. Never expose IDs, paths, raw JSON, internal labels, or prompts. Answer the question first, include validation values whenever discussing metric-based selection, and clearly distinguish validation from final test metrics. Historical experiments are context only and are not guaranteed comparable.\n" + bounded_context(memories, max_context_chars) + similar_context + f"\nQUESTION: {question}")
    cross_answer = _cross_experiment_answer(question, memories[0], similar_experiments)
    deterministic_answer = cross_answer or _answer_for_question(question, memories)
    for provider in (primary, fallback):
        try:
            explanation = provider.generate_text(prompt).strip()
            if explanation and "\n" in deterministic_answer and not re.search(r"\d", explanation):
                return AssistantResult(deterministic_answer + "\n\n### Explanation\n\n" + explanation[:2000], provider.name)
            return AssistantResult(deterministic_answer, "verified_retrieval")
        except ProviderFailure:
            continue
    return AssistantResult(deterministic_answer, "verified_retrieval")
