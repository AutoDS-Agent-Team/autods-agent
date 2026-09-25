"""Natural language to bounded analytics plans; this module never executes generated code."""
import json
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.llm.base import LLMProvider, ProviderFailure
from app.schemas.auto_analysis import FeatureRole
from app.schemas.analytics import Aggregation, AnalyticsFilter, AnalyticsPlanStep, DatasetAnalyticsPlan, FilterOperator, PlanStepOperation
from app.services.auto_analysis_service import detect_feature_roles


UNSUPPORTED_QUESTION = "I can't answer that reliably from the available dataset with the currently supported analytical operations."
_RATE_WORDS = ("rate", "percent", "percentage", "proportion", "share")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_NUMERIC_AGGREGATIONS = {Aggregation.MEAN, Aggregation.MEDIAN, Aggregation.SUM, Aggregation.MIN, Aggregation.MAX, Aggregation.STD, Aggregation.VARIANCE, Aggregation.QUANTILE}


def _normalize(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _stems(value: str) -> set[str]:
    tokens = _normalize(value).split()
    stems = set(tokens)
    for token in tokens:
        if len(token) > 4:
            stems.add(token.rstrip("s"))
            stems.add(re.sub(r"(?:ing|ed|al|or|ion)$", "", token))
    return {item for item in stems if len(item) >= 3}


def _column(frame: pd.DataFrame, phrase: str) -> str | None:
    normalized, words = _normalize(phrase), set(_stems(phrase))
    choices = {str(name): _normalize(str(name)) for name in frame.columns}
    exact = next((name for name, label in choices.items() if label == normalized), None)
    if exact:
        return exact
    compact = normalized.replace(" ", "")
    compact_exact = next((name for name, label in choices.items() if label.replace(" ", "") == compact), None)
    if compact_exact:
        return compact_exact
    candidates = []
    for name, label in choices.items():
        label_words = set(_stems(label))
        overlap = len(words & label_words)
        # Suffix matching supports compact generic identifiers such as Pclass
        # without maintaining a dataset-specific alias list.
        suffix = any(len(word) >= 4 and any(token.endswith(word) for token in label_words) for word in words)
        if overlap or suffix:
            candidates.append((overlap + int(suffix), len(label), name))
    return max(candidates)[2] if candidates else None


def _mentioned(frame: pd.DataFrame, question: str) -> list[str]:
    question_words = set(_stems(question))
    found = []
    for name in frame.columns:
        label_words = set(_stems(str(name)))
        if label_words & question_words or any(len(word) >= 4 and any(token.endswith(word) for token in label_words) for word in question_words) or any(len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5] for left in label_words for right in question_words):
            found.append(str(name))
            continue
        # Category values can identify their containing column (for example a
        # comparison between two displayed categories) without any fixed names.
        series = frame[name].dropna()
        if not pd.api.types.is_numeric_dtype(series) and series.nunique() <= 50:
            for value in series.unique():
                value_words = set(_stems(str(value)))
                if value_words and value_words <= question_words:
                    found.append(str(name))
                    break
    return list(dict.fromkeys(found))


def _positive_value(series: pd.Series, question: str = "") -> Any | None:
    values = list(series.dropna().unique())
    if len(values) != 2:
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()) == {0, 1}:
        value = next(value for value in values if float(value) == 1.0)
        return value.item() if hasattr(value, "item") else value
    if numeric.notna().all():
        return None
    if pd.api.types.is_bool_dtype(series):
        return True
    normalized = {_normalize(str(value)): value for value in values}
    question_text = _normalize(question)
    mentioned = [value for label, value in normalized.items() if label and re.search(rf"\b{re.escape(label)}\b", question_text)]
    if len(mentioned) == 1:
        return mentioned[0]
    # Generic conventional binary encodings only; unknown labels remain ambiguous.
    for label in ("yes", "true", "1"):
        if label in normalized:
            return normalized[label]
    return None


def _numeric_filters(question: str, frame: pd.DataFrame) -> list[AnalyticsFilter]:
    text = _normalize(question)
    operator_pattern = r"(greater than|more than|above|over|older than|at least|>=|>|less than|below|under|at most|<=|<)"
    filters: list[AnalyticsFilter] = []
    for column in frame.select_dtypes(include="number").columns:
        labels = {_normalize(str(column)), *[item.rstrip("s") for item in _stems(str(column))]}
        match = next((
            found
            for label in labels if label
            for pattern in (
                rf"\b{re.escape(label)}\b\s*(?:is\s+)?{operator_pattern}\s*(\d+(?:\.\d+)?)\b",
                rf"{operator_pattern}\s*(\d+(?:\.\d+)?)\s+\b{re.escape(label)}\b",
            )
            if (found := re.search(pattern, text))
        ), None)
        if not match:
            continue
        operator, raw = match.groups()
        comparison = FilterOperator.GT if operator in {"greater than", "more than", "above", "over", "older than", ">"} else FilterOperator.GTE if operator in {"at least", ">="} else FilterOperator.LT if operator in {"less than", "below", "under", "<"} else FilterOperator.LTE
        filters.append(AnalyticsFilter(column=str(column), operator=comparison, value=float(raw)))
    return filters[:5]


def _ordinal_filters(question: str, frame: pd.DataFrame) -> list[AnalyticsFilter]:
    text = _normalize(question)
    tokens = text.split()
    filters = []
    for ordinal, value in _ORDINALS.items():
        if ordinal not in tokens:
            continue
        position = tokens.index(ordinal)
        candidate = _column(frame, " ".join(tokens[position + 1:position + 4]))
        if candidate and pd.api.types.is_numeric_dtype(frame[candidate]) and value in set(pd.to_numeric(frame[candidate], errors="coerce").dropna().unique()):
            filters.append(AnalyticsFilter(column=candidate, operator=FilterOperator.EQ, value=float(value)))
    return filters


def _categorical_filters(question: str, frame: pd.DataFrame, excluded: set[str]) -> list[AnalyticsFilter]:
    words = set(_stems(question))
    filters = []
    for column in frame.columns:
        if str(column) in excluded or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna().unique()
        if len(values) > 50:
            continue
        matched = [value for value in values if (value_words := set(_stems(str(value)))) and value_words <= words]
        if len(matched) == 1:
            filters.append(AnalyticsFilter(column=str(column), operator=FilterOperator.EQ, value=matched[0]))
    return filters


def _context(frame: pd.DataFrame) -> dict[str, Any]:
    roles = _role_map(frame)
    return {"rows": len(frame), "columns": [{"name": str(name), "dtype": str(frame[name].dtype), "role": roles[str(name)].value, "non_null": int(frame[name].notna().sum()), "unique": int(frame[name].nunique(dropna=True)), "examples": [str(value)[:80] for value in frame[name].dropna().unique()[:5]]} for name in frame.columns]}


def _explicit_groups(question: str, frame: pd.DataFrame) -> list[str]:
    lower = question.lower()
    phrases = re.findall(r"\b(?:grouped?\s+by|by|across|between|versus|vs)\s+([^?.!,]+)", lower)
    phrases += re.findall(r"\bwhich\s+([^?.!,]+?)\s+has\b", lower)
    groups = []
    for phrase in phrases:
        parts = re.split(r"\s+(?:and|/)\s+|,\s*", phrase.strip())
        for part in parts:
            normalized_phrase = _normalize(part.strip())
            candidate = next(
                (str(name) for name in frame.columns
                 if normalized_phrase == _normalize(str(name))
                 or normalized_phrase.startswith(f"{_normalize(str(name))} ")),
                None,
            )
            if candidate and candidate not in groups:
                groups.append(candidate)
    return groups


def _include_missing_requested(question: str) -> bool:
    return bool(re.search(r"\b(?:include|including|analyze|analyse|show|compare)\b[^?.!,]{0,40}\bmissing\b", question.lower()))


def _requested_aggregations(question: str) -> list[Aggregation]:
    aliases = {
        Aggregation.MEAN: ("average", "mean"),
        Aggregation.MEDIAN: ("median",),
        Aggregation.SUM: ("sum", "total"),
        Aggregation.COUNT: ("count", "how many"),
    }
    lower = question.lower()
    matches = [(min(re.search(rf"\b{re.escape(word)}\b", lower).start() for word in words if re.search(rf"\b{re.escape(word)}\b", lower)), aggregation) for aggregation, words in aliases.items() if any(re.search(rf"\b{re.escape(word)}\b", lower) for word in words)]
    return [aggregation for _, aggregation in sorted(matches)]


def _role_map(frame: pd.DataFrame) -> dict[str, FeatureRole]:
    return {item.column: item.role for item in detect_feature_roles(frame)}


def _is_binary_column(frame: pd.DataFrame, column: str, roles: dict[str, FeatureRole]) -> bool:
    if roles.get(column) in {FeatureRole.BOOLEAN, FeatureRole.CATEGORICAL}:
        return True
    if roles.get(column) != FeatureRole.NUMERICAL:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    return len(values) == 2 and set(values) == {0, 1}


def _semantic_validate_plan(question: str, frame: pd.DataFrame, plan: DatasetAnalyticsPlan) -> DatasetAnalyticsPlan | None:
    """Repair unambiguous role mismatches and reject incompatible operations."""
    requested_aggregations = _requested_aggregations(question)
    roles = _role_map(frame)
    explicit_groups = _explicit_groups(question, frame)
    include_missing = _include_missing_requested(question)
    rate_requested = bool(re.search(r"\b(?:rate|rates|percent|percentage|proportion|share)\b", question.lower()))
    payload = plan.model_dump()
    for step in payload["steps"]:
        names = [step.get("column"), *step.get("group_by", []), *(item["column"] for item in step.get("filters", []))]
        if any(name and name not in frame.columns for name in names):
            return None
        if step["operation"] == PlanStepOperation.GROUP_AGGREGATE:
            if explicit_groups:
                step["group_by"] = explicit_groups
            step["include_missing"] = include_missing
            if any(roles.get(name) == FeatureRole.IDENTIFIER and name not in explicit_groups for name in step.get("group_by", [])):
                return None
        if requested_aggregations and step["operation"] in {PlanStepOperation.AGGREGATE, PlanStepOperation.GROUP_AGGREGATE}:
            step["aggregation"] = requested_aggregations[0]
            step["aggregations"] = requested_aggregations if len(requested_aggregations) > 1 else []
        aggregation = step.get("aggregation")
        column = step.get("column")
        if aggregation in {item.value for item in _NUMERIC_AGGREGATIONS} and not (
            pd.api.types.is_numeric_dtype(frame[column]) or (aggregation == Aggregation.MEAN.value and _is_binary_column(frame, column, roles))
        ):
            return None
        if step.get("positive_value") is not None:
            if not _is_binary_column(frame, column, roles) or not rate_requested:
                return None
        if rate_requested and step["operation"] in {PlanStepOperation.AGGREGATE, PlanStepOperation.GROUP_AGGREGATE}:
            if not _is_binary_column(frame, column, roles):
                return None
    try:
        return DatasetAnalyticsPlan.model_validate(payload)
    except ValidationError:
        return None


def _preserve_group_comparisons(question: str, plan: DatasetAnalyticsPlan) -> DatasetAnalyticsPlan:
    lower = question.lower()
    compare = bool(re.search(r"\b(?:compare[ds]?|comparing|comparisons?|versus|vs|across)\b", lower))
    highest = bool(re.search(r"\b(?:highest|top|most)\b", lower))
    lowest = bool(re.search(r"\b(?:lowest|bottom|least)\b", lower))
    grouped_index = next((index for index, step in enumerate(plan.steps) if step.operation == PlanStepOperation.GROUP_AGGREGATE), None)
    if grouped_index is None or not (compare or highest or lowest):
        return plan
    requested = re.search(r"\b(?:top|bottom)\s+(\d{1,3})\b", lower)
    limit = 100 if compare or not requested else min(100, int(requested.group(1)))
    direction = "ascending" if lowest and not highest and not compare else "descending"
    payload = plan.model_dump()
    payload["steps"][grouped_index].update(sort=direction, limit=100 if compare else payload["steps"][grouped_index]["limit"])
    if compare:
        payload["steps"][grouped_index + 1:] = []
    elif len(payload["steps"]) > grouped_index + 1:
        payload["steps"][grouped_index + 1:] = [{"operation": "sort_limit", "sort": direction, "limit": limit}]
    elif len(payload["steps"]) < 6:
        payload["steps"].append({"operation": "sort_limit", "sort": direction, "limit": limit})
    else:
        payload["steps"][grouped_index]["limit"] = limit
    return DatasetAnalyticsPlan.model_validate(payload)


def _fallback(question: str, frame: pd.DataFrame) -> DatasetAnalyticsPlan | None:
    lower, mentioned = question.lower(), _mentioned(frame, question)
    requested_aggregations = _requested_aggregations(question)
    numeric = [name for name in mentioned if pd.api.types.is_numeric_dtype(frame[name])]
    if "associated" in lower or "association" in lower:
        target = next(iter(numeric), None)
        return DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.ASSOCIATION, column=target, limit=10)]) if target else None
    if "correlat" in lower:
        return DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.CORRELATION, correlation_method="spearman" if "spearman" in lower else "pearson", limit=10)])
    if "outlier" in lower and numeric:
        return DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.OUTLIER_SUMMARY, column=numeric[0])])
    if "distribution" in lower and mentioned:
        return DatasetAnalyticsPlan(steps=[AnalyticsPlanStep(operation=PlanStepOperation.DISTRIBUTION, column=mentioned[0])])

    rate_requested = bool(re.search(r"\b(?:rate|rates|percent|percentage|proportion|share)\b", lower))
    count_requested = "how many" in lower or bool(re.search(r"\bcount\b", lower))
    binary = [name for name in frame.columns if _positive_value(frame[name], question) is not None]
    comparison_requested = bool(re.search(r"\b(?:compare|between|versus|vs|across|highest|lowest|top|bottom)\b", lower))
    measure_match = re.search(r"(?:average|mean|median|total|sum)\s+(?:of\s+)?([a-z0-9 _-]+?)(?:\s+(?:of|by|among|for)|\?|$)", lower)
    explicit_measure = _column(frame, measure_match.group(1)) if measure_match else None
    target = explicit_measure or (next((name for name in binary if name not in _explicit_groups(question, frame)), None) if binary and (rate_requested or count_requested or comparison_requested) else None)
    aggregate_words = {"average": Aggregation.MEAN, "mean": Aggregation.MEAN, "median": Aggregation.MEDIAN, "total": Aggregation.SUM, "sum": Aggregation.SUM, "highest": Aggregation.MEAN, "lowest": Aggregation.MEAN}
    aggregation = next((value for word, value in aggregate_words.items() if re.search(rf"\b{word}\b", lower)), None)
    if requested_aggregations:
        aggregation = requested_aggregations[0]
    if target is None:
        target = next((name for name in mentioned if name in numeric), None)
    if target is None and count_requested and len(frame.columns):
        target = str(frame.columns[0])
    if target is None:
        return None
    explicit_groups = _explicit_groups(question, frame)
    if count_requested and target in explicit_groups:
        target = next((str(name) for name in frame.columns if str(name) not in explicit_groups), target)
    positive = _positive_value(frame[target], question) if rate_requested or count_requested else None
    if rate_requested and positive is None:
        return None
    if count_requested and positive is not None and not any(word in lower for word in _RATE_WORDS):
        return DatasetAnalyticsPlan(steps=[
            AnalyticsPlanStep(operation=PlanStepOperation.FILTER, filters=[AnalyticsFilter(column=target, operator=FilterOperator.EQ, value=positive)]),
            AnalyticsPlanStep(operation=PlanStepOperation.AGGREGATE, column=target, aggregation=Aggregation.COUNT),
        ])
    aggregation = Aggregation.COUNT if count_requested and positive is None else Aggregation.MEAN if rate_requested else aggregation
    if aggregation is None:
        # A comparison of a verified binary target is a group proportion even
        # when the user says "survival" or another domain-neutral outcome.
        if target in binary or (target in mentioned and set(pd.to_numeric(frame[target], errors="coerce").dropna().unique()) == {0, 1}):
            aggregation = Aggregation.MEAN
        else:
            return None
    filters = _numeric_filters(question, frame) + _ordinal_filters(question, frame)
    # When a question asks for a measure "of" a semantic binary outcome
    # (e.g. average age of survivors), constrain that outcome without relying
    # on dataset-specific aliases.
    if explicit_measure:
        q_stems = _stems(question)
        for candidate in frame.columns:
            candidate_positive = _positive_value(frame[candidate], question)
            if candidate_positive is None or str(candidate) == explicit_measure:
                continue
            candidate_stems = _stems(str(candidate))
            if any(len(a) >= 5 and len(b) >= 5 and (a.startswith(b[:5]) or b.startswith(a[:5])) for a in candidate_stems for b in q_stems):
                filters.append(AnalyticsFilter(column=str(candidate), operator=FilterOperator.EQ, value=candidate_positive))
                break
    # "older than" is a semantic numeric constraint; infer the dataset's
    # age-like column only when that column is uniquely present.
    if "older than" in lower and not any(item.operator in {FilterOperator.GT, FilterOperator.GTE} for item in filters):
        age_columns = [str(name) for name in frame.columns if "age" in _stems(str(name)) and pd.api.types.is_numeric_dtype(frame[name])]
        age_match = re.search(r"older\s+than\s+(\d+(?:\.\d+)?)", lower)
        if len(age_columns) == 1 and age_match:
            filters.append(AnalyticsFilter(column=age_columns[0], operator=FilterOperator.GT, value=float(age_match.group(1))))
    filter_columns = {item.column for item in filters}
    explicit_value_columns = {
        str(column) for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        and sum(1 for value in frame[column].dropna().unique() if set(_stems(str(value))) and set(_stems(str(value))) <= set(_stems(question))) == 1
    }
    role_map = _role_map(frame)
    group_candidates = [
        name for name in mentioned
        if name != target and name not in filter_columns
        and name not in explicit_value_columns
        and role_map.get(name) != FeatureRole.IDENTIFIER
        and (role_map.get(name) in {FeatureRole.CATEGORICAL, FeatureRole.BOOLEAN} or frame[name].nunique(dropna=True) <= 20)
    ]
    grouping_requested = bool(re.search(r"\b(?:by|across|between|compare|versus|vs|highest|lowest|top|bottom)\b", lower))
    group = next((name for name in explicit_groups if name in frame.columns and name != target and name not in filter_columns), None)
    group = group or (group_candidates[0] if grouping_requested and group_candidates else None)
    filters.extend(_categorical_filters(question, frame, {target, group} if group else {target}))
    # Preserve the order in which constraints appear in the user's question;
    # this keeps plans readable while remaining fully schema-driven.
    question_text = _normalize(question)
    def _filter_position(item: AnalyticsFilter) -> int:
        positions = []
        label = _normalize(str(item.column))
        if label in question_text:
            positions.append(question_text.find(label))
        series = frame[item.column]
        if not pd.api.types.is_numeric_dtype(series):
            for value in series.dropna().unique():
                token = _normalize(str(value))
                if token and re.search(rf"\b{re.escape(token)}\b", question_text):
                    positions.append(question_text.find(token))
        return min(positions) if positions else len(question_text)
    filters.sort(key=_filter_position)
    if group:
        steps = ([AnalyticsPlanStep(operation=PlanStepOperation.FILTER, filters=filters)] if filters else [])
        steps.append(AnalyticsPlanStep(operation=PlanStepOperation.GROUP_AGGREGATE, column=target, group_by=[group], aggregation=aggregation, aggregations=requested_aggregations if len(requested_aggregations) > 1 else [], positive_value=positive, include_missing=_include_missing_requested(question)))
        return DatasetAnalyticsPlan(steps=steps)
    steps = ([AnalyticsPlanStep(operation=PlanStepOperation.FILTER, filters=filters)] if filters else [])
    steps.append(AnalyticsPlanStep(operation=PlanStepOperation.AGGREGATE, column=target, aggregation=aggregation, aggregations=requested_aggregations if len(requested_aggregations) > 1 else [], positive_value=positive))
    return DatasetAnalyticsPlan(steps=steps)


def plan_open_ended_dataset_question(question: str, frame: pd.DataFrame, providers: list[LLMProvider]) -> DatasetAnalyticsPlan | None:
    try:
        local = _fallback(question, frame)
    except (ValidationError, ValueError):
        local = None
    if local is not None:
        return _semantic_validate_plan(question, frame, _preserve_group_comparisons(question, local))
    schema = DatasetAnalyticsPlan.model_json_schema()
    prompt = (
        "Translate the dataset question into JSON matching this schema. Return a plan only, never an answer or code. "
        "Use only exact listed columns and allowlisted operations. Never use SQL, Python, expressions, imports, paths, or invented columns. "
        "Exclude missing grouping keys by default; set include_missing true only when the question explicitly asks to include or analyze missing values. "
        "For grouped comparisons, retain all groups up to 100 sorted by the measure; choose a top group only when no comparison is requested. "
        "If it cannot be represented, return an empty steps list.\n"
        f"Dataset metadata: {json.dumps(_context(frame), ensure_ascii=False)}\nQuestion: {question}\nSchema: {json.dumps(schema)}"
    )
    known = {str(name) for name in frame.columns}
    for provider in providers:
        try:
            raw = provider.generate_structured(prompt, schema)
            candidate = DatasetAnalyticsPlan.model_validate_json(raw) if isinstance(raw, str) else DatasetAnalyticsPlan.model_validate(raw)
            if not all(name in known for step in candidate.steps for name in [step.column, *step.group_by, *(item.column for item in step.filters)] if name):
                continue
            if not all(step.positive_value is None or _positive_value(frame[step.column], question) == step.positive_value for step in candidate.steps):
                continue
            validated = _semantic_validate_plan(question, frame, _preserve_group_comparisons(question, candidate))
            if validated is not None:
                return validated
        except (ProviderFailure, ValidationError, ValueError):
            continue
    return None
