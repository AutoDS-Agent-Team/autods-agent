from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from uuid import uuid4
import math
import joblib
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Circle, Drawing, Line, String
from reportlab.graphics.charts.barcharts import HorizontalBarChart

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import TrainingNotReadyError
from app.models.evaluation_result import EvaluationResult
from app.models.experiment import Experiment
from app.models.model_run import ModelRun
from app.models.optimization_result import OptimizationResult
from app.models.prediction_run import PredictionRun
from app.models.report import Report
from app.schemas.evaluation import ReportResponse
from app.services.artifact_service import confined_artifact_path
from app.services.evaluation_service import get_evaluation
from app.services.explainability_service import get_explainability
from app.services.training_service import _load_training_definition
from app.services.dataset_service import get_dataset_metadata
from app.services.dataset_service import load_dataset_dataframe
from app.services.profiling_service import profile_dataframe
from app.services.evaluation_service import load_model_artifact, model_artifact_path
from app.services.training_service import _prepare_features


def _pdf_metric(value: float | None, name: str) -> str:
    return _metric(value, name)


def _pdf_bar_chart(title: str, labels: list[str], values: list[float], color: colors.Color = colors.HexColor("#1687d4")) -> Drawing | None:
    if not labels or not values:
        return None
    labels, values = labels[:8], values[:8]
    drawing = Drawing(480, max(150, len(labels) * 26 + 54))
    drawing.add(String(0, drawing.height - 16, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#075c9e")))
    chart = HorizontalBarChart()
    chart.x, chart.y = 145, 12
    chart.width, chart.height = 300, drawing.height - 45
    chart.data = [values]
    chart.categoryAxis.categoryNames = [label[:24] for label in labels]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = color
    chart.bars[0].strokeColor = color
    drawing.add(chart)
    return drawing


def _pdf_scatter_chart(title: str, x_values: list[float], y_values: list[float], x_label: str, y_label: str, reference: bool = False) -> Drawing | None:
    if not x_values or not y_values:
        return None
    if reference:
        low, high = min(x_values + y_values), max(x_values + y_values)
        x_low, x_spread, y_low, y_spread = low, high - low or 1.0, low, high - low or 1.0
    else:
        x_low, x_high = min(x_values), max(x_values)
        y_low, y_high = min(y_values), max(y_values)
        x_spread, y_spread = x_high - x_low or 1.0, y_high - y_low or 1.0
    drawing = Drawing(480, 260)
    drawing.add(String(0, 244, title, fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor("#075c9e")))
    left, bottom, width, height = 55, 35, 385, 175
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=colors.HexColor("#405772")))
    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=colors.HexColor("#405772")))
    if reference:
        drawing.add(Line(left, bottom, left + width, bottom + height, strokeColor=colors.HexColor("#99aec2"), strokeDashArray=[4, 3]))
    for x_value, y_value in zip(x_values[:250], y_values[:250], strict=True):
        drawing.add(Circle(left + width * (x_value - x_low) / x_spread, bottom + height * (y_value - y_low) / y_spread, 1.7, fillColor=colors.HexColor("#1687d4"), strokeColor=None))
    drawing.add(String(left, 17, x_label, fontSize=8, fillColor=colors.HexColor("#536b84")))
    drawing.add(String(2, 218, y_label, fontSize=8, fillColor=colors.HexColor("#536b84")))
    return drawing


def _histogram_counts(values: list[float], bins: int = 8) -> tuple[list[str], list[float]]:
    if not values:
        return [], []
    low, high = min(values), max(values)
    spread = high - low or 1.0
    counts = [0.0] * bins
    for value in values:
        counts[min(bins - 1, int((value - low) / spread * bins))] += 1
    labels = [_number(low + index * spread / bins) for index in range(bins)]
    return labels, counts


def _pdf_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e5ef")); canvas.line(48, 36, letter[0] - 48, 36)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#536b84"))
    canvas.drawString(48, 23, "AutoDS-Agent - Verified analytics report")
    canvas.drawRightString(letter[0] - 48, 23, f"Page {doc.page}")
    canvas.restoreState()


_PERCENT_METRICS = frozenset({"accuracy", "precision", "recall", "f1", "roc_auc", "r2"})
_METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "f1": "F1", "roc_auc": "ROC-AUC"}


def _metric(value: float | None, name: str) -> str:
    """The sole report formatter for every verified model metric."""
    if value is None:
        return "Not applicable"
    if name in _PERCENT_METRICS:
        return f"{value * 100:.1f}%"
    return f"{value:,.3f}"


def _metric_rows(values: dict[str, float | None]) -> str:
    return "".join(f"<tr><th>{_METRIC_LABELS.get(name, name.replace('_', ' ').title())}</th><td>{_metric(value, name)}</td></tr>" for name, value in values.items())


def _number(value: float | None, decimals: int = 0) -> str:
    if value is None:
        return "Not applicable"
    return f"{value:,.{decimals}f}"


def _metric_label(name: str) -> str:
    return _METRIC_LABELS.get(name, name.replace("_", " ").title())


def _bar_chart(title: str, labels: list[str], values: list[float], *, color: str = "#1687d4") -> str:
    """Small self-contained SVG built only from verified values."""
    if not labels or not values:
        return ""
    labels, values = labels[:10], values[:10]
    maximum = max(values) or 1
    width, row_height = 680, 32
    rows = "".join(
        f"<text x='4' y='{32 + index * row_height}' class='label'>{escape(label[:28])}</text>"
        f"<rect x='180' y='{12 + index * row_height}' width='{max(2, 440 * value / maximum):.1f}' height='18' rx='4' fill='{color}'/>"
        f"<text x='628' y='{27 + index * row_height}' class='value'>{_number(value)}</text>"
        for index, (label, value) in enumerate(zip(labels, values, strict=True))
    )
    return f"<figure><figcaption>{escape(title)}</figcaption><svg viewBox='0 0 {width} {50 + len(labels) * row_height}' role='img' aria-label='{escape(title)}'>{rows}</svg></figure>"


def _scatter_chart(actual: list[float], predicted: list[float]) -> str:
    if not actual or not predicted:
        return ""
    combined = actual + predicted
    low, high = min(combined), max(combined)
    spread = high - low or 1
    points = "".join(f"<circle cx='{45 + 560 * (x-low)/spread:.1f}' cy='{315 - 260 * (y-low)/spread:.1f}' r='3' fill='#1687d4' fill-opacity='.65'/>" for x, y in zip(actual[:300], predicted[:300], strict=True))
    return f"<figure><figcaption>Actual vs predicted values (held-out test set)</figcaption><svg viewBox='0 0 650 350' role='img' aria-label='Actual versus predicted scatter plot'><line x1='45' y1='315' x2='605' y2='55' stroke='#99aec2' stroke-dasharray='5 4'/><line x1='45' y1='315' x2='605' y2='315' stroke='#405772'/><line x1='45' y1='315' x2='45' y2='35' stroke='#405772'/>{points}<text x='280' y='342' class='label'>Actual</text><text x='8' y='38' class='label'>Predicted</text></svg></figure>"


def _histogram_chart(title: str, values: list[float], x_label: str) -> str:
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    width, bins = 650, 12
    spread = high - low or 1.0
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int((value - low) / spread * bins))] += 1
    maximum = max(counts) or 1
    bars = "".join(
        f"<rect x='{45 + index * 46}' y='{315 - count / maximum * 250:.1f}' width='38' height='{count / maximum * 250:.1f}' fill='#6d5dfc'/>"
        for index, count in enumerate(counts)
    )
    return f"<figure><figcaption>{escape(title)}</figcaption><svg viewBox='0 0 {width} 350' role='img' aria-label='{escape(title)}'><line x1='45' y1='315' x2='605' y2='315' stroke='#405772'/><line x1='45' y1='315' x2='45' y2='35' stroke='#405772'/>{bars}<text x='45' y='337' class='label'>{_number(low)}</text><text x='520' y='337' class='label'>{_number(high)}</text><text x='275' y='347' class='label'>{escape(x_label)}</text><text x='8' y='38' class='label'>Count</text></svg></figure>"


def _residual_scatter_chart(predicted: list[float], residuals: list[float]) -> str:
    if not predicted or not residuals:
        return ""
    low, high = min(predicted), max(predicted)
    residual_limit = max(abs(min(residuals)), abs(max(residuals))) or 1.0
    spread = high - low or 1.0
    points = "".join(
        f"<circle cx='{45 + 560 * (prediction-low)/spread:.1f}' cy='{175 - 130 * residual/residual_limit:.1f}' r='3' fill='#d16a9d' fill-opacity='.65'/>"
        for prediction, residual in zip(predicted[:300], residuals[:300], strict=True)
    )
    return f"<figure><figcaption>Residuals vs predicted values (held-out test set)</figcaption><svg viewBox='0 0 650 350' role='img' aria-label='Residuals versus predicted values'><line x1='45' y1='175' x2='605' y2='175' stroke='#99aec2' stroke-dasharray='5 4'/><line x1='45' y1='315' x2='605' y2='315' stroke='#405772'/><line x1='45' y1='315' x2='45' y2='35' stroke='#405772'/>{points}<text x='250' y='342' class='label'>Predicted value</text><text x='8' y='38' class='label'>Residual</text></svg></figure>"


def _regression_arrays(selected: ModelRun, dataframe: pd.DataFrame, target: str, settings: Settings) -> tuple[list[float], list[float], list[float]]:
    """Recreate predictions only from the selected trusted artifact and its saved test indices."""
    raw = joblib.load(model_artifact_path(settings, selected.artifact_filename or ""))
    artifact = load_model_artifact(selected, settings)
    features, target_values, *_ = _prepare_features(dataframe, target)
    indexes = raw.get("test_indices", [])
    x_test = features.loc[indexes, artifact.feature_columns]
    actual = pd.to_numeric(target_values.loc[indexes], errors="raise").astype(float)
    predicted = artifact.model.predict(artifact.preprocessor.transform(x_test))
    residuals = actual.to_numpy() - predicted
    return actual.tolist(), [float(value) for value in predicted], residuals.tolist()


def _regression_diagnostics(selected: ModelRun, dataframe: pd.DataFrame, target: str, settings: Settings) -> tuple[str, str]:
    try:
        actual_values, predicted_values, residual_values = _regression_arrays(selected, dataframe, target, settings)
        residuals = pd.Series(residual_values)
        charts = "".join((
            _scatter_chart(actual_values, predicted_values),
            _histogram_chart("Residual distribution (held-out test set)", residuals.tolist(), "Residual in target units"),
            _residual_scatter_chart(predicted_values, residuals.tolist()),
        ))
        error = pd.Series(abs(residuals))
        summary = (
            "<table><tr><th>Verified diagnostic</th><th>Value (target units)</th></tr>"
            f"<tr><td>Mean residual</td><td>{_number(float(residuals.mean()), 3)}</td></tr>"
            f"<tr><td>Residual standard deviation</td><td>{_number(float(residuals.std()), 3)}</td></tr>"
            f"<tr><td>Median absolute error</td><td>{_number(float(error.quantile(.5)), 3)}</td></tr>"
            f"<tr><td>90th percentile absolute error</td><td>{_number(float(error.quantile(.9)), 3)}</td></tr>"
            f"<tr><td>Largest absolute error</td><td>{_number(float(error.max()), 3)}</td></tr></table>"
        )
        return charts, summary
    except Exception:
        return "", "<p class='note'>Prediction diagnostics are unavailable because the trusted test artifact could not be read.</p>"


def _prediction_summaries(experiment_id: str, database: Session, settings: Settings, task_type: str) -> list[dict[str, object]]:
    """Load only persisted prediction outputs for the experiment, never regenerate or infer them."""
    records = (
        database.query(PredictionRun)
        .filter_by(experiment_id=experiment_id)
        .order_by(PredictionRun.created_at.desc())
        .limit(5)
        .all()
    )
    summaries: list[dict[str, object]] = []
    for record in records:
        summary: dict[str, object] = {
            "created_at": record.created_at.isoformat(),
            "row_count": record.row_count,
            "columns": [],
            "preview": [],
            "insight": "Saved prediction output could not be read.",
            "chart_title": "",
            "chart_labels": [],
            "chart_values": [],
            "confidence_labels": [],
            "confidence_values": [],
            "statistics": [],
        }
        try:
            path = confined_artifact_path(settings.prediction_storage_path, record.artifact_filename, ".csv")
            output = pd.read_csv(path)
            columns = list(output.columns[:4])
            preview = []
            for _, row in output.loc[:, columns].head(20).iterrows():
                rendered_row: list[str] = []
                for column in columns:
                    value = row[column]
                    if pd.isna(value):
                        rendered_row.append("-")
                    elif column == "confidence":
                        rendered_row.append(f"{float(value):.4f}")
                    elif isinstance(value, float):
                        rendered_row.append(f"{value:.6g}")
                    else:
                        rendered_row.append(str(value))
                preview.append(rendered_row)
            summary["columns"] = columns
            summary["preview"] = preview
            if "prediction" in output.columns:
                prediction = output["prediction"]
                numeric_prediction = pd.to_numeric(prediction, errors="coerce")
                if "classification" in task_type:
                    counts = prediction.astype("string").fillna("Missing").value_counts().head(8)
                    summary["chart_title"] = "Predicted class distribution (supplied rows)"
                    summary["chart_labels"] = [str(label) for label in counts.index]
                    summary["chart_values"] = [float(count) for count in counts.values]
                    total = int(counts.sum())
                    summary["insight"] = "Predicted class counts: " + ", ".join(
                        f"{label} ({count:,}; {count / total * 100:.1f}%)" for label, count in counts.items()
                    ) + "." if total else "No non-missing predictions were found."
                elif numeric_prediction.notna().all():
                    numeric_prediction = numeric_prediction.astype(float)
                    summary["insight"] = (
                        f"Predicted values for supplied rows range from {numeric_prediction.min():,.3f} to {numeric_prediction.max():,.3f}; "
                        f"mean: {numeric_prediction.mean():,.3f}; median: {numeric_prediction.median():,.3f}."
                    )
                    labels, counts = _histogram_counts(numeric_prediction.tolist(), bins=8)
                    summary["chart_title"] = "Predicted value distribution (supplied rows)"
                    summary["chart_labels"] = labels
                    summary["chart_values"] = counts
                    summary["statistics"] = [
                        ("Mean prediction", _number(float(numeric_prediction.mean()), 3)),
                        ("Median prediction", _number(float(numeric_prediction.median()), 3)),
                        ("10th percentile", _number(float(numeric_prediction.quantile(.1)), 3)),
                        ("90th percentile", _number(float(numeric_prediction.quantile(.9)), 3)),
                        ("Prediction range", f"{_number(float(numeric_prediction.min()), 3)} – {_number(float(numeric_prediction.max()), 3)}"),
                    ]
                else:
                    counts = prediction.astype(str).value_counts().head(3)
                    summary["insight"] = "Top predicted classes: " + ", ".join(
                        f"{label} ({count:,})" for label, count in counts.items()
                    ) + "."
            if "confidence" in output.columns:
                confidence = pd.to_numeric(output["confidence"], errors="coerce").dropna()
                if not confidence.empty:
                    confidence_bins = pd.cut(confidence.clip(0, 1), bins=[0, .2, .4, .6, .8, 1], include_lowest=True).value_counts(sort=False)
                    summary["confidence_labels"] = [str(interval).replace("(", "[") for interval in confidence_bins.index]
                    summary["confidence_values"] = [float(count) for count in confidence_bins.values]
                    summary["insight"] = str(summary["insight"]).rstrip(".") + (
                        f". Mean model-reported confidence: {confidence.mean() * 100:.1f}% across {len(confidence):,} rows."
                    )
        except (OSError, ValueError, pd.errors.ParserError):
            # A historical report remains useful even if a separately stored prediction file was removed.
            pass
        summaries.append(summary)
    return summaries


def _prediction_html(summaries: list[dict[str, object]]) -> str:
    if not summaries:
        return "<p class='note'>No prediction runs have been created for this experiment yet. Create predictions from a CSV containing the required feature columns to add a verified outlook here. Reports never invent future values.</p>"
    blocks: list[str] = [
        "<p class='note'>Outlook is calculated from the selected saved model and the feature rows supplied for prediction. It is not an automatic time-based forecast; classification confidence is a model-reported score, not accuracy or a calibrated probability.</p>"
    ]
    for index, item in enumerate(summaries, start=1):
        columns = item["columns"]
        preview = item["preview"]
        preview_table = ""
        if columns and preview:
            headers = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
            rows = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in preview)
            preview_table = f"<table><tr>{headers}</tr>{rows}</table><p class='note'>Showing the first {len(preview):,} saved output rows.</p>"
        prediction_chart = _bar_chart(str(item["chart_title"]), list(item["chart_labels"]), list(item["chart_values"]), color="#1687d4")
        confidence_chart = _bar_chart("Model-reported confidence distribution", list(item["confidence_labels"]), list(item["confidence_values"]), color="#44b78b")
        statistics_table = ""
        if item["statistics"]:
            statistics_table = "<table><tr><th>Prediction summary</th><th>Verified value</th></tr>" + "".join(
                f"<tr><td>{escape(str(label))}</td><td>{escape(str(value))}</td></tr>" for label, value in item["statistics"]
            ) + "</table>"
        blocks.append(
            f"<div class='card'><h3>Prediction run {index}</h3>"
            f"<p><strong>Generated:</strong> {escape(str(item['created_at']))} · <strong>Rows:</strong> {int(item['row_count']):,}</p>"
            f"<p><strong>Verified output summary:</strong> {escape(str(item['insight']))}</p>{statistics_table}{prediction_chart}{confidence_chart}{preview_table}</div>"
        )
    return "".join(blocks)


def create_report(experiment_id: str, database: Session, settings: Settings) -> ReportResponse:
    experiment, _, plan = _load_training_definition(experiment_id, database)
    evaluation = get_evaluation(experiment.id, database)
    selected = database.get(ModelRun, evaluation.selected_model_run_id)
    if selected is None:
        raise TrainingNotReadyError("Selected model is unavailable.")
    optimization = database.query(OptimizationResult).filter_by(experiment_id=experiment.id).one_or_none()
    prediction_summaries = _prediction_summaries(experiment.id, database, settings, plan.task_type.value)
    prediction_section = _prediction_html(prediction_summaries)
    try:
        explanation = get_explainability(experiment.id, database, settings)
        feature_items = explanation.global_feature_importance[:10]
        features = "".join(f"<li>{escape(item.feature_name)}: {item.importance:.6f}</li>" for item in feature_items)
    except TrainingNotReadyError:
        feature_items = []
        features = "<li>Native explanation unavailable.</li>"
    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    profile_summary, column_profiles, correlations = profile_dataframe(dataframe, top_values_limit=8)
    missing = [column for column in column_profiles if column.missing_count]
    quality_rows = "".join(f"<tr><td>{escape(column.name)}</td><td>{column.missing_count:,}</td><td>{column.missing_percentage:.2f}%</td><td>{column.unique_count:,}</td></tr>" for column in missing[:15]) or "<tr><td colspan='4'>No missing values detected.</td></tr>"
    numeric = [column for column in column_profiles if column.logical_type == "numerical"]
    categorical = [column for column in column_profiles if column.logical_type == "categorical"]
    target_profile = next((column for column in column_profiles if column.name == plan.target_column), None)
    target_stats = target_profile.numeric_statistics if target_profile else None
    target_chart = ""
    if plan.task_type.value == "regression" and target_stats:
        target_chart = _bar_chart("Target distribution summary", ["Minimum", "Q1", "Median", "Mean", "Q3", "Maximum"], [target_stats.min or 0, target_stats.percentile_25 or 0, target_stats.median or 0, target_stats.mean or 0, target_stats.percentile_75 or 0, target_stats.max or 0], color="#6d5dfc")
    elif target_profile and target_profile.top_values:
        target_chart = _bar_chart("Target class distribution", [value.value or "Missing" for value in target_profile.top_values], [float(value.count) for value in target_profile.top_values], color="#44b78b")
    correlation_chart = _bar_chart("Strongest verified correlations", [f"{item.column_x} ↔ {item.column_y}" for item in correlations[:8]], [abs(item.coefficient) for item in correlations[:8]], color="#d16a9d")
    missing_chart = _bar_chart("Columns with missing values", [column.name for column in missing[:10]], [float(column.missing_count) for column in missing[:10]], color="#e6a23c")
    feature_chart = _bar_chart("Top feature importance", [item.feature_name for item in feature_items[:8]], [item.importance for item in feature_items[:8]], color="#44b78b")
    diagnostic_chart, diagnostic_summary = ("", "")
    if plan.task_type.value == "regression":
        diagnostic_chart, diagnostic_summary = _regression_diagnostics(selected, dataframe, plan.target_column, settings)
    comparison_rows = "".join(
        f"<tr{' class=\"selected\"' if row['model_run_id'] == selected.id else ''}><td>{escape(row['model_name'].replace('_', ' ').title())}{' (selected)' if row['model_run_id'] == selected.id else ''}</td><td>{_metric(row['metrics'].get(evaluation.primary_metric), evaluation.primary_metric)}</td></tr>"
        for row in evaluation.validation_comparison
    )
    optimization_text = "Not run." if optimization is None else (
        f"Status: {escape(optimization.status)} · Model: {escape(optimization.model_name)} · "
        f"Trials: {optimization.trial_count:,} · Best validation {_metric_label(evaluation.primary_metric)}: "
        f"{_metric(optimization.best_validation_score, evaluation.primary_metric)}."
    )
    recommendations = ["HIGH PRIORITY: Validate on future or external data before operational use; final performance is based on one persisted split."]
    if profile_summary.total_missing_values:
        recommendations.append(f"HIGH PRIORITY: Review the {profile_summary.total_missing_values:,} missing values identified in the data-quality section.")
    if profile_summary.duplicate_row_count:
        recommendations.append(f"MEDIUM PRIORITY: Review the {profile_summary.duplicate_row_count:,} duplicate rows before retraining.")
    if profile_summary.possible_id_columns:
        recommendations.append("MEDIUM PRIORITY: Confirm identifier-like columns are excluded from future training runs.")
    if profile_summary.constant_columns:
        recommendations.append("OPTIONAL: Remove constant columns because they cannot add predictive signal.")
    recommendation_html = "".join(f"<li>{escape(item)}</li>" for item in recommendations)
    task_words = "classification" if plan.task_type.value != "regression" else "regression"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>AutoDS-Agent analytics report</title><style>body{{font-family:Arial,sans-serif;max-width:980px;margin:2rem auto;color:#172033;line-height:1.55}}h1{{font-size:2rem}}h1,h2,h3{{color:#075c9e}}section{{break-inside:avoid}}.summary,.card{{padding:1rem 1.3rem;border:1px solid #d8e5ef;border-radius:10px;margin:1rem 0;background:#fbfdff}}.summary{{border-left:5px solid #1687d4}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem}}.metrics div{{background:#eef7fc;padding:.7rem;border-radius:7px}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd;padding:.55rem;text-align:left}}th{{background:#f3f8fc}}tr.selected td{{background:#e7f5ff;font-weight:bold}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem}}figure{{margin:1rem 0;padding:1rem;border:1px solid #d8e5ef;border-radius:8px}}figcaption{{font-weight:bold;margin-bottom:.5rem}}svg{{max-width:100%;height:auto}}.label{{font:12px Arial;fill:#34495e}}.value{{font:12px Arial;fill:#172033}}.note{{color:#536b84}}@media print{{body{{margin:1cm}}section{{page-break-inside:avoid}}}}</style></head><body><h1>AutoDS-Agent Analytics Report</h1><p>Generated: {datetime.now(timezone.utc).isoformat()}</p><section class='summary'><h2>Executive Summary</h2><div class='metrics'><div><strong>Dataset</strong><br>{escape(dataset.original_filename)}<br>{dataset.row_count:,} rows · {dataset.column_count} columns</div><div><strong>Best model</strong><br>{escape(selected.model_name.replace('_', ' ').title())}<br>Primary metric: {escape(_metric_label(evaluation.primary_metric))}</div><div><strong>Final test result</strong><br>{_metric(evaluation.final_test_metrics.get(evaluation.primary_metric), evaluation.primary_metric)}<br>Held-out test set</div></div><ul><li><strong>Objective:</strong> {escape(experiment.user_objective)}</li><li><strong>Task and target:</strong> {escape(plan.task_type.value.replace('_', ' '))} · <strong>{escape(plan.target_column)}</strong></li><li><strong>Model selection:</strong> selected by verified validation {escape(_metric_label(evaluation.primary_metric))}; final metrics use an untouched test split.</li><li><strong>Main limitation:</strong> results are limited to this dataset and one persisted train/validation/test split.</li></ul></section><section><h2>Dataset Overview</h2><div class='card'><p>This dataset contains {dataset.row_count:,} records and {dataset.column_count} columns for a {escape(task_words)} experiment targeting <strong>{escape(plan.target_column)}</strong>.</p><p><strong>Feature roles:</strong> {len(numeric)} numerical · {len(categorical)} categorical · {len(plan.text_columns)} text · {len(profile_summary.possible_id_columns)} identifier-like.</p><p><strong>Quality summary:</strong> {profile_summary.total_missing_values:,} missing values · {profile_summary.duplicate_row_count:,} duplicate rows · constants: {escape(', '.join(profile_summary.constant_columns) or 'None')}.</p></div></section><section><h2>Data Quality Analysis</h2><table><tr><th>Column</th><th>Missing</th><th>Missing %</th><th>Unique values</th></tr>{quality_rows}</table>{missing_chart}<p><strong>Possible identifiers:</strong> {escape(', '.join(profile_summary.possible_id_columns) or 'None detected')}.</p></section><section><h2>Exploratory and Target Analysis</h2>{target_chart}{correlation_chart}<p class='note'>Correlations describe association only; they do not establish causation.</p>{'' if not target_stats else f"<table><tr><th>Target statistic</th><th>Verified value</th></tr><tr><td>Minimum</td><td>{_number(target_stats.min)}</td></tr><tr><td>Q1</td><td>{_number(target_stats.percentile_25)}</td></tr><tr><td>Median</td><td>{_number(target_stats.median)}</td></tr><tr><td>Mean</td><td>{_number(target_stats.mean)}</td></tr><tr><td>Q3</td><td>{_number(target_stats.percentile_75)}</td></tr><tr><td>Maximum</td><td>{_number(target_stats.max)}</td></tr></table>"}</section><section><h2>Data Preparation</h2><ul><li>Numerical values: {escape(plan.numeric_imputation.value)} imputation; scaling: {escape(plan.numeric_scaling.value)}.</li><li>Categorical values: {escape(plan.categorical_imputation.value)} imputation and {escape(plan.categorical_encoding.value)} encoding.</li><li>Excluded identifier-like columns: {escape(', '.join(plan.excluded_columns) or 'None detected')}.</li><li>Text processing: {escape(', '.join(plan.text_columns) or 'No text columns')}.</li><li>Split strategy: persisted random state with separate training, validation, and untouched test partitions.</li></ul></section><section><h2>Model Comparison</h2><table><tr><th>Model</th><th>Validation {escape(_metric_label(evaluation.primary_metric))}</th></tr>{comparison_rows}</table><p><strong>{escape(selected.model_name.replace('_', ' ').title())}</strong> won because it had the selected verified validation metric.</p></section><section><h2>Final Model Performance</h2><h3>Final untouched test metrics</h3><table>{_metric_rows(evaluation.final_test_metrics)}</table>{'' if plan.task_type.value == 'regression' else '<p>Classification metrics are percentages of the relevant held-out outcomes; unavailable metrics are marked not applicable.</p>'}{'' if plan.task_type.value != 'regression' else f"<p>The model explains approximately {_metric(evaluation.final_test_metrics.get('r2'), 'r2')} of observed target variation on the held-out test set. Average absolute error is {_metric(evaluation.final_test_metrics.get('mae'), 'mae')} target units and RMSE is {_metric(evaluation.final_test_metrics.get('rmse'), 'rmse')} target units.</p>"}</section><section><h2>Prediction Quality Analysis</h2>{diagnostic_chart}{diagnostic_summary}</section><section><h2>Future Prediction Results</h2><p class='note'>Verified from persisted prediction outputs. No future values are generated while creating this report.</p>{prediction_section}</section><section><h2>Feature Analysis</h2>{feature_chart}<ul>{features}</ul><p>Feature importance indicates predictive influence, not causation.</p></section><section><h2>Decision Audit and Reproducibility</h2><div class='card'><p><strong>Selection evidence:</strong> {len(evaluation.validation_comparison)} candidate model(s) compared using validation {escape(_metric_label(evaluation.primary_metric))}; the final result is calculated once on an untouched test partition.</p><p><strong>Data provenance:</strong> {escape(dataset.original_filename)} · {dataset.row_count:,} rows · {dataset.column_count} columns · experiment {escape(experiment.id)}.</p><p><strong>Governance note:</strong> prediction samples above are persisted outputs from the selected saved artifact, not LLM-generated estimates. Feature importance and correlations are associative, not causal.</p></div></section><section><h2>Optimization</h2><p>{optimization_text}</p></section><section><h2>Recommendations</h2><ul>{recommendation_html}</ul></section><section><h2>Technical Appendix</h2><p><strong>Experiment ID:</strong> {escape(experiment.id)} · <strong>Generated:</strong> {datetime.now(timezone.utc).isoformat()}</p><table><tr><th>Configuration</th><th>Verified setting</th></tr><tr><td>Numeric imputation</td><td>{escape(plan.numeric_imputation.value)}</td></tr><tr><td>Categorical encoding</td><td>{escape(plan.categorical_encoding.value)}</td></tr><tr><td>Scaling</td><td>{escape(plan.numeric_scaling.value)}</td></tr><tr><td>Excluded columns</td><td>{escape(', '.join(plan.excluded_columns) or 'None')}</td></tr></table></section></body></html>"""
    html = html.replace("<h2>Future Prediction Results</h2>", "<h2>Prediction Outlook for Supplied Rows</h2>")
    report_id = str(uuid4())
    filename = f"{report_id}.html"
    confined_artifact_path(settings.report_storage_path, filename, ".html").write_text(html, encoding="utf-8")
    record = Report(id=report_id, experiment_id=experiment.id, artifact_filename=filename)
    database.add(record)
    database.commit()
    database.refresh(record)
    return ReportResponse(report_id=record.id, experiment_id=experiment.id, status="COMPLETED", download_url=f"/api/v1/reports/{record.id}", created_at=record.created_at)


def create_pdf_report(experiment_id: str, database: Session, settings: Settings) -> ReportResponse:
    experiment, _, plan = _load_training_definition(experiment_id, database)
    evaluation = get_evaluation(experiment.id, database)
    selected = database.get(ModelRun, evaluation.selected_model_run_id)
    if selected is None:
        raise TrainingNotReadyError("Selected model is unavailable.")
    dataset = get_dataset_metadata(experiment.dataset_id, database, settings)
    dataframe = load_dataset_dataframe(dataset, settings)
    profile_summary, columns, correlations = profile_dataframe(dataframe, top_values_limit=8)
    target = next((column for column in columns if column.name == plan.target_column), None)
    numeric = [column for column in columns if column.logical_type == "numerical"]
    categorical = [column for column in columns if column.logical_type == "categorical"]
    missing = [column for column in columns if column.missing_count]
    try:
        explanation = get_explainability(experiment.id, database, settings)
        features = explanation.global_feature_importance[:10]
    except TrainingNotReadyError:
        features = []
    optimization = database.query(OptimizationResult).filter_by(experiment_id=experiment.id).one_or_none()
    prediction_summaries = _prediction_summaries(experiment.id, database, settings, plan.task_type.value)
    report_id = str(uuid4())
    filename = f"{report_id}.pdf"
    path = confined_artifact_path(settings.report_storage_path, filename, ".pdf")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#075c9e"), spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportHeading", parent=styles["Heading2"], textColor=colors.HexColor("#075c9e"), spaceBefore=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#536b84")))
    story = []
    def heading(text: str) -> None:
        story.append(Paragraph(text, styles["ReportHeading"]))
    def paragraph(text: str) -> None:
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 6))
    def table(rows: list[list[str]], widths: list[float] | None = None) -> None:
        item = Table(rows, colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf6ff")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#075c9e")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#ccd9e5")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")])]))
        story.append(item); story.append(Spacer(1, 10))

    story += [Paragraph("AutoDS-Agent Analytics Report", styles["ReportTitle"]), Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()}", styles["Small"]), Spacer(1, 10)]
    heading("Executive Summary")
    table([["Dataset", "Best model", "Final test result"], [f"{dataset.original_filename}\n{dataset.row_count:,} rows, {dataset.column_count} columns", selected.model_name.replace("_", " ").title(), f"{evaluation.primary_metric}: {_pdf_metric(evaluation.final_test_metrics.get(evaluation.primary_metric), evaluation.primary_metric)}"]], [2.1 * inch, 2.1 * inch, 2.3 * inch])
    paragraph(f"Objective: {experiment.user_objective}. Task: {plan.task_type.value.replace('_', ' ')}. Target: {plan.target_column}. The selected model was chosen on verified validation {evaluation.primary_metric}; final metrics come from the untouched test split.")
    paragraph("Main limitation: this result is specific to the uploaded dataset and one persisted train/validation/test split. Recommended next action: validate on future or external data before operational use.")
    heading("Key Findings")
    findings = [f"Dataset quality: {profile_summary.total_missing_values:,} missing values and {profile_summary.duplicate_row_count:,} duplicate rows.", f"Selected model: {selected.model_name.replace('_', ' ').title()} using validation {evaluation.primary_metric}."]
    findings += [f"Final test {name.upper() if name in {'mae', 'rmse'} else name}: {_pdf_metric(value, name)}." for name, value in evaluation.final_test_metrics.items() if value is not None]
    for finding in findings[:6]: paragraph("- " + finding)
    heading("Dataset Overview and Data Quality")
    paragraph(f"Feature roles: {len(numeric)} numerical, {len(categorical)} categorical, {len(plan.text_columns)} text, and {len(profile_summary.possible_id_columns)} identifier-like columns. Possible IDs: {', '.join(profile_summary.possible_id_columns) or 'None detected'}. Constants: {', '.join(profile_summary.constant_columns) or 'None detected'}.")
    table([["Column", "Missing", "Missing %", "Unique"]] + [[column.name, f"{column.missing_count:,}", f"{column.missing_percentage:.2f}%", f"{column.unique_count:,}"] for column in missing[:15]] or [["Column", "Missing", "Missing %", "Unique"], ["None", "0", "0.00%", "-"]], [2.4 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch])
    chart = _pdf_bar_chart("Missing values by column", [column.name for column in missing], [float(column.missing_count) for column in missing], colors.HexColor("#e6a23c"))
    if chart: story.append(chart)
    heading("Exploratory Data Analysis and Target Analysis")
    if target and target.numeric_statistics:
        stats = target.numeric_statistics
        table([["Target statistic", "Verified value"], ["Minimum", _number(stats.min)], ["Q1", _number(stats.percentile_25)], ["Median", _number(stats.median)], ["Mean", _number(stats.mean)], ["Q3", _number(stats.percentile_75)], ["Maximum", _number(stats.max)]], [2.3 * inch, 3.4 * inch])
        chart = _pdf_bar_chart("Target distribution summary", ["Minimum", "Q1", "Median", "Mean", "Q3", "Maximum"], [stats.min or 0, stats.percentile_25 or 0, stats.median or 0, stats.mean or 0, stats.percentile_75 or 0, stats.max or 0], colors.HexColor("#6d5dfc"))
    elif target and target.top_values:
        chart = _pdf_bar_chart("Target class distribution", [item.value or "Missing" for item in target.top_values], [float(item.count) for item in target.top_values], colors.HexColor("#44b78b"))
    else: chart = None
    if chart: story.append(chart)
    correlation_chart = _pdf_bar_chart("Strongest verified correlations", [f"{item.column_x} / {item.column_y}" for item in correlations[:8]], [abs(item.coefficient) for item in correlations[:8]], colors.HexColor("#d16a9d"))
    if correlation_chart: story.append(correlation_chart)
    paragraph("Correlations and feature importance indicate association or predictive influence only; they do not establish causation.")
    heading("Data Preparation")
    paragraph(f"Numerical values used {plan.numeric_imputation.value} imputation and {plan.numeric_scaling.value} scaling. Categorical values used {plan.categorical_imputation.value} imputation and {plan.categorical_encoding.value} encoding. Excluded identifier-like columns: {', '.join(plan.excluded_columns) or 'None detected'}. Text processing columns: {', '.join(plan.text_columns) or 'None detected'}. The saved split uses separate training, validation, and untouched test partitions.")
    heading("Model Comparison")
    comparison_headers = ["Model", f"Validation {evaluation.primary_metric}"]
    table([comparison_headers] + [[item["model_name"].replace("_", " ").title() + (" (selected)" if item["model_run_id"] == selected.id else ""), _pdf_metric(item["metrics"].get(evaluation.primary_metric), evaluation.primary_metric)] for item in evaluation.validation_comparison], [3.4 * inch, 2.3 * inch])
    compare_chart = _pdf_bar_chart("Validation model comparison", [item["model_name"].replace("_", " ") for item in evaluation.validation_comparison if item["metrics"].get(evaluation.primary_metric) is not None], [float(item["metrics"][evaluation.primary_metric]) for item in evaluation.validation_comparison if item["metrics"].get(evaluation.primary_metric) is not None])
    if compare_chart: story.append(compare_chart)
    heading("Final Test Performance")
    table([["Metric", "Verified test value"]] + [[name.upper() if name in {"mae", "rmse"} else {"r2": "R²", "f1": "F1", "roc_auc": "ROC-AUC"}.get(name, name.title()), _pdf_metric(value, name)] for name, value in evaluation.final_test_metrics.items()], [2.5 * inch, 3.2 * inch])
    if evaluation.final_test_confusion_matrix:
        heading("Classification Confusion Matrix")
        matrix = evaluation.final_test_confusion_matrix
        table([["Actual / Predicted", *matrix["labels"]]] + [[matrix["labels"][index], *[str(value) for value in row]] for index, row in enumerate(matrix["matrix"])], None)
    if plan.task_type.value == "regression":
        heading("Prediction Quality")
        paragraph("Regression diagnostics use predictions regenerated locally from the selected trusted artifact and its persisted test indices.")
        try:
            actual_values, predicted_values, residual_values = _regression_arrays(selected, dataframe, plan.target_column, settings)
            diagnostics = pd.Series(residual_values)
            table([["Verified diagnostic", "Value (target units)"], ["Mean residual", _number(float(diagnostics.mean()), 3)], ["Residual standard deviation", _number(float(diagnostics.std()), 3)], ["90th percentile absolute error", _number(float(abs(diagnostics).quantile(.9)), 3)], ["Largest absolute error", _number(float(abs(diagnostics).max()), 3)]], [2.8 * inch, 2.9 * inch])
            actual_chart = _pdf_scatter_chart("Actual vs predicted (held-out test set)", actual_values, predicted_values, "Actual target", "Predicted target", reference=True)
            residual_chart = _pdf_scatter_chart("Residuals vs predicted (held-out test set)", predicted_values, residual_values, "Predicted target", "Residual", reference=False)
            labels, counts = _histogram_counts(residual_values)
            residual_histogram = _pdf_bar_chart("Residual distribution", labels, counts, colors.HexColor("#6d5dfc"))
            for chart in (actual_chart, residual_chart, residual_histogram):
                if chart: story.append(chart)
        except Exception:
            paragraph("Prediction diagnostics are unavailable because the trusted test artifact could not be read.")
    heading("Prediction Outlook for Supplied Rows")
    paragraph("These are verified predictions from the selected model for rows in user-supplied prediction files. They are not automatic time-series forecasts or guaranteed future outcomes. Report generation never creates or invents prediction values. Classification confidence is a model-reported score, not accuracy or a calibrated probability.")
    if not prediction_summaries:
        paragraph("No prediction runs have been created yet. Create predictions from a CSV containing the required feature columns to add a verified outlook here.")
    for index, item in enumerate(prediction_summaries, start=1):
        paragraph(f"Prediction run {index}: generated {item['created_at']}; {int(item['row_count']):,} output rows. Verified output summary: {item['insight']}")
        if item["statistics"]:
            table([["Prediction summary", "Verified value"]] + [[str(label), str(value)] for label, value in item["statistics"]], [2.5 * inch, 3.2 * inch])
        if item["chart_labels"]:
            chart = _pdf_bar_chart(str(item["chart_title"]), list(item["chart_labels"]), list(item["chart_values"]), colors.HexColor("#1687d4"))
            if chart: story.append(chart)
        if item["confidence_labels"]:
            chart = _pdf_bar_chart("Model-reported confidence distribution", list(item["confidence_labels"]), list(item["confidence_values"]), colors.HexColor("#44b78b"))
            if chart: story.append(chart)
        columns = item["columns"]
        preview = item["preview"]
        if columns and preview:
            pdf_preview = preview[:12]
            table([list(columns)] + [list(row) for row in pdf_preview], None)
            paragraph(f"Preview limited to the first {len(pdf_preview):,} saved output rows.")
    heading("Feature Analysis")
    if features:
        table([["Feature", "Importance", "Direction"]] + [[feature.feature_name, f"{feature.importance:.6f}", feature.direction or "-"] for feature in features], [3.5 * inch, 1.3 * inch, 1.0 * inch])
        feature_chart = _pdf_bar_chart("Top feature importance", [feature.feature_name for feature in features[:8]], [feature.importance for feature in features[:8]], colors.HexColor("#44b78b"))
        if feature_chart: story.append(feature_chart)
    else: paragraph("Native feature importance was unavailable for the selected artifact.")
    paragraph("Feature importance indicates predictive influence, not causation.")
    heading("Decision Audit and Reproducibility")
    paragraph(f"Selection evidence: {len(evaluation.validation_comparison)} candidate model(s) were compared using validation {evaluation.primary_metric}; final performance was calculated on the untouched test partition. Data provenance: {dataset.original_filename}, {dataset.row_count:,} rows, {dataset.column_count} columns, experiment {experiment.id}.")
    paragraph("Governance note: the future-prediction preview is a persisted model artifact. It is not an LLM-generated forecast. Correlations and feature importance are associative evidence, not causal claims.")
    heading("Optimization")
    paragraph("Optimization was not run." if optimization is None else f"Optimization status: {optimization.status}. Model: {optimization.model_name}. Trials: {optimization.trial_count}. Best validation score: {_pdf_metric(optimization.best_validation_score, evaluation.primary_metric)}.")
    heading("Recommendations and Limitations")
    pdf_recommendations = ["HIGH PRIORITY: Validate on future or external data before operational use; final performance is based on one persisted split."]
    if profile_summary.total_missing_values:
        pdf_recommendations.append(f"HIGH PRIORITY: Review the {profile_summary.total_missing_values:,} missing values identified in the data-quality section.")
    if profile_summary.duplicate_row_count:
        pdf_recommendations.append(f"MEDIUM PRIORITY: Review the {profile_summary.duplicate_row_count:,} duplicate rows before retraining.")
    if profile_summary.possible_id_columns:
        pdf_recommendations.append("MEDIUM PRIORITY: Confirm identifier-like columns are excluded from future training runs.")
    if profile_summary.constant_columns:
        pdf_recommendations.append("OPTIONAL: Remove constant columns because they cannot add predictive signal.")
    for item in pdf_recommendations:
        paragraph(item)
    heading("Technical Appendix")
    paragraph(f"Experiment ID: {experiment.id}. Dataset: {dataset.original_filename}. Target: {plan.target_column}. Task: {plan.task_type.value}. Numeric imputation: {plan.numeric_imputation.value}. Categorical encoding: {plan.categorical_encoding.value}. Scaling: {plan.numeric_scaling.value}. Randomized split configuration and raw plan configuration remain persisted with the experiment.")
    document = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=52, title="AutoDS-Agent Analytics Report")
    document.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    record = Report(id=report_id, experiment_id=experiment.id, artifact_filename=filename)
    database.add(record); database.commit(); database.refresh(record)
    return ReportResponse(report_id=record.id, experiment_id=experiment.id, status="COMPLETED", download_url=f"/api/v1/reports/{record.id}", created_at=record.created_at)
