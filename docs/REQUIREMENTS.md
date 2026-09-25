# Requirements

## Product goal

AutoDS-Agent will turn a structured CSV dataset and natural-language objective into a safe, reproducible data-science workflow with evaluation, explanations, predictions, and reports.

## Implemented scope

- React/Vite application with backend health status
- Versioned FastAPI application with a health endpoint
- Environment-driven configuration, CORS, logging, and exception handling
- SQLAlchemy and Alembic database foundation
- PostgreSQL and Redis development services
- Dockerfiles and Docker Compose orchestration
- Automated health-route test
- Size-limited multipart CSV upload with safe generated filenames
- Dataset metadata persistence and Alembic migration
- Deterministic Pandas profiling with bounded categorical frequencies
- Dataset metadata and profile endpoints
- Minimal React dataset upload and profile display
- Header and headerless CSV support with persisted detection metadata and deterministic generated column names
- Persistent experiment definitions linked to datasets
- Deterministic target candidate generation and objective-to-column matching
- Binary classification, multiclass classification, and regression task suggestions
- Explicit ambiguity for low-cardinality integer targets
- Backend-validated task and target confirmation
- React objective definition and confirmation flow
- Gemini primary and Ollama fallback providers with bounded retries/timeouts
- Metadata-only structured pipeline planning
- Strict Pydantic plan validation and allowlisted values
- Persistent validated pipeline plans and provider attribution
- Reproducible train/validation/test splitting
- Training-only preprocessing fit with mean/median imputation, one-hot encoding, and optional standard scaling
- Trusted logistic/ridge regression, random forest, and XGBoost baseline factories
- Model-run metadata and filesystem artifact persistence
- React plan review and baseline-training status display
- Validation comparison, deterministic primary-metric selection, and untouched-test evaluation
- Bounded trusted Optuna optimization of the selected baseline
- Native model feature-importance/coefficient explainability
- Compatible prediction CSV upload and downloadable prediction artifacts
- Verified HTML report generation and download

## Explicitly out of scope

- Celery workers or tasks
- SHAP-specific explanations and PDF rendering
- Authentication
- Complete product dashboard

## Safety requirement

All future LLM-controlled operations must use validated structured plans and a registry of trusted functions. Generated code must never be executed.
