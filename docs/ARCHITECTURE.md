# Architecture

## General data-science extension

```text
                DATASET
                   │
                   ▼
         Data Analyst Agent
                   │
        deterministic profiling
                   │
                   ▼
       Auto Problem Detection
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
 Supervised    Unsupervised   Time Series
       │           │            │
       ▼           ▼            ▼
 ML Planner    Safe Planner   Forecast Planner
       │           │            │
       └───────────┼────────────┘
                   ▼
           Trusted ML Engine
                   │
                   ▼
        Verified Results Store
                   │
          ┌────────┴────────┐
          ▼                 ▼
 Insight/Reflection      Ask AutoDS
                            │
                     Question Planner
                            │
                     Pydantic Validation
                            │
                     Trusted Analytics
                            │
                     Verified Answer
```

Feature roles, target scores, task compatibility, aggregates, correlations, anomaly scores, forecast metrics, and model metrics are calculated by deterministic trusted code. LLMs do not calculate arbitrary dataset statistics or model metrics. Unsupported questions return a missing-evidence response rather than an invented answer.

The analytics engine accepts only enumerated operations, aggregations, and filter operators, with at most five AND filters, two group-by columns, and 100 result rows. It accepts no SQL, Python expressions, imports, paths, URLs, `eval`, `exec`, or shell input.

## Frontend

The React/Vite frontend is the browser-facing client. It renders backend health, CSV upload and profiles, objective confirmation, provider-attributed pipeline-plan review, baseline statuses, validation comparison, selected model/test metrics, confusion matrix, optimization result, native feature importance, prediction upload/download, and HTML report download. API calls are isolated in small client modules and configured with `VITE_API_BASE_URL`.

## FastAPI backend

FastAPI exposes versioned endpoints under `/api/v1`. Routes remain thin and delegate business operations to services. Pydantic models define request and response contracts. Centralized settings, logging, exception handling, database sessions, and Alembic migrations provide the application foundation.

## Dataset ingestion and profiling

The dataset route streams multipart CSV uploads through the dataset service. The service validates filename/type, enforces the configured byte limit, generates a UUID filename, confines writes to the configured dataset directory, detects whether a header is present, and asks Pandas to parse the file before committing metadata. PostgreSQL stores the original display filename, internal stored filename, size, dimensions, header decision, and creation time; CSV content remains in artifact storage.

Header detection samples up to 25 non-empty rows. It recognizes a header only when the first row consists entirely of plausible, unique textual column names and later values provide consistent numeric or boolean type evidence. Ambiguous input is treated as headerless to preserve its first row. Headerless datasets receive deterministic names (`column_1`, `column_2`, ...), and the persisted decision is reused on every profile read.

A separate structural check recognizes a scikit-learn-style metadata preamble only when its declared feature count matches the row width, its sample count is plausible, its remaining cells are textual class labels, and sampled feature values are consistently numeric. That metadata row is skipped without being treated as a CSV header; generated names remain visible and the encoded label column is treated categorically.

Migration marks pre-detection dataset records as having an unknown header state. The dataset service lazily re-detects those stored files on their next metadata or profile request, updates dimensions, and persists the decision before producing a response.

The profiling service deterministically calculates dimensions, duplicates, missingness, Pandas and logical types, cardinality, constants, conservative possible-ID flags, bounded categorical frequencies, numerical statistics, and finite pairwise numeric correlations. Logical numeric inference uses actual values: object/string columns with at least 95% numeric non-null values are profiled numerically without changing their stored raw values or reported Pandas dtype. It does not call an LLM and converts non-finite statistics to JSON `null`.

## Objective interpretation and confirmation

The experiment service creates an experiment from a dataset ID and objective, then uses the trusted dataset schema and values to produce conservative target candidates. It matches normalized column names mentioned in the objective; it does not claim general natural-language understanding. Constant, unusable, high-cardinality categorical, and possible-ID columns are not automatically suggested.

Categorical targets with two classes suggest binary classification, while categorical targets with more classes suggest multiclass classification. Numeric targets suggest regression unless they are low-cardinality integers, which remain explicitly ambiguous between classification and regression. Confirmation is a separate backend operation that validates the target against the stored dataset and validates task-specific class or numeric constraints. A valid confirmation changes the experiment status to `READY_FOR_PLANNING`; it does not create a pipeline.

## Future multi-agent controller

A controller will coordinate specialized agents for profiling, planning, training, evaluation, explanation, and reporting. Agents will exchange typed state and plans; they will not execute generated source code.

## LLM router and structured planning

The planning service constructs a bounded prompt from objective and profile metadata only. It sends column names, logical types, dimensions, missingness, cardinality, constant flags, and possible-ID flags; it never sends CSV rows or top-value samples. Gemini uses the current `google-genai` SDK as the primary provider. Ollama's `/api/chat` endpoint with a configurable model (default `qwen3:8b`) is the fallback.

The router retries within configured bounds and falls back on provider errors, timeout, unavailability, rate limiting, malformed JSON, or local contract failure. If both providers fail, it returns a controlled error and persists nothing. Logs contain provider names and failure categories but not secrets, complete datasets, or prompt contents.

Provider-native JSON schemas improve response reliability, but local `PipelinePlan` validation remains authoritative. Strict enums allow only supported preprocessing, models, and metrics. Cross-field validation rejects classification/regression mismatches and requires the plan task and target to exactly match the confirmed experiment.

## Trusted execution rule

Every future LLM-controlled operation must follow:

```text
LLM -> structured JSON plan -> Pydantic validation -> registered trusted Python functions
```

The runtime must reject invalid plans and unknown operations. It must never evaluate or execute arbitrary Python, shell commands, SQL, or serialized objects supplied by an LLM.

## Trusted preprocessing and baseline training

The training service loads the persisted validated plan, separates the confirmed target, and creates reproducible train/validation/test splits. Classification uses stratification when class counts and split sizes permit it and falls back cleanly when they do not. Possible-ID feature columns remain present.

A trusted registry maps validated names to scikit-learn and XGBoost factories. Numeric features use configured mean/median imputation and optional standard scaling. Categorical features use most-frequent imputation plus one-hot encoding with unknown-category handling. The `ColumnTransformer` is fit once on training data; validation and test data are transformed without fitting.

Only requested allowlisted baselines are trained. Classification supports logistic regression, random forest, and XGBoost; regression supports ridge regression, random forest, and XGBoost. Each model failure is isolated and recorded. Successful trusted artifacts are written under configured model storage with generated filenames, while PostgreSQL stores metadata and references.

## Evaluation, optimization, and final artifacts

The evaluation service loads only successful trusted artifacts and reconstructs the persisted validation/test rows from saved indices. It computes real classification or regression metrics from model predictions. All candidates are compared on validation data, and deterministic selection uses the plan's primary metric. Exact ties use model name then generated run ID. Only the selected candidate reaches the untouched test split.

The optimization service chooses the baseline selection result as the single promising candidate. Optuna receives bounded application-owned search spaces and trains on training data while measuring validation only. It never reads test rows. The optimized run is persisted as a normal trusted model artifact and comparison is rerun.

Native feature importances are used for forest/XGBoost models and coefficients for logistic/ridge models. Prediction CSVs are checked for all required feature columns, transformed by the final persisted pipeline, and saved using generated CSV filenames. HTML reports contain only persisted/calculated facts and remain available without an LLM.

## PostgreSQL

PostgreSQL stores dataset, experiment, validated plan, model-run, evaluation, optimization, prediction-run, and report metadata. `experiments.selected_model_run_id` freezes the currently selected model. CSV bytes, model binaries, prediction CSVs, and HTML reports remain in confined artifact storage rather than ordinary database rows. SQLAlchemy provides sessions and Alembic owns schema migrations.

## Redis and Celery

Redis is provisioned now as infrastructure. Celery will later execute long-running profiling and ML workflows outside HTTP request processes. No Celery tasks exist yet.

## Artifact storage

The `storage/` tree separates uploaded datasets, trained model artifacts, generated prediction CSVs, and generated HTML reports. All generated artifact paths are UUID filenames and must resolve directly under their configured root. Storage contents are ignored by Git while directory markers remain tracked. A future storage abstraction can move artifacts to object storage without changing service interfaces.

## Docker deployment

Docker Compose runs frontend, backend, PostgreSQL, and Redis services with health checks and dependency ordering. Environment variables provide all deployment-specific values and secrets.
