# AutoDS-Agent Architecture

## Purpose and trust rule

AutoDS-Agent is an authenticated web application for dataset profiling, safe ML workflow automation, verified analytics, reports, and context-aware questions.

Every LLM-controlled action follows one mandatory boundary:

```text
LLM -> structured plan -> Pydantic validation -> allowlisted trusted Python -> verified persisted result
```

LLMs can plan or explain. They cannot execute arbitrary code, query the database directly, read arbitrary files, or invent dataset/model metrics.

## Complete system overview

```text
 Optional advisory providers
 ┌───────────────────────────┐
 │ Gemini primary | Ollama   │
 └─────────────┬─────────────┘
               │ structured plans / bounded explanation only
               ▼
┌──────────┐  ┌─────────────────────────────────────────────────────────────┐
│ Browser  │  │ FastAPI application                                           │
│ React UI │─►│ auth · owner scope · Pydantic schemas · routes · rate limits │
└────▲─────┘  └───┬───────────────────┬──────────────────┬──────────────────┘
     │            │                   │                  │
     │            ▼                   ▼                  ▼
     │     Dataset services     Trusted ML engine    Ask AutoDS router
     │     profile/analytics    plan/train/evaluate  evidence/context/RAG
     │            │                   │                  │
     │            │                   ▼                  │
     │            │              Celery worker           │
     │            │                   │                  │
     │            ▼                   ▼                  ▼
     │     ┌──────────────────────────────────────────────────────────────┐
     └─────│ PostgreSQL         Redis              Confined artifact store │
           │ users/results      queue/rate limits  data/models/reports     │
           └──────────────────────────────────────────────────────────────┘
```

## Docker runtime services

| Service | Technology | Responsibility |
|---|---|---|
| `frontend` | React + Vite in development; React + Nginx in production | Browser UI, workflow pages, dashboards, Ask Q&A, history, reports, settings. |
| `backend` | FastAPI, SQLAlchemy, Alembic | API, auth, ownership, input/output contracts, orchestration, health/readiness. |
| `worker` | Celery | Queued training, evaluation, optimization, explainability, and report jobs. |
| `postgres` | PostgreSQL | Authoritative users, metadata, plans, verified results, jobs, and bounded assistant memory. |
| `redis` | Redis | Celery broker/result backend and distributed production rate-limit counters. |
| `autods_storage` | Docker volume / configured directories | Uploaded files, trusted model artifacts, prediction CSVs, HTML/PDF reports. |

Development exposes the UI on port `5173` and API on `8000`. The production overlay exposes Nginx on `8080` by default and proxies `/api/` to FastAPI. PostgreSQL and Redis are not intended to be public services.

## Frontend

The React frontend provides:

- Password login/registration and optional Google Identity Services login.
- Dashboard with persisted workspace information and a controlled multi-agent workflow summary.
- New Experiment workflow: upload, profile, objective confirmation, plan, queued execution, results, optimization, explainability, predictions, and reports.
- History for owner-scoped experiments and saved datasets.
- Ask Q&A with separate Dataset Questions and Experiment Questions contexts.
- Report history with HTML/PDF view/download actions.
- Settings that change local workspace preferences, not datasets or ML results.

API clients use `VITE_API_BASE_URL`. The frontend never calculates trusted metrics and never connects directly to PostgreSQL, Redis, or artifact storage.

## API layer

FastAPI exposes versioned endpoints under `/api/v1`.

| Group | Responsibilities |
|---|---|
| `/health`, `/health/ready` | Liveness and dependency readiness. Readiness checks PostgreSQL, Redis, expected Alembic revision, and a responding Celery worker. |
| `/auth` | Password and optional Google ID-token authentication; AutoDS JWT issuance. |
| `/datasets` | Owner-scoped CSV/XLSX lifecycle, worksheet selection, profiles, and safe analytics. |
| `/experiments` | Objective confirmation, plans, queued ML actions, evaluation, optimization, explainability, predictions, reflections, and reports. |
| `/jobs` | Background job state and experiment history/details. |
| `/artifacts` | Owner-scoped prediction/report listing, display, and download. |
| `/assistant` | Dataset, experiment, mixed, conversation, general, and research questions. |

Routes remain thin. Pydantic schemas validate request/response contracts; service modules contain business logic; SQLAlchemy sessions isolate database access; Alembic manages migrations.

## Identity, ownership, and data protection

```text
Credentials or Google ID token
  -> server-side verification
  -> AutoDS JWT
  -> authenticated API request
  -> owner-scoped metadata/artifact operation
```

- Passwords use Argon2 hashes and are never returned.
- Google tokens are verified server-side using `GOOGLE_CLIENT_ID`; `VITE_GOOGLE_CLIENT_ID` only enables the browser button.
- Datasets, experiments, jobs, reports, predictions, and assistant turns are scoped to their owner.
- Production rate limiting uses Redis and fails closed. Development can use a local fallback if Redis is unavailable.
- Secrets are injected through environment variables and excluded from Git and Docker build contexts.
- Generated artifact paths are confined to configured storage roots; internal stored dataset filenames are not exposed through dataset responses.

## Dataset ingestion and profiling

```text
CSV/XLSX upload
  -> extension/size/worksheet/parse validation
  -> generated internal storage filename
  -> owner-scoped dataset metadata in PostgreSQL
  -> deterministic profile
  -> profile + safe target/task recommendations in UI
```

The deterministic profiler calculates rows, columns, missingness, duplicates, Pandas/logical types, cardinality, constants, possible identifiers, bounded categorical frequencies, numeric summaries, and finite correlations. The Data Analyst Agent may interpret bounded profile metadata, but it does not replace those calculations.

Raw CSV/XLSX rows are not supplied to LLMs for numeric analysis.

## Controlled multi-agent workflow

```text
Data Analyst Agent
  Deterministic profile evidence and data-quality context
       │
       ▼
ML Planner Agent
  Structured task-compatible pipeline proposal
       │
       ▼
Pydantic validation + adaptive decisions
  Supported target/task/preprocessing/models/metrics only
       │
       ▼
Trusted ML Engine
  Train, evaluate, optimize, explain, predict, and persist
       │
       ▼
Insight & Reflection Agent
  Explains verified evidence and gives advisory next steps
```

These are controlled functional boundaries, not independent arbitrary-code agents.

## Objective, planning, and adaptive pipeline decisions

1. A user creates an experiment for a saved dataset and objective.
2. Deterministic/profile-backed logic identifies conservative target/task candidates.
3. The user confirms the target and task.
4. The ML Planner receives only bounded profile metadata plus the confirmed target/task.
5. Gemini is the primary structured provider; Ollama is the fallback.
6. The `PipelinePlan` contract validates the provider response before execution.
7. Adaptive services evaluate verified characteristics: dataset size, missingness, class balance, cardinality, outliers, feature types, text/datetime presence, dimensionality, and target distribution.
8. The UI records each decision as decision, reason, evidence, and allowlisted action taken.

Malformed provider output, invalid configuration, timeouts, or provider failures result in a controlled failure. No unvalidated partial plan can run.

## Trusted ML engine

### Training

```text
Persisted validated plan
  -> owner-scoped dataset load
  -> reproducible train / validation / untouched test split
  -> fit preprocessing on training rows only
  -> train registered allowlisted candidates
  -> persist model-run metadata and artifact references
```

- Preprocessing is fit only on training data to avoid leakage.
- Numeric, categorical, text, and safe datetime handling follow the persisted plan.
- Candidate model factories are registered trusted implementations; no LLM code is executed.
- Model artifacts use generated names under confined storage.
- Local measurements such as training time, inference time when available, complexity summaries, and run status support benchmarking.

### Evaluation and model selection

- Candidate models are compared using the configured **validation** primary metric.
- Selection and tie-breaking are deterministic.
- Only the selected model is evaluated against the untouched final test partition.
- Classification and regression results use task-specific metrics.
- Benchmark UI shows persisted validation performance, timing, complexity, generalization evidence, and status. It does not invent a weighted AI score.

### Optimization, explainability, predictions, reports

- Optuna optimization uses bounded application-owned search spaces and never uses test rows for tuning.
- Explainability uses supported native feature importances or model coefficients from trusted persisted artifacts.
- Prediction uploads must provide required features and produce persisted output files.
- HTML/PDF report generation uses persisted/calculated facts, task-specific diagnostics, feature analysis, existing prediction outputs, limitations, and recommendations.

## Background jobs

```text
UI action
  -> FastAPI creates PENDING job in PostgreSQL
  -> Celery message through Redis
  -> worker executes trusted service
  -> result reference / status persisted
  -> frontend polls job and refreshes experiment detail
```

Supported queued types are `train`, `evaluate`, `optimize`, `explain`, and `report`. The worker records `PENDING`, `RUNNING`, `COMPLETED`, or `FAILED`, uses bounded retries, and stores safe error summaries without secrets or raw data.

## Ask AutoDS

### Context-aware evidence routing

```text
Question + selected context + bounded previous turns
  -> evidence router
  -> dataset | experiment | dataset+experiment | conversation | general | research
  -> safe handler
  -> provenance-labelled answer
  -> bounded persisted assistant turn
```

The router uses selected context, dataset schema, persisted experiment evidence, question signals, and bounded conversation context. A Dataset Question is never sent to experiment retrieval merely because an experiment exists.

### Dataset Questions

```text
Natural-language question
  -> generic local planner or structured provider fallback
  -> DatasetAnalyticsPlan Pydantic validation
  -> allowlisted Pandas/NumPy analytics executor
  -> verified answer, table/chart data, calculation explanation
```

Supported operations are bounded filtering, aggregation, grouping, ranking, sorting, distributions, correlations, associations, outlier summaries, comparisons, and safe derived calculations. The schemas bound filters, grouping, rows, columns, operators, and aggregations.

Arbitrary Python, SQL, `eval`, `exec`, imports, paths, URLs, and shell commands are forbidden.

### Experiment Questions and other evidence

- **Experiment:** retrieves owner-scoped persisted task/target, plan, model runs, validation comparison, final-test metrics, optimization, predictions/reports, and compatible historical evidence. It never invents a metric.
- **Mixed:** compares supported persisted experiment metrics with safe local dataset statistics only when the link is unambiguous.
- **Conversation:** stores a bounded history of intent/context; numeric results are recalculated instead of copied from an earlier answer.
- **General:** provider output must satisfy a structured claim-basis response and is displayed as **AI Explanation**, not as a verified result.
- **Research:** local curated AutoML/data-science papers provide advisory sources only. RAG cannot alter training, model selection, metrics, or execution.

Each answer has an evidence type and provenance label, including **Verified from dataset**, **Verified from experiment**, **Verified from dataset + experiment**, **Retrieved research evidence**, or **AI Explanation**.

## Persistence and artifact model

| Domain | PostgreSQL records |
|---|---|
| Identity | users and Google identity linkage |
| Datasets | owner, metadata, lifecycle state; not raw dataset bytes |
| Experiments | objective, confirmed task/target, selected model reference, status |
| ML | validated plans, model runs, evaluation results, optimization results |
| Artifacts | prediction runs and report records linked to confined files |
| Operations | jobs and safe status/error metadata |
| Assistant | bounded owner-scoped turns and source metadata |

Raw data, model binaries, prediction CSVs, and HTML/PDF files live under configured artifact storage, not database rows.

## Deployment and health flow

```text
Docker Compose
  -> PostgreSQL + Redis become healthy
  -> backend applies Alembic migrations and starts FastAPI
  -> worker connects to Redis and registers tasks
  -> backend readiness verifies DB, Redis, migration revision, worker response
  -> frontend starts after backend is healthy
```

Production uses protected environment variables, persistent volumes, and a reverse proxy/TLS layer. See [FREE_DEPLOYMENT_GUIDE.md](FREE_DEPLOYMENT_GUIDE.md) for deployment instructions.

## Explicit limitations

- LLMs are not trusted numeric sources and cannot execute generated code.
- Offline curated RAG is advisory, not live scholarly search.
- Historical experiments are context, not a guarantee for a different dataset.
- Feature importance and correlation are associative, not causal.
- Free cloud resources can constrain training, uptime, worker capacity, and storage; these constraints do not relax safety rules.
- Report prediction sections describe already persisted predictions for user-supplied feature rows. They are not automatic forecasts unless the explicit time-series capability is used on suitable data.
