# AutoDS-Agent

AutoDS-Agent is a secure, end-to-end data-science workflow for general structured/tabular CSV and Excel datasets within the implemented analysis types. It profiles a dataset, detects feature roles, recommends a target and task, validates plans and analytical queries, runs trusted functions, and preserves experiments, artifacts, and grounded assistant answers.

## Architecture

React (development Vite / production Nginx) → FastAPI → authentication and ownership → dataset/profile/auto-detection → Gemini/Ollama planning → Pydantic-validated plan → trusted allowlisted ML functions → Celery/Redis worker → PostgreSQL metadata and confined artifact storage → verified dataset analytics and experiment memory → grounded assistant.

Dataset questions follow a separate strict boundary: natural-language question → bounded question planner → Pydantic analytics query → allowlisted Pandas/NumPy calculation → structured verified answer. Dataset rows are never handed to an LLM for calculation.

### Controlled Multi-Agent AutoDS

Data Analyst Agent → user confirmation → ML Planner Agent → Pydantic validation → Trusted ML Engine → Insight & Reflection Agent → Ask AutoDS / reports.

The three LLM-powered agents reason over bounded verified metadata. Pandas, scikit-learn, XGBoost, Optuna, SHAP, Celery, Redis, and PostgreSQL are deterministic tools and infrastructure, not agents. **LLMs reason and interpret; trusted deterministic code calculates and executes.**

The critical trust boundary is permanent:

`LLM → structured JSON → Pydantic validation → trusted allowlisted Python functions`

LLMs never execute generated code, SQL, shell commands, imports, or arbitrary file access.

## Stack

- React + Vite, Nginx for production frontend
- FastAPI, Pydantic, SQLAlchemy, Alembic
- PostgreSQL, Redis, Celery
- Pandas, scikit-learn, XGBoost, Optuna, SHAP/native explainability
- Gemini primary provider and Ollama `qwen3:8b` fallback

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

Open `http://localhost:5173`; backend liveness is `http://localhost:8000/api/v1/health` and dependency readiness is `http://localhost:8000/api/v1/health/ready`.

For production-style frontend/Nginx:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

## Environment configuration

Keep real values in the ignored root `.env`, never in source control. Important names include `JWT_SECRET_KEY`, PostgreSQL/Redis/Celery settings, storage-root settings, `GEMINI_API_KEY`, `GEMINI_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `GOOGLE_CLIENT_ID`, `VITE_GOOGLE_CLIENT_ID`, `DATA_RETENTION_DAYS`, `ASSISTANT_MAX_EXPERIMENTS`, and `ASSISTANT_MAX_CONTEXT_CHARS`.

Google Identity Services uses the same public Web Client ID for `GOOGLE_CLIENT_ID` (backend verifier) and `VITE_GOOGLE_CLIENT_ID` (frontend button). This ID-token design does not require a Client Secret. Google credentials are verified server-side and exchanged for an AutoDS JWT; the AutoDS JWT is the only application authorization credential.

For host Ollama with Docker, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` and run `ollama pull qwen3:8b`.

## Workflow

Register or sign in → upload CSV/XLSX (one selected worksheet at a time) → inspect the deterministic profile and recommendation → accept or override the target/task → run supervised, clustering, anomaly, or chronological forecasting analysis → explain/generate predictions and HTML/PDF reports where supported → ask AutoDS for verified dataset calculations or persisted experiment facts.

Baseline candidates are selected using validation data only. The chosen baseline alone is evaluated on the untouched test set. Optimized artifacts are presented separately and never mixed into the baseline comparison.

## Testing

```bash
cd backend
python -m pytest -q

cd ../frontend
npm run build
```

## Security and operations

Passwords are Argon2 hashes; hashes, secrets, JWTs, paths, and raw CSV rows are never exposed in responses or assistant context. Every persisted resource and assistant-memory query is owner-scoped. Artifact operations are confined to configured roots. Write APIs are rate-limited. Retention is disabled by default (`DATA_RETENTION_DAYS=0`); an administrator may inspect cleanup with `python -m app.scripts.cleanup_retention --dry-run`.

See [Deployment](docs/DEPLOYMENT.md), [Architecture](docs/ARCHITECTURE.md), and [Project status](docs/PROJECT_STATUS.md).
