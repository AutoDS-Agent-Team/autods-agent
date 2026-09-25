# Deployment

Deploy on Linux with Docker Engine and Docker Compose. Put configuration only in a protected root `.env`; never commit it.

## Configuration

Set values by name only: `JWT_SECRET_KEY`, `DATABASE_URL` or PostgreSQL variables, `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `CORS_ORIGINS`, storage path variables, `GEMINI_API_KEY`, `GEMINI_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `GOOGLE_CLIENT_ID`, `VITE_GOOGLE_CLIENT_ID`, `DATA_RETENTION_DAYS`, `ASSISTANT_MAX_EXPERIMENTS`, and `ASSISTANT_MAX_CONTEXT_CHARS`.

`GOOGLE_CLIENT_ID` and `VITE_GOOGLE_CLIENT_ID` may use the same public Google Web Client ID. No Google Client Secret is required for the current GIS ID-token flow. API keys and database credentials remain server-side.

## Start and verify

Validate the production overlay first:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose ps
curl -f http://127.0.0.1:8000/api/v1/health/ready
docker compose exec -T backend alembic current
docker compose logs --tail=100 worker
```

The production frontend uses Nginx, not the Vite development server. Put TLS termination in a reverse proxy/load balancer and forward `/api` to FastAPI while serving the frontend at `/`. Restrict CORS to the real browser origins.

## Data and backups

PostgreSQL and the artifact volume are persistent and must be backed up together. Back up PostgreSQL with `pg_dump` and archive the configured storage volume. Restore PostgreSQL before its matching artifacts. Test restore procedures before relying on them.

The backend applies Alembic migrations at startup. Before upgrades, back up database and artifacts, inspect the target migration, deploy, check worker logs and health, then run an authenticated smoke test. Roll back only to an image compatible with the migrated schema; otherwise restore the backup.

## Retention

Retention is off by default: `DATA_RETENTION_DAYS=0`. It is never automatic merely by deploying. When intentionally enabled, first inspect the impact:

```bash
docker compose exec -T backend python -m app.scripts.cleanup_retention --dry-run
```

The administrator-invoked service only removes expired persisted records and artifacts under configured, confined roots. It handles datasets, experiments, jobs through database relationships, model artifacts, prediction CSVs, and HTML/PDF reports. Take a backup before enabling destructive cleanup.

## Health and recovery

Confirm frontend availability through the public proxy, backend liveness at `/api/v1/health`, backend readiness at `/api/v1/health/ready`, PostgreSQL/Redis container health, and worker registration of `autods.execute_job`. Investigate failed health checks with `docker compose logs backend worker`; do not expose credentials or copied environment values in support logs.

## GitHub Actions

The repository includes `.github/workflows/ci.yml`. It installs the locked frontend dependencies, installs the backend package with development dependencies, runs the backend test suite, and builds the production frontend on pushes and pull requests targeting `main`. Deployment credentials and runtime secrets are intentionally not stored in the workflow.
