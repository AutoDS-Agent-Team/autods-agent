# Free Deployment Guide: AutoDS-Agent

This guide deploys the **complete AutoDS-Agent stack** for learning, portfolio demonstrations, and small private testing without paying for hosting.

It uses one Linux virtual machine and Docker Compose, so every required component runs together:

```text
Browser
  -> Nginx frontend
  -> FastAPI backend
  -> PostgreSQL (users, experiments, metadata)
  -> Redis (rate limits and Celery queue)
  -> Celery worker (training, evaluation, reports)
  -> Docker volume (uploaded datasets, models, predictions, reports)
```

## Important limits

- This is suitable for a student project, demo, or small personal deployment—not a high-traffic production service.
- Never commit `.env`, uploaded datasets, reports, models, or database dumps to GitHub.
- Free cloud capacity can be unavailable or reclaimed. Keep local backups.
- Gemini, Google OAuth, and a custom domain are optional. The project still works with email/password login and local deterministic analytics if Gemini is not configured.

## Recommended free setup

| Component | Free tool | Why it is used |
|---|---|---|
| Source control + CI | GitHub | Stores source code and runs the included CI workflow. |
| Complete runtime | Oracle Cloud Always Free Compute | Runs Docker, backend, frontend, PostgreSQL, Redis, Celery, and storage together. |
| HTTPS / custom domain (optional) | Cloudflare | DNS and edge TLS options. |
| Google sign-in (optional) | Google Cloud OAuth | Provides the Google Web Client ID. |
| Gemini explanations (optional) | Google AI Studio / Gemini API | Used only for explanations and plans; verified calculations remain local. |

Oracle documents Always Free compute resources, but availability varies by home region and idle instances can be reclaimed. Check the current limits before creating a VM: [Oracle Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm).

## 1. Prepare GitHub

1. Create an empty GitHub repository, for example `autods-agent`.
2. From your project folder, confirm that `.env` is ignored:

   ```bash
   git check-ignore .env
   ```

3. Review files before the first commit:

   ```bash
   git status
   git add .
   git status
   ```

4. Confirm that these are **not** staged:

   - `.env` or `.env.*` except `.env.example`
   - `storage/datasets/*`, `storage/models/*`, `storage/reports/*`, and `storage/predictions/*`
   - `.venv/`, `node_modules/`, `dist/`, test-result folders, and local databases
   - passwords, API keys, JWT secrets, private keys, and certificates

5. Commit and push only after that review:

   ```bash
   git commit -m "Initial AutoDS-Agent release"
   git branch -M main
   git remote add origin https://github.com/YOUR-ACCOUNT/autods-agent.git
   git push -u origin main
   ```

## 2. Create a free Linux VM

1. Create an Oracle Cloud account and choose your home region carefully.
2. In **Compute > Instances**, create an Always Free eligible Ubuntu VM.
3. Prefer an Ampere A1 shape if it is available and has enough RAM. A small x86 micro VM can be too limited for simultaneous model training, PostgreSQL, Redis, and a worker.
4. Create or download an SSH key during instance setup. Keep the private key only on your computer; never put it in GitHub.
5. In the VM network security list/firewall, allow inbound TCP:

   - `22` only from your own IP address, for SSH
   - `80` for HTTP
   - `443` for HTTPS when TLS is configured

   Do **not** expose ports `5432`, `6379`, or `8000` publicly.

6. Connect from your computer:

   ```bash
   ssh -i /path/to/private-key.pem ubuntu@YOUR_SERVER_PUBLIC_IP
   ```

## 3. Install Docker on the VM

Run these commands on the Ubuntu server:

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit
```

Connect again, then confirm Docker is usable without `sudo`:

```bash
ssh -i /path/to/private-key.pem ubuntu@YOUR_SERVER_PUBLIC_IP
docker --version
docker compose version
```

## 4. Clone the project and configure secrets

```bash
git clone https://github.com/YOUR-ACCOUNT/autods-agent.git
cd autods-agent
cp .env.example .env
chmod 600 .env
```

Edit the protected `.env` file:

```bash
nano .env
```

Set unique values for the following settings. Do not use the example placeholders.

```env
APP_ENV=production
POSTGRES_PASSWORD=use-a-long-random-password
JWT_SECRET_KEY=use-a-long-random-secret-at-least-32-characters
DATABASE_URL=postgresql+psycopg://autods:YOUR_POSTGRES_PASSWORD@postgres:5432/autods
FRONTEND_URL=http://YOUR_SERVER_PUBLIC_IP:8080
CORS_ORIGINS=["http://YOUR_SERVER_PUBLIC_IP:8080"]
PRODUCTION_HTTP_PORT=8080
```

Generate strong local values on the server without printing them into Git:

```bash
openssl rand -base64 48
```

### Optional Google login

If you want the Google sign-in button:

1. Create a **Web application** OAuth client in Google Cloud Console.
2. Add the deployed origin, for example:

   ```text
   http://YOUR_SERVER_PUBLIC_IP:8080
   ```

   Use your `https://your-domain.example` origin after TLS is set up.

3. Add the same Web Client ID to both variables:

   ```env
   GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   ```

The current application uses Google Identity Services ID tokens, so it does not use or need a Google OAuth client secret.

### Optional Gemini

For Gemini-generated explanations, set your own key:

```env
GEMINI_API_KEY=your-own-key
```

Without it, verified local dataset analytics, training, reports, and deterministic fallbacks continue to work. Do not publish the key.

## 5. Start the complete production stack

From the project root on the VM, validate the compose configuration:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet
```

Build and start every component:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

This starts:

| Service | Purpose |
|---|---|
| `frontend` | Nginx-served React application on port `8080` by default. |
| `backend` | FastAPI API and Alembic migrations. |
| `worker` | Celery worker for queued ML training/evaluation/report jobs. |
| `postgres` | Persistent users, datasets metadata, experiments, and results. |
| `redis` | Celery broker/result backend and distributed rate limiting. |
| `autods_storage` volume | Persistent uploaded data, models, predictions, and reports. |

Check every service:

```bash
docker compose ps
curl -f http://127.0.0.1:8000/api/v1/health/ready
docker compose logs --tail=100 backend
docker compose logs --tail=100 worker
```

Open this from your browser:

```text
http://YOUR_SERVER_PUBLIC_IP:8080
```

The backend readiness endpoint verifies PostgreSQL, Redis, the required Alembic revision, and a responding Celery worker.

## 6. Add HTTPS and a free domain layer (recommended)

For a public demonstration, use a domain you control and Cloudflare DNS.

1. Add the domain to Cloudflare and change its authoritative nameservers at your domain registrar.
2. Add an `A` DNS record pointing to the VM public IP.
3. Update `.env`:

   ```env
   FRONTEND_URL=https://your-domain.example
   CORS_ORIGINS=["https://your-domain.example"]
   ```

4. Update the Google OAuth authorized JavaScript origin to:

   ```text
   https://your-domain.example
   ```

5. Put a TLS reverse proxy such as Caddy or Nginx in front of the frontend container and expose only `80`/`443`.

Cloudflare Pages is a good free option for a **static-only** React frontend; its current limits are documented here: [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/). For this project, one VM is simpler because the backend also needs Celery, Redis, persistent datasets, report files, and model artifacts.

## 7. Daily operations

### View status and logs

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f worker
```

### Update after pushing to GitHub

```bash
cd ~/autods-agent
git pull --ff-only
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose ps
```

Back up first if migrations or model/report storage changed.

### Stop or restart

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart
```

Do not run `docker compose down -v` unless you intentionally want to delete the PostgreSQL and artifact volumes.

## 8. Backups

The database and artifact volume must be backed up together.

```bash
mkdir -p ~/autods-backups
docker compose exec -T postgres pg_dump -U autods autods > ~/autods-backups/autods-$(date +%F).sql
docker run --rm \
  -v autods-agent_autods_storage:/data:ro \
  -v ~/autods-backups:/backup \
  alpine tar czf /backup/autods-storage-$(date +%F).tgz -C /data .
```

Copy the backup files to your own computer or private cloud storage. Test restoring a backup before relying on it.

## 9. Free-tier alternatives and trade-offs

### Render + hosted services

Render can host free web services, Postgres, and key-value services, but its free web services spin down after inactivity and have an ephemeral filesystem. Free background workers are not a dependable fit for this full Celery architecture. See [Render free-tier limits](https://render.com/docs/free) and [Render compute plans](https://render.com/docs/compute-plans).

Use Render only for a short demo API/frontend or after changing the architecture to use paid/persistent workers and object storage.

### Cloudflare Pages + separate backend

Cloudflare Pages can deploy the frontend from GitHub for free. You would still need a compatible backend host, PostgreSQL, Redis, a Celery worker, and durable artifact storage. This is more complicated than the one-VM approach.

### Neon PostgreSQL and Upstash Redis

Neon and Upstash can be useful free managed services for prototypes. They are optional and should be selected only after testing connectivity, TLS, Celery compatibility, quotas, and regional latency. Upstash documents a free Redis tier, but this project uses standard Redis URLs for Celery and should not be switched to an HTTP-only Redis API without code/configuration changes. See [Upstash Redis](https://upstash.com/redis) and [SQLAlchemy PostgreSQL connection URLs](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html).

## 10. Final public-deployment checklist

- [ ] `.env` is not staged or committed.
- [ ] GitHub repository contains only source, migrations, Docker files, documentation, tests, and `.env.example`.
- [ ] Strong unique PostgreSQL and JWT values are set.
- [ ] Ports `5432`, `6379`, and `8000` are not public.
- [ ] `docker compose ps` shows backend healthy and worker running.
- [ ] `/api/v1/health/ready` returns success locally on the VM.
- [ ] A new account can be created and a small CSV can be uploaded.
- [ ] One experiment completes and an HTML/PDF report downloads.
- [ ] Google sign-in is tested only after the correct deployed origin is authorized.
- [ ] PostgreSQL and `autods_storage` backups are tested.

## Useful references

- [Oracle Cloud Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- [Render Docker deployment](https://render.com/docs/docker)
- [Render free-tier limits](https://render.com/docs/free)
- [Cloudflare Pages documentation](https://developers.cloudflare.com/pages/)
- [Upstash Redis documentation](https://upstash.com/docs/redis/overall/getstarted)
