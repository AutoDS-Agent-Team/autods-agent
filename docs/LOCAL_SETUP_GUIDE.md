# AutoDS-Agent: Set Up on a Laptop

Use this guide to run AutoDS-Agent on a Windows, macOS, or Linux laptop. The easiest method uses Docker, so you do not need to install Python, Node.js, PostgreSQL, or Redis separately.

## What you need

1. A laptop with internet access and at least 8 GB RAM available for development.
2. [Git](https://git-scm.com/downloads).
3. [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   - Windows users: during installation, choose the recommended WSL 2 backend if Docker asks.
   - Start Docker Desktop after installing it and wait until it says it is running.
4. An editor such as [Visual Studio Code](https://code.visualstudio.com/) (optional, but recommended).
5. A copy of this project. Ask the project owner for the GitHub repository link if you do not have one.

## Step 1: Get the project

Open **PowerShell** on Windows, or **Terminal** on macOS/Linux. Go to the folder where you keep projects, then clone the repository.

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
```

Replace `OWNER/REPOSITORY` with the link supplied by the project owner. For example, if the repository is called `my-org/autods-agent`, use:

```bash
git clone https://github.com/my-org/autods-agent.git
cd autods-agent
```

If you received the project as a ZIP file instead, extract it, open a terminal in the extracted `autods-agent` folder, and continue with Step 2.

## Step 2: Create your local settings file

In the project root (the folder containing `docker-compose.yml`), make a copy of `.env.example` called `.env`.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS or Linux**

```bash
cp .env.example .env
```

The `.env` file contains settings that are only for your laptop. Do not upload it to GitHub or share its secrets.

## Step 3: Set the required secrets

Open `.env` in your editor. Change these two example values:

```env
POSTGRES_PASSWORD=replace-with-a-strong-local-password
JWT_SECRET_KEY=replace-with-a-random-secret-at-least-32-characters
```

Use your own values. For example:

```env
POSTGRES_PASSWORD=MyLocalDatabasePassword_2026
JWT_SECRET_KEY=change-this-to-a-long-random-private-value-12345
```

Keep the database password the same in both places in the file:

```env
POSTGRES_PASSWORD=MyLocalDatabasePassword_2026
DATABASE_URL=postgresql+psycopg://autods:MyLocalDatabasePassword_2026@postgres:5432/autods
```

Leave the other settings as they are unless your project owner tells you otherwise.

### Optional: enable Gemini

The app can start without a Gemini key, but AI planning and assistant features need one. If you have a Gemini API key, add it here:

```env
GEMINI_API_KEY=your-key-goes-here
```

### Optional: enable Google sign-in

Google sign-in is optional. If the project owner provides a Google Web Client ID, put the same value in both settings:

```env
GOOGLE_CLIENT_ID=your-google-web-client-id
VITE_GOOGLE_CLIENT_ID=your-google-web-client-id
```

## Step 4: Start the application

Make sure Docker Desktop is running. In the project root, run:

```bash
docker compose up -d --build
```

The first startup downloads images and builds the app, so it may take several minutes. Later starts are usually faster.

## Step 5: Confirm that every service started

Run:

```bash
docker compose ps
```

You should see four services: `postgres`, `redis`, `backend`, `frontend`, and `worker`. Their status should become `running` or `healthy`.

If a service is still starting, wait about one minute and run the command again.

## Step 6: Open AutoDS-Agent

Open this address in a browser:

```text
http://localhost:5173
```

You can also check the backend directly:

```text
http://localhost:8000/api/v1/health
```

The health page should return a small JSON response showing that the backend is alive.

## Step 7: Use the application

1. Create an account or sign in.
2. Upload a CSV or Excel file.
3. Review the dataset profile and suggested analysis.
4. Select or confirm the target and task.
5. Run an analysis, then view predictions, explanations, reports, or Ask AutoDS results.

## Everyday commands

Run these commands from the project root.

| What you want to do | Command |
| --- | --- |
| Start the app | `docker compose up -d` |
| Stop the app | `docker compose down` |
| See status | `docker compose ps` |
| See all logs | `docker compose logs -f` |
| See backend logs only | `docker compose logs -f backend` |
| Rebuild after code changes | `docker compose up -d --build` |

To stop viewing logs, press `Ctrl+C`. This only stops the log view; it does not stop the app.

## If something goes wrong

### Docker command is not found

Install Docker Desktop, restart the terminal, and make sure Docker Desktop is open and running.

### The page at localhost:5173 does not open

1. Run `docker compose ps`.
2. If `frontend` or `backend` is not running, inspect the logs:

   ```bash
   docker compose logs --tail=100 frontend backend
   ```

3. Rebuild and start again:

   ```bash
   docker compose up -d --build
   ```

### Port 5173, 8000, 5432, or 6379 is already in use

Another program is using a required port. Stop that program, or ask the project owner before changing the port mappings in `docker-compose.yml`.

### The backend keeps restarting

Check that `.env` exists and that `POSTGRES_PASSWORD` matches the password used inside `DATABASE_URL`. Then run:

```bash
docker compose logs --tail=100 backend
```

### I changed `.env` but nothing changed

Restart the containers so they read the updated settings:

```bash
docker compose down
docker compose up -d --build
```

## Stop or remove local data

To stop the project but keep your local database and uploaded artifacts, run:

```bash
docker compose down
```

To delete all local project containers **and all local database/uploaded data**, run the following only if you are sure you no longer need that data:

```bash
docker compose down -v
```

## Getting project updates later

Before pulling updates, save or commit your own changes. Then run:

```bash
git pull
docker compose up -d --build
```

Do not replace your `.env` file with a version from Git. If `.env.example` has new settings, compare it with your `.env` and add only the needed new keys.
