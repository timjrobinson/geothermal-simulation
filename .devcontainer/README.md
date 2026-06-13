# Dev Container

Reproducible dev environment for the geothermal simulator. Brings up the full
local stack from the design docs (`design/DECISIONS.md`): a Python+Node app
container plus **PostgreSQL/PostGIS** and **Redis** service containers, and
**Claude Code** preinstalled.

## What's inside

| Service | Image | Purpose | Port |
|---|---|---|---|
| `app` | `Dockerfile` (Python 3.12 + Node 20 + GDAL/PROJ + Claude Code + Gas Town) | where you develop/run code | 5173, 8000, 9181 |
| `db` | `postgis/postgis:16-3.4` | catalog DB (doc 04) | 5432 |
| `redis` | `redis:7-alpine` | RQ job queue (doc 04) | 6379 |

### Ports forwarded to your host

| Port | Service |
|---|---|
| 5173 | Vite — frontend dev server (doc 06) |
| 8000 | FastAPI — backend API (doc 04) |
| 9181 | RQ dashboard — job monitoring |
| 5432 | PostgreSQL + PostGIS |
| 6379 | Redis |

## Opening it

VS Code (or Cursor) → **"Reopen in Container"**, or the CLI:

```bash
devcontainer up --workspace-folder .
```

First build installs system libs and Claude Code (a few minutes); subsequent
starts are fast. The Postgres/Redis data persist in named Docker volumes
(`pgdata`, `redisdata`) across rebuilds.

## Claude inside the container

Your host `~/.claude` and `~/.claude.json` are bind-mounted in, so your existing
login and history carry over — just run:

```bash
claude
```

If you'd rather use an API key, export `ANTHROPIC_API_KEY` on the host before
launching; it's passed through to the container.

## Gas Town (`gt`)

[Gas Town](https://github.com/gastownhall/gastown) is preinstalled, so `gt` is on
the `PATH` as soon as the container is up. Its dependencies are provisioned for you:

| Tool | How it's installed | Why |
|---|---|---|
| `gt` | `npm i -g @gastown/gt` (post-create) | the CLI itself (prebuilt native binary) |
| `bd` | `npm i -g @beads/bd` (post-create) | beads — gt's issue/memory store |
| `dolt` | official installer (Dockerfile) | gt's versioned datastore |
| `tmux`, `sqlite3` | apt (Dockerfile) | session multiplexing + local stores |
| `git`, `claude` | base image / post-create | worktrees + default agent runtime |

Bootstrap a workspace the first time you use it:

```bash
gt install ~/gt --git && cd ~/gt
gt rig add myproject https://github.com/you/repo.git
gt mayor attach
```

## Connecting to the services

From inside the `app` container, reach the services by name:

```bash
psql "$DATABASE_URL"            # or: psql -h db -U geo geothermal   (password: geo)
redis-cli -h redis ping
```

`DATABASE_URL` and `REDIS_URL` are set in the environment for the backend.
PostGIS is enabled automatically on the `geothermal` database by the image.

## Notes

- **Port conflicts:** if your host already runs Postgres (5432) or Redis (6379),
  change the left side of the `ports:` mappings in `docker-compose.yml` (e.g.
  `"55432:5432"`).
- **MinIO (object store):** commented out in `docker-compose.yml` — uncomment it
  when you want an S3-compatible store locally for the hosted-phase path (doc 04 §10).
- **Dependencies:** `post-create.sh` installs backend/frontend deps only if
  `backend/` or `frontend/` already exist, so it's safe to build before the repo
  is scaffolded.
