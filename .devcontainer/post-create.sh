#!/usr/bin/env bash
# Runs once after the container is created (devcontainer.json postCreateCommand).
# Installs Claude Code and any project dependencies that already exist.
set -euo pipefail

echo "==> Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Backend deps — only if the project has been scaffolded yet (greenfield-safe).
if [ -f backend/pyproject.toml ]; then
  echo "==> Installing backend (pyproject)..."
  (cd backend && uv pip install --system -e ".[dev]") || echo "  (skipped — fix backend deps later)"
elif [ -f backend/requirements.txt ]; then
  echo "==> Installing backend (requirements.txt)..."
  uv pip install --system -r backend/requirements.txt || echo "  (skipped — fix backend deps later)"
fi

# Frontend deps — only if scaffolded.
if [ -f frontend/package.json ]; then
  echo "==> Installing frontend deps..."
  (cd frontend && npm install) || echo "  (skipped — fix frontend deps later)"
fi

echo ""
echo "==> Dev container ready."
echo "    Postgres : ${DATABASE_URL:-postgresql+psycopg://geo:geo@db:5432/geothermal}"
echo "    Redis    : ${REDIS_URL:-redis://redis:6379/0}"
echo "    Run 'claude' to start Claude Code in the container."
