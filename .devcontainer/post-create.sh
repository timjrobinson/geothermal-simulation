#!/usr/bin/env bash
# Runs once after the container is created (devcontainer.json postCreateCommand).
# Installs Claude Code and any project dependencies that already exist.
set -euo pipefail

# Use the native installer (installs to ~/.local/bin/claude). The host's mounted
# ~/.claude.json records "installMethod": "native", so Claude Code expects the
# binary there — an npm -g install lands elsewhere and triggers a repair error.
echo "==> Installing Claude Code CLI..."
curl -fsSL https://claude.ai/install.sh | bash

# Gas Town toolchain. The `gt` CLI and its `bd` (beads) dependency ship as
# prebuilt native binaries via npm — installed here (not in the Dockerfile)
# because the Node feature only becomes available after the image is built.
# Their system deps (dolt, tmux, sqlite3, git) are baked into the image.
echo "==> Installing Gas Town (gt) + beads (bd)..."
npm install -g @gastown/gt @beads/bd

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
echo "    Run 'gt'     for the Gas Town CLI (gt install ~/gt --git to bootstrap a workspace)."
