# App/dev container. Node + Claude Code are layered on via devcontainer features
# and post-create (see devcontainer.json); this image provides Python + the
# geoscience system libraries the backend stack needs.
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# System dependencies:
#  - GDAL/PROJ      : underpin pyproj, rasterio, the spatial framework (doc 01)
#  - postgresql-client / redis-tools : psql + redis-cli for debugging the services
#  - build-essential: building any C-extension Python wheels
#  - tmux / sqlite3 : required by Gas Town (gt) — tmux drives the Mayor/agent
#                     sessions, sqlite3 backs its local stores
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gdal-bin libgdal-dev proj-bin proj-data \
       postgresql-client redis-tools \
       build-essential \
       tmux sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config

# Dolt — Gas Town's versioned datastore (the 'gt' workflow needs `dolt` on PATH).
# Official Linux installer detects arch and drops the binary in /usr/local/bin.
RUN curl -fsSL https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash

# Fast Python package manager (used by the backend; optional but standard now).
RUN pip install --no-cache-dir uv
