# syntax=docker/dockerfile:1
# Reinforcement Car — single-image deploy: Node builds the UI, Python serves
# both the API and the built UI same-origin (no CORS, no separate frontend host).
# CPU-only torch (cloud has no GPU). Works on Render / Fly / Railway / HF Spaces.

# 1) Build the frontend
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # -> /web/dist

# 2) Python backend + built UI
FROM python:3.10-slim AS app
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 RLCAR_HOST=0.0.0.0
# CPU torch first (its own index), then the lighter deps
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY backend/ ./backend/
COPY --from=web /web/dist ./frontend/dist
WORKDIR /app/backend
EXPOSE 5000
CMD ["python", "serve.py"]
