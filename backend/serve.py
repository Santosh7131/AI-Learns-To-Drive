"""Production entrypoint — serves the Flask app (API + built frontend) with the
waitress WSGI server (cross-platform, threaded, streaming-capable for SSE).

Run ONE instance only (the trainer is a single stateful in-process object):

    cd backend
    python serve.py            # serves on 0.0.0.0:5000 by default

Build the frontend first so it gets served same-origin:

    cd frontend && npm run build

Environment: RLCAR_HOST, RLCAR_PORT, RLCAR_THREADS, RLCAR_TOKEN, RLCAR_CORS_ORIGINS.
"""

import os

from waitress import serve

from app import app, SERVE_FRONTEND

if __name__ == "__main__":
    host = os.environ.get("RLCAR_HOST", "0.0.0.0")
    # honor the platform-provided $PORT (Render/Fly/Railway/HF), else RLCAR_PORT
    port = int(os.environ.get("PORT") or os.environ.get("RLCAR_PORT") or "5000")
    threads = int(os.environ.get("RLCAR_THREADS", "16"))
    if not SERVE_FRONTEND:
        print("[serve] WARNING: frontend/dist not found — API only. "
              "Run `cd frontend && npm run build` to serve the UI.")
    print(f"[serve] waitress serving on http://{host}:{port}  (threads={threads})")
    # channel_timeout kept generous so long-lived SSE connections aren't dropped
    serve(app, host=host, port=port, threads=threads, channel_timeout=3600)
