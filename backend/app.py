"""Flask backend for the Reinforcement Car trainer.

Exposes REST endpoints for control + checkpoints and a Server-Sent-Events
stream for live car telemetry, and (in production) serves the built frontend so
the whole app is a single same-origin deployable.

IMPORTANT (deployment): the trainer is a single in-process, stateful object with
a background training thread. Run exactly ONE worker/process — do NOT scale to
multiple gunicorn/uwsgi workers, or each would get its own independent trainer.
Use the provided `serve.py` (waitress) for production; `python app.py` is dev only.

Optional environment configuration (all safe to leave unset for local use):
  RLCAR_TOKEN         if set, POST/PUT/DELETE require header  X-Auth-Token: <token>
  RLCAR_CORS_ORIGINS  comma-separated allowed origins for /api/* (default: none,
                      i.e. same-origin only — not needed when Flask serves the UI)
  RLCAR_HOST / RLCAR_PORT   bind address/port (dev server defaults 127.0.0.1:5000)
"""

import json
import os
import time

from flask import Flask, jsonify, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS

from rl.trainer import Trainer
from rl.tracks import list_tracks

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.abspath(os.path.join(HERE, "..", "frontend", "dist"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("RLCAR_CORS_ORIGINS", "").split(",") if o.strip()]
AUTH_TOKEN = os.environ.get("RLCAR_TOKEN", "").strip()
SERVE_FRONTEND = os.path.isdir(FRONTEND_DIST)

app = Flask(__name__, static_folder=None)

# CORS is only enabled when origins are explicitly configured. In the normal
# single-origin deploy (Flask serves the built UI) and in dev (Vite proxies /api
# same-origin) no cross-origin requests happen, so no CORS headers are needed.
if CORS_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

trainer = Trainer(num_envs=20, window=8, rollout_steps=128)


@app.before_request
def _auth_gate():
    # Optional shared-token protection for state-changing requests. No-op unless
    # RLCAR_TOKEN is set, so local/dev usage is unaffected.
    if AUTH_TOKEN and request.method in ("POST", "PUT", "DELETE"):
        if request.headers.get("X-Auth-Token", "") != AUTH_TOKEN:
            return jsonify({"error": "unauthorized"}), 401


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/tracks")
def tracks():
    return jsonify(list_tracks())


@app.route("/api/track", methods=["GET", "POST"])
def track():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        geo = trainer.set_track(data.get("name", "default"))
        return jsonify(geo)
    return jsonify(trainer.env.track_geometry())


@app.route("/api/status")
def status():
    return jsonify(trainer.metrics())


@app.route("/api/history")
def history():
    return jsonify(trainer.get_history())


@app.route("/api/control", methods=["POST"])
def control():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    if action == "start":
        trainer.start()
    elif action == "pause":
        trainer.pause()
    elif action == "resume":
        trainer.resume()
    elif action == "stop":
        trainer.stop()
    elif action == "reset":
        trainer.reset_env()
    elif action == "resetProgress":
        trainer.reset_progress()
    else:
        return jsonify({"error": f"unknown action '{action}'"}), 400
    return jsonify(trainer.metrics())


@app.route("/api/config", methods=["POST"])
def config():
    data = request.get_json(force=True, silent=True) or {}
    if "fleet" in data:
        trainer.set_fleet(data["fleet"])
    trainer.set_config(data)
    return jsonify(trainer.metrics())


# ----------------------------------------------------------------- checkpoints
@app.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    return jsonify(trainer.list_checkpoints())


@app.route("/api/checkpoints", methods=["POST"])
def save_checkpoint():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name") or f"ckpt-step-{trainer.global_step}"
    info = trainer.save_checkpoint(name)
    return jsonify(info)


@app.route("/api/checkpoints/load", methods=["POST"])
def load_checkpoint():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name")
    try:
        info = trainer.load_checkpoint(name)
    except FileNotFoundError:
        return jsonify({"error": "checkpoint not found"}), 404
    except Exception as e:
        return jsonify({"error": f"incompatible checkpoint: {e}"}), 400
    return jsonify(info)


@app.route("/api/checkpoints/<name>", methods=["DELETE"])
def delete_checkpoint(name):
    ok = trainer.delete_checkpoint(name)
    if not ok:
        return jsonify({"error": "checkpoint not found"}), 404
    return jsonify({"deleted": name})


# ------------------------------------------------------------------- telemetry
@app.route("/api/telemetry")
def telemetry():
    @stream_with_context
    def gen():
        try:
            while True:
                payload = {"telemetry": trainer.get_snapshot(), "metrics": trainer.metrics()}
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1 / 30.0)  # ~30 Hz
        except GeneratorExit:
            # client disconnected — stop cleanly so the worker thread is freed
            return

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------- static frontend (prod)
if SERVE_FRONTEND:
    @app.route("/")
    def _index():
        return send_from_directory(FRONTEND_DIST, "index.html")

    @app.route("/<path:path>")
    def _static(path):
        if path.startswith("api/"):
            return jsonify({"error": "not found"}), 404
        full = os.path.join(FRONTEND_DIST, path)
        if os.path.isfile(full):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")  # SPA fallback


if __name__ == "__main__":
    # Development server only. For production use `python serve.py` (waitress).
    host = os.environ.get("RLCAR_HOST", "127.0.0.1")
    port = int(os.environ.get("RLCAR_PORT", "5000"))
    app.run(host=host, port=port, threaded=True, debug=False)
