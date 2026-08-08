"""PPO training manager.

Runs a background thread that:
  * maintains a sliding window of the last K observations per car,
  * rolls out `rollout_steps` across all 20 envs,
  * performs a PPO update on the Transformer actor-critic,
  * publishes a thread-safe telemetry snapshot for the SSE stream.

Also handles checkpoint save / load / list / delete.
"""

import os
import time
import json
import glob
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .environment import CarEnv
from .model import TransformerActorCritic
from .tracks import list_tracks

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
HISTORY_PATH = os.path.join(CKPT_DIR, "history.json")
AUTOSAVE_NAME = "autosave"
BEST_NAME = "best"


class Trainer:
    def __init__(self, num_envs=20, window=8, rollout_steps=128, track="default"):
        # Use the GPU automatically when CUDA is available (falls back to CPU).
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
        if use_cuda:
            torch.backends.cudnn.benchmark = True
        print(f"[trainer] device: {self.device} ({self.device_name})")
        self.num_envs = num_envs
        self.window = window
        self.rollout_steps = rollout_steps
        self.track_name = track

        self.env = CarEnv(num_envs=num_envs, track=track)
        self.model = TransformerActorCritic(
            obs_dim=CarEnv.OBS_DIM, act_dim=CarEnv.ACT_DIM, window=window
        ).to(self.device)

        # --- hyperparameters (some adjustable from the frontend) ---
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.epochs = 4
        self.minibatches = 4
        self.ent_coef = 0.005
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.sim_delay = 0.012  # per-step sleep so motion is watchable

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # rolling observation window per car
        obs = self.env.reset()
        self.obs_window = np.zeros((num_envs, window, CarEnv.OBS_DIM), dtype=np.float32)
        self.obs_window[:, -1, :] = obs

        # --- runtime state ---
        self.status = "idle"  # idle | running | paused | stopped
        self._stop_flag = False
        self._pending_fleet = None  # requested num_envs, applied at a safe boundary
        self.lock = threading.Lock()
        self.thread = None

        # --- metrics ---
        self.global_step = 0
        self.updates = 0
        self.total_laps = 0
        self.ep_returns = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.best_return = float("-inf")
        self.last_loss = {}
        self._fps_t = time.time()
        self._fps_steps = 0
        self.fps = 0.0

        # persistence: training history + best-tracking
        self.history = []          # list of {step, updates, meanReturn, bestReturn}
        self.best_mean = float("-inf")
        self._load_history()
        self._try_resume()

        # telemetry snapshot consumed by the SSE stream
        self.snapshot = self._make_snapshot()

    # ------------------------------------------------------------- snapshot
    def _make_snapshot(self):
        render = self.env.render_state()
        render.update({
            "step": self.global_step,
            "status": self.status,
        })
        return render

    def get_snapshot(self):
        with self.lock:
            return self.snapshot

    def metrics(self):
        with self.lock:
            mean_ret = float(np.mean(self.ep_returns)) if self.ep_returns else 0.0
            mean_len = float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0
            return {
                "status": self.status,
                "track": self.track_name,
                "device": self.device_name,
                "numEnvs": self.num_envs,
                "globalStep": int(self.global_step),
                "updates": int(self.updates),
                "totalLaps": int(self.total_laps),
                "meanReturn": round(mean_ret, 3),
                "meanEpisodeLen": round(mean_len, 1),
                "bestReturn": round(self.best_return, 3) if self.best_return > float("-inf") else 0.0,
                "fps": round(self.fps, 1),
                "lr": self.lr,
                "loss": {k: round(float(v), 4) for k, v in self.last_loss.items()},
            }

    # --------------------------------------------------------------- control
    def start(self):
        if self.thread and self.thread.is_alive():
            if self.status == "paused":
                self.status = "running"
            return
        self._stop_flag = False
        self.status = "running"
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def pause(self):
        if self.status == "running":
            self.status = "paused"

    def resume(self):
        if self.status == "paused":
            self.status = "running"

    def stop(self):
        self._stop_flag = True
        self.status = "stopped"

    def reset_env(self):
        with self.lock:
            obs = self.env.reset()
            self.obs_window[:] = 0
            self.obs_window[:, -1, :] = obs
            self.snapshot = self._make_snapshot()

    def reset_progress(self):
        """Wipe all learning: fresh (random) model + optimizer, zeroed metrics
        and reward history, and remove the auto-saved resume/best checkpoints for
        this circuit. User-named checkpoints are kept (still loadable)."""
        with self.lock:
            self.model = TransformerActorCritic(
                obs_dim=CarEnv.OBS_DIM, act_dim=CarEnv.ACT_DIM, window=self.window
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            obs = self.env.reset()
            self.obs_window[:] = 0
            self.obs_window[:, -1, :] = obs
            self.global_step = 0
            self.updates = 0
            self.total_laps = 0
            self.ep_returns.clear()
            self.ep_lengths.clear()
            self.best_return = float("-inf")
            self.best_mean = float("-inf")
            self.last_loss = {}
            self.fps = 0.0
            self._fps_steps = 0
            self.history = []
            self.snapshot = self._make_snapshot()
        self._save_history()  # persist the now-empty curve
        for name in (self._autosave_name(), self._best_name()):
            try:
                os.remove(os.path.join(CKPT_DIR, name + ".pt"))
            except OSError:
                pass

    def set_track(self, name):
        """Switch circuits. Each track keeps its own saved model: if one exists
        it is loaded, otherwise the current weights carry over (transfer)."""
        valid = {t["id"] for t in list_tracks()}
        if name not in valid:
            name = "default"
        with self.lock:
            self.track_name = name
            self.env = CarEnv(num_envs=self.num_envs, track=name)
            obs = self.env.reset()
            self.obs_window[:] = 0
            self.obs_window[:, -1, :] = obs
            self.ep_returns.clear()
            self.ep_lengths.clear()
            self.total_laps = 0
            self.best_mean = float("-inf")
            self.best_return = float("-inf")
            self.snapshot = self._make_snapshot()

        # load this track's own model if we have one (else keep current weights)
        name_auto = self._autosave_name()
        if os.path.exists(os.path.join(CKPT_DIR, name_auto + ".pt")):
            try:
                self.load_checkpoint(name_auto)
                print(f"[trainer] loaded saved model for '{name}'")
            except Exception as e:
                print(f"[trainer] keeping current weights for '{name}' ({e})")
        return self.env.track_geometry()

    def set_fleet(self, n):
        """Change how many cars train in parallel (more = better GPU usage).
        Applied at a rollout boundary so it can't tear the in-flight buffers."""
        n = int(max(4, min(240, n)))
        self._pending_fleet = n
        running = self.thread is not None and self.thread.is_alive() and self.status == "running"
        if not running:
            self._apply_fleet()
        return n

    def _apply_fleet(self):
        n = self._pending_fleet
        if n is None:
            return
        self._pending_fleet = None
        with self.lock:
            self.num_envs = n
            self.env = CarEnv(num_envs=n, track=self.track_name)
            obs = self.env.reset()
            self.obs_window = np.zeros((n, self.window, CarEnv.OBS_DIM), dtype=np.float32)
            self.obs_window[:, -1, :] = obs
            self.ep_returns.clear()
            self.ep_lengths.clear()
            self.total_laps = 0
            self.snapshot = self._make_snapshot()

    def set_config(self, cfg):
        with self.lock:
            if "lr" in cfg:
                self.lr = float(cfg["lr"])
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr
            if "simDelay" in cfg:
                self.sim_delay = max(0.0, float(cfg["simDelay"]))
            if "entCoef" in cfg:
                self.ent_coef = float(cfg["entCoef"])

    # ---------------------------------------------------------------- window
    def _push_obs(self, obs):
        self.obs_window = np.roll(self.obs_window, -1, axis=1)
        self.obs_window[:, -1, :] = obs

    # ------------------------------------------------------------- main loop
    def _loop(self):
        while not self._stop_flag:
            if self._pending_fleet is not None:
                self._apply_fleet()  # safe: between rollouts
            if self.status == "paused":
                time.sleep(0.05)
                continue
            self._rollout_and_update()

    def _rollout_and_update(self):
        T, N = self.rollout_steps, self.num_envs
        K, O = self.window, CarEnv.OBS_DIM

        b_win = np.zeros((T, N, K, O), dtype=np.float32)
        b_act = np.zeros((T, N, CarEnv.ACT_DIM), dtype=np.float32)
        b_logp = np.zeros((T, N), dtype=np.float32)
        b_val = np.zeros((T, N), dtype=np.float32)
        b_rew = np.zeros((T, N), dtype=np.float32)
        # GAE mask: only TRUE terminations (off-track) cut the value bootstrap;
        # time-limit truncations keep bootstrapping (avoids time-limit bias).
        b_term = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            if self._stop_flag:
                return
            while self.status == "paused" and not self._stop_flag:
                time.sleep(0.05)

            env = self.env  # capture once for a consistent step across track swaps
            win_t = torch.from_numpy(self.obs_window).to(self.device)
            action, logp, value = self.model.act(win_t)
            action_np = action.cpu().numpy()

            obs, reward, done, info = env.step(action_np)

            b_win[t] = self.obs_window
            b_act[t] = action_np
            b_logp[t] = logp.cpu().numpy()
            b_val[t] = value.cpu().numpy()
            b_rew[t] = reward
            b_term[t] = np.asarray(info["terminated"], dtype=np.float32)

            self._push_obs(obs)
            # zero the window history for cars that just reset (off-track OR timeout)
            if done.any():
                self.obs_window[done, :-1, :] = 0.0

            now = time.time()
            update_fps = now - self._fps_t >= 0.5
            with self.lock:
                self.global_step += N
                self._fps_steps += N
                self.total_laps = info["total_laps"]
                for r in env.finished_returns:
                    self.ep_returns.append(r)
                    self.best_return = max(self.best_return, r)
                for l in env.finished_lengths:
                    self.ep_lengths.append(l)
                if update_fps:
                    self.fps = self._fps_steps / (now - self._fps_t)
                    self._fps_t = now
                    self._fps_steps = 0
                self.snapshot = self._make_snapshot()

            if self.sim_delay > 0:
                time.sleep(self.sim_delay)

        # bootstrap value for the final window
        with torch.no_grad():
            win_t = torch.from_numpy(self.obs_window).to(self.device)
            _, _, last_val = self.model.forward(win_t)
            last_val = last_val.cpu().numpy()

        self._ppo_update(b_win, b_act, b_logp, b_val, b_rew, b_term, last_val)
        self.updates += 1
        self._persist_after_update()

    # ------------------------------------------------------------- ppo update
    def _ppo_update(self, b_win, b_act, b_logp, b_val, b_rew, b_term, last_val):
        T, N = self.rollout_steps, self.num_envs

        # Generalized Advantage Estimation
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else b_val[t + 1]
            next_nonterminal = 1.0 - b_term[t]
            delta = b_rew[t] + self.gamma * next_val * next_nonterminal - b_val[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + b_val

        # flatten
        win = torch.from_numpy(b_win.reshape(T * N, self.window, CarEnv.OBS_DIM)).to(self.device)
        act = torch.from_numpy(b_act.reshape(T * N, CarEnv.ACT_DIM)).to(self.device)
        old_logp = torch.from_numpy(b_logp.reshape(T * N)).to(self.device)
        adv_t = torch.from_numpy(adv.reshape(T * N)).to(self.device)
        ret_t = torch.from_numpy(returns.reshape(T * N)).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        batch = T * N
        mb_size = batch // self.minibatches
        idxs = np.arange(batch)

        pol_losses, val_losses, ent_losses = [], [], []
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch, mb_size):
                mb = idxs[start:start + mb_size]
                mb_t = torch.from_numpy(mb).to(self.device)

                new_logp, entropy, value = self.model.evaluate(win[mb_t], act[mb_t])
                ratio = torch.exp(new_logp - old_logp[mb_t])
                a = adv_t[mb_t]

                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * a
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = ((value - ret_t[mb_t]) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pol_losses.append(policy_loss.item())
                val_losses.append(value_loss.item())
                ent_losses.append(entropy.mean().item())

        self.last_loss = {
            "policy": np.mean(pol_losses),
            "value": np.mean(val_losses),
            "entropy": np.mean(ent_losses),
        }

    # ------------------------------------------------------------- persistence
    def _load_history(self):
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "r") as f:
                    self.history = json.load(f)
                if self.history:
                    self.best_mean = max(
                        (h.get("meanReturn", float("-inf")) for h in self.history),
                        default=float("-inf"),
                    )
        except Exception:
            self.history = []

    def _save_history(self):
        try:
            tmp = HISTORY_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.history[-3000:], f)
            os.replace(tmp, HISTORY_PATH)  # atomic
        except Exception as e:
            print(f"[trainer] history save failed: {e}")

    # checkpoints are named per-track so each circuit keeps its own model
    def _autosave_name(self):
        return f"{AUTOSAVE_NAME}-{self.track_name}"

    def _best_name(self):
        return f"{BEST_NAME}-{self.track_name}"

    def _try_resume(self):
        """Resume the current track's autosave if it exists and matches."""
        name = self._autosave_name()
        if not os.path.exists(os.path.join(CKPT_DIR, name + ".pt")):
            return
        try:
            self.load_checkpoint(name)
            print(f"[trainer] resumed '{name}' @ step {self.global_step}")
        except Exception as e:
            print(f"[trainer] could not resume {name} ({e}); starting fresh")

    def _persist_after_update(self):
        with self.lock:
            mean_ret = float(np.mean(self.ep_returns)) if self.ep_returns else 0.0
            enough = len(self.ep_returns) >= 5
            self.history.append({
                "step": int(self.global_step),
                "updates": int(self.updates),
                "meanReturn": round(mean_ret, 3),
                "bestReturn": round(self.best_return, 3)
                if self.best_return > float("-inf") else 0.0,
            })
        self._save_history()

        # auto-save the best model for this track (by mean episode return)
        if enough and mean_ret > self.best_mean + 1e-3:
            self.best_mean = mean_ret
            try:
                self.save_checkpoint(self._best_name())
            except Exception as e:
                print(f"[trainer] best-save failed: {e}")

        # periodic autosave so a restart can resume this track
        if self.updates % 5 == 0:
            try:
                self.save_checkpoint(self._autosave_name())
            except Exception as e:
                print(f"[trainer] autosave failed: {e}")

    def get_history(self):
        with self.lock:
            return list(self.history)

    # ------------------------------------------------------------- checkpoints
    @staticmethod
    def _ckpt_path(name):
        """Map a checkpoint name to a safe path INSIDE CKPT_DIR.

        Returns (safe_name, abs_path) or (None, None) if the name is unusable.
        The name is reduced to a bare basename (alnum, dash, underscore, space) —
        this alone removes any '/', '\\' or '.' so path traversal is impossible;
        a containment check is kept as defence in depth.
        """
        safe = "".join(c for c in str(name or "") if c.isalnum() or c in ("-", "_", " ")).strip()
        if not safe:
            return None, None
        root = os.path.abspath(CKPT_DIR)
        path = os.path.abspath(os.path.join(root, safe + ".pt"))
        if os.path.commonpath([root, path]) != root:
            return None, None
        return safe, path

    def save_checkpoint(self, name):
        safe, path = self._ckpt_path(name)
        if safe is None:
            safe, path = self._ckpt_path("checkpoint")
        # Snapshot a consistent CPU copy of the weights, then write to disk
        # WITHOUT holding the lock (torch.save is slow and would otherwise
        # block the SSE stream and every API request for the write duration).
        with self.lock:
            mean_ret = float(np.mean(self.ep_returns)) if self.ep_returns else 0.0
            payload = {
                "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "updates": self.updates,
                "best_return": self.best_return,
                "mean_return": mean_ret,
                "created": time.time(),
                "track": self.track_name,
                "obs_dim": CarEnv.OBS_DIM,
                "window": self.window,
            }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic publish
        return self.checkpoint_info(safe)

    def checkpoint_info(self, name):
        safe, path = self._ckpt_path(name)
        if safe is None or not os.path.exists(path):
            return None
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # full unpickle executes arbitrary code — only safe for your OWN
            # trusted checkpoints, never a .pt from an untrusted source.
            ck = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "name": safe,
            "globalStep": int(ck.get("global_step", 0)),
            "updates": int(ck.get("updates", 0)),
            "meanReturn": round(float(ck.get("mean_return", 0.0)), 3),
            "bestReturn": round(float(ck.get("best_return", 0.0)), 3),
            "created": ck.get("created", os.path.getmtime(path)),
            "sizeKB": round(os.path.getsize(path) / 1024, 1),
        }

    def list_checkpoints(self):
        out = []
        for p in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime, reverse=True):
            name = os.path.splitext(os.path.basename(p))[0]
            info = self.checkpoint_info(name)
            if info:
                out.append(info)
        return out

    def load_checkpoint(self, name):
        safe, path = self._ckpt_path(name)
        if safe is None or not os.path.exists(path):
            raise FileNotFoundError(name)
        try:
            ck = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            ck = torch.load(path, map_location=self.device, weights_only=False)  # trusted local ckpt only
        # architecture guard: refuse clearly-incompatible checkpoints up front
        ck_obs, ck_win = ck.get("obs_dim"), ck.get("window")
        if ck_obs is not None and ck_obs != CarEnv.OBS_DIM:
            raise ValueError(f"obs_dim {ck_obs} != {CarEnv.OBS_DIM}")
        if ck_win is not None and ck_win != self.window:
            raise ValueError(f"window {ck_win} != {self.window}")
        with self.lock:
            self.model.load_state_dict(ck["model"])
            try:
                self.optimizer.load_state_dict(ck["optimizer"])
            except Exception as e:
                print(f"[trainer] optimizer state not restored: {e}")
            self.global_step = int(ck.get("global_step", 0))
            self.updates = int(ck.get("updates", 0))
            self.best_return = float(ck.get("best_return", float("-inf")))
        return self.checkpoint_info(name)

    def delete_checkpoint(self, name):
        safe, path = self._ckpt_path(name)
        if safe is not None and os.path.exists(path):
            os.remove(path)
            return True
        return False
