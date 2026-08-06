"""Transformer actor-critic policy.

The policy receives a *sequence* of the last K observations (a temporal window)
and attends over them with a small Transformer encoder. The last token's
representation feeds two heads:

  * actor  -> 3 action means (steering, acceleration, brake)
  * critic -> scalar state value

Actions are sampled from a diagonal Gaussian with a learnable, state-independent
log-std. Raw actions are clipped when applied to the environment (standard
clipped-action PPO), so no tanh-squash Jacobian correction is needed.
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerActorCritic(nn.Module):
    def __init__(self, obs_dim=10, act_dim=3, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, window=8):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.window = window

        self.embed = nn.Linear(obs_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=window)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        # state-independent log std, initialized fairly exploratory
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def _features(self, obs_seq):
        # obs_seq: (B, K, obs_dim)
        h = self.embed(obs_seq)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        return h[:, -1, :]  # last token summarizes the window

    def forward(self, obs_seq):
        feat = self._features(obs_seq)
        mean = self.actor_mean(feat)
        value = self.critic(feat).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value

    @torch.no_grad()
    def act(self, obs_seq):
        mean, std, value = self.forward(obs_seq)
        dist = Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate(self, obs_seq, action):
        mean, std, value = self.forward(obs_seq)
        dist = Normal(mean, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value
