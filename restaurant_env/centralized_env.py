"""
Centralized Gymnasium interface: a single agent controls both the chef and the assistant.

- Observation: the full kitchen state as one vector with values in [0, 1]
- Action: MultiDiscrete([S+1, S+1]) where S is the number of stations;
  0 = wait, s+1 = go to station s and interact
- Reward: the shared team reward (see rewards.py)

Provides action_masks() for use with sb3-contrib's MaskablePPO.
The encode_* functions can be reused by the multi-agent version to build local observations.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .orders import arrival_rate_at
from .core import C_BURNT, C_COOKING, C_DONE, C_FILLING, KitchenConfig, KitchenCore

ITEM_SCALE = 3.0   # max copies of one component on a plate, used for normalization


def encode_item(core: KitchenCore, it) -> list:
    """One thing: [is food, is plate] + count of each component (C+2 values)."""
    v = [0.0] * (len(core.book.vocab) + 2)
    if it is None:
        return v
    idx = core.book.vocab_index
    if it.kind == "food":
        v[0] = 1.0
        v[2 + idx[it.comp]] = 1.0 / ITEM_SCALE
    else:
        v[1] = 1.0
        for c in it.contents:
            v[2 + idx[c]] += 1.0 / ITEM_SCALE
    return v


def encode_agent(core: KitchenCore, a: int) -> list:
    """One agent: position one-hot (S) + target one-hot (S) + [busy, walking, steps left] + item in hand."""
    S = core.S
    pos = [0.0] * S
    pos[core.pos[a]] = 1.0
    tgt = [0.0] * S
    task = core.task[a]
    if task is not None:
        tgt[task.target] = 1.0
    status = [
        float(task is not None),
        float(task is not None and not task.arrived),
        0.0 if task is None else (task.travel + task.work) / 10.0,
    ]
    return pos + tgt + status + encode_item(core, core.holding[a])


def encode_station(core: KitchenCore, s: int) -> list:
    """A counter or cooker: its contents (C+2) + cooker status one-hot (4) + progress (C+7 values in total)."""
    kind = core.layout.kinds[s]
    if kind == "counter":
        return encode_item(core, core.counter[s]) + [0.0] * 5
    c = core.cookers[s]
    v = [0.0] * (len(core.book.vocab) + 2)
    for x in c.contents:
        v[2 + core.book.vocab_index[x]] += 1.0 / ITEM_SCALE
    st = [float(c.status == k) for k in (C_FILLING, C_COOKING, C_DONE, C_BURNT)]
    if c.status == C_COOKING:
        prog = 1.0 - c.remaining / max(c.total, 1)
    elif c.status == C_DONE:
        prog = c.done_for / max(core.cfg.burn_after, 1)       # closer to 1 = about to burn
    else:
        prog = 0.0
    return v + st + [prog]


def encode_orders(core: KitchenCore) -> list:
    """The K open orders with the earliest deadlines: [remaining count per dish (D), time left, time waited]."""
    cfg, menu = core.cfg, core.book.menu
    K, pmax = cfg.max_visible_orders, cfg.orders["patience_max"]
    orders = sorted(core.open_orders, key=lambda o: o.deadline_t)[:K]
    v = []
    for j in range(K):
        if j < len(orders):
            o = orders[j]
            v += [o.remaining[d] / ITEM_SCALE for d in menu]
            v += [(o.deadline_t - core.t) / pmax, (core.t - o.arrival_t) / cfg.tip_zero_wait]
        else:
            v += [0.0] * (len(menu) + 2)
    v.append(len(core.open_orders) / K)
    return v


def encode_state(core: KitchenCore) -> np.ndarray:
    cfg = core.cfg
    feats = []
    for a in (0, 1):
        feats += encode_agent(core, a)
    for s in core.layout.stateful:
        feats += encode_station(core, s)
    feats += [
        core.clean_plates / cfg.n_plates,
        core.sink_dirty / cfg.n_plates,
        len(core.returning) / cfg.n_plates,
    ]
    feats += encode_orders(core)
    feats += [core.t / cfg.episode_length, arrival_rate_at(cfg.orders, core.t) / core.max_rate]
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, 1.0)


class CentralizedKitchenEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, config: Optional[KitchenConfig] = None,
                 config_path: Optional[str] = None, render_mode: Optional[str] = None):
        super().__init__()
        if config is None:
            config = KitchenConfig.from_yaml(config_path) if config_path else KitchenConfig()
        self.core = KitchenCore(config)
        self.render_mode = render_mode
        n = self.core.n_actions
        self.action_space = spaces.MultiDiscrete([n, n])
        self.core.reset(seed=0)
        dim = encode_state(self.core).shape[0]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is None:   # derive a seed from gym's np_random so runs are reproducible after env.reset(seed)
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self.core.reset(seed=seed)
        return encode_state(self.core), self.core.summary()

    def step(self, action):
        _, reward, terminated, truncated, info = self.core.step((int(action[0]), int(action[1])))
        if self.render_mode == "human":
            print(self.render())
        return encode_state(self.core), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """As MaskablePPO expects: the masks of each MultiDiscrete dimension, concatenated."""
        return np.array(self.core.action_mask(0) + self.core.action_mask(1), dtype=bool)

    def render(self):
        return self.core.render_text()
