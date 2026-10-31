"""
中心化版本的 Gymnasium 接口：一个智能体同时控制厨师和服务员。

- 观测：整个餐厅的完整状态，拼成一个 0~1 之间的向量
- 动作：MultiDiscrete([厨师动作数, 服务员动作数])
- 奖励：团队奖励（见 rewards.py）

可直接用于 Stable-Baselines3 的 PPO；
提供 action_masks()，也可以配合 sb3-contrib 的 MaskablePPO 使用。

观测维度与菜单长度无关（每桌只编码菜的数量和总做菜时间，不编码具体菜名），
所以扩充或修改菜单不会改变网络输入大小。
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import N_TABLE_STATES, RestaurantConfig, RestaurantCore


def encode_table(core: RestaurantCore, tb: dict) -> list:
    """单张桌子的特征（17 维）。多智能体版本可以复用。"""
    cfg, P = core.cfg, core.max_party
    s = [0.0] * N_TABLE_STATES
    s[tb["state"]] = 1.0
    n = max(tb["n_items"], 1)
    return s + [
        tb["capacity"] / P,                                  # 桌子座位数
        tb["party_size"] / P,                                # 这桌几个人
        float(tb["phase"]),                                  # 是否在甜点阶段
        tb["patience_frac"],                                 # 剩余耐心比例
        tb["eat_remaining"] / cfg.eat_time_max,              # 用餐剩余时间
        tb["n_pending"] / P,                                 # 待做的菜
        tb["n_cooking"] / P,                                 # 在灶台上的菜
        tb["n_ready"] / P,                                   # 在出餐口的菜
        tb["n_delivered"] / n,                               # 已上菜比例
        tb["pending_cook_time"] / (P * core.max_cook_time),  # 待做菜的总做菜时间
        tb["max_pass_wait"] / (2 * max(cfg.cold_after, 1)),  # 出餐口上放得最久的菜（≥0.5 表示已凉）
        tb["wait_so_far"] / cfg.tip_zero_wait,               # 已等待时间（影响小费）
    ]


def encode_state(core: RestaurantCore, state: dict) -> np.ndarray:
    cfg = core.cfg
    feats = []

    # 每张桌子：17 维
    for tb in state["tables"]:
        feats += encode_table(core, tb)

    # 门口排队：每个位置 [是否有人, 人数, 耐心比例]，共 3 × max_queue 维
    for j in range(cfg.max_queue):
        if j < len(state["queue"]):
            g = state["queue"][j]
            feats += [1.0, g["size"] / core.max_party, g["patience_frac"]]
        else:
            feats += [0.0, 0.0, 0.0]

    # 厨房资源：4 维
    feats += [
        state["clean_plates"] / cfg.n_plates,
        state["dirty_plates"] / cfg.n_plates,
        state["pass_count"] / cfg.pass_capacity,
        len(state["stoves"]) / cfg.n_stoves,
    ]

    # 灶台：每个灶台的剩余时间，按剩余时间排序，共 n_stoves 维
    rem = sorted(s["remaining"] for s in state["stoves"])
    rem += [0] * (cfg.n_stoves - len(rem))
    feats += [r / core.max_cook_time for r in rem]

    # 厨师 / 服务员：3 维
    max_srv = max(cfg.seat_time, cfg.order_time, cfg.deliver_time, cfg.clear_time)
    feats += [
        state["chef_busy"] / max(cfg.prep_time, cfg.wash_time),
        float(state["chef_washing"]),
        state["server_busy"] / max_srv,
    ]

    # 时间与客流：2 维
    feats += [state["t"] / cfg.episode_length, state["arrival_rate"] / core.max_rate]

    return np.asarray(feats, dtype=np.float32)


class CentralizedRestaurantEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, config: Optional[RestaurantConfig] = None,
                 config_path: Optional[str] = None, render_mode: Optional[str] = None):
        super().__init__()
        if config is None:
            config = RestaurantConfig.from_yaml(config_path) if config_path else RestaurantConfig()
        self.core = RestaurantCore(config)
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete(
            [self.core.n_chef_actions, self.core.n_server_actions]
        )
        obs_dim = encode_state(self.core, self.core.reset(seed=0)).shape[0]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is None:  # 由 gym 的 np_random 派生种子，保证 env.reset(seed) 后可复现
            seed = int(self.np_random.integers(0, 2**31 - 1))
        state = self.core.reset(seed=seed)
        return self._obs(state), self.core.summary()

    def step(self, action):
        chef_a, server_a = int(action[0]), int(action[1])
        state, reward, terminated, truncated, info = self.core.step(chef_a, server_a)
        if self.render_mode == "human":
            print(self.render())
        return self._obs(state), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """MaskablePPO 要求：MultiDiscrete 的各维 mask 依次拼接。"""
        return np.concatenate([self.core.chef_mask(), self.core.server_mask()])

    def render(self):
        return self.core.render_text()

    def _obs(self, state: dict) -> np.ndarray:
        return np.clip(encode_state(self.core, state), 0.0, 1.0)
