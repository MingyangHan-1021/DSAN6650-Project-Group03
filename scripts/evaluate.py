"""Shared, deterministic evaluation and CSV output for centralized agents."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from restaurant_env import CentralizedKitchenEnv


METRICS = (
    "revenue", "bill_total", "tip_total", "orders_arrived", "orders_completed",
    "orders_expired", "dishes_delivered", "avg_completion_time", "avg_tip_rate",
    "burnt", "wasted", "handoffs", "conflicts", "a0_invalid", "a1_invalid",
)


def evaluate(
    policy,
    config_path: str,
    algorithm: str,
    train_seed: int,
    eval_seeds: list[int],
    output_dir: str | Path = "results/csv",
) -> list[dict]:
    """Evaluate a callable ``policy(obs, action_masks) -> [a0, a1]``.

    One row per evaluation seed is saved to ``{algorithm}_{config}.csv``. Rows
    from other training seeds are retained; rerunning a seed replaces its rows.
    """
    config_name = Path(config_path).stem
    path = Path(output_dir) / f"{algorithm}_{config_name}.csv"
    env = CentralizedKitchenEnv(config_path=config_path)
    rows = []
    try:
        for eval_seed in eval_seeds:
            obs, _ = env.reset(seed=int(eval_seed))
            episode_reward = 0.0
            done = False
            while not done:
                action = np.asarray(policy(obs, env.action_masks()), dtype=np.int64)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            row = {
                "algorithm": algorithm,
                "config": config_name,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "total_reward": episode_reward,
            }
            row.update({name: info[name] for name in METRICS})
            rows.append(row)
    finally:
        env.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            old = [r for r in csv.DictReader(handle) if int(r["train_seed"]) != train_seed]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(old + rows)
    return rows
