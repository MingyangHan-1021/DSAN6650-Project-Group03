"""
Run one episode with random legal actions and save the logs, to check the environment
and the log format.

Usage (from the project root):
    python scripts/run_random.py --config configs/env/easy.yaml --seed 0 --out results/logs/random_easy
Add --print to print the kitchen state at every step.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restaurant_env import EpisodeLogger, KitchenConfig, KitchenCore  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/env/easy.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/logs/random")
    p.add_argument("--print", action="store_true")
    args = p.parse_args()

    core = KitchenCore(KitchenConfig.from_yaml(args.config))
    core.reset(args.seed)
    logger = EpisodeLogger(core)
    rng = np.random.default_rng(args.seed)
    total, done = 0.0, False
    while not done:
        acts = [int(rng.choice(np.flatnonzero(core.action_mask(a)))) for a in (0, 1)]
        _, r, _, done, info = core.step(acts)
        logger.record(acts, r, info)
        total += r
        if args.print:
            print(core.render_text())
    logger.save(args.out)
    print(f"total reward {total:.2f} | orders completed {info['orders_completed']}/{info['orders_arrived']} "
          f"| expired {info['orders_expired']} | logs saved to {args.out}")


if __name__ == "__main__":
    main()
