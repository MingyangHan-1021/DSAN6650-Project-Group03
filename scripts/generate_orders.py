"""
Generate order CSV files from a config.

Usage (from the project root):
    python scripts/generate_orders.py --config configs/env/medium.yaml --seeds 0 1 2 --out data/orders/medium

Writes one orders_seed{seed}.csv per seed. Point orders.orders_file in a config at one
of these files and the environment will always use that order stream.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restaurant_env import KitchenConfig, generate_orders, save_orders_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = KitchenConfig.from_yaml(args.config)
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        orders = generate_orders(cfg.orders, cfg.menu, cfg.episode_length, seed)
        path = os.path.join(args.out, f"orders_seed{seed}.csv")
        save_orders_csv(orders, path)
        n_dishes = sum(len(o.dishes) for o in orders)
        print(f"{path}: {len(orders)} orders, {n_dishes} dishes")


if __name__ == "__main__":
    main()
