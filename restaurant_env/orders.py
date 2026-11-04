"""
Order stream: generate, save and load CSV.

When orders arrive and what they contain does not depend on the agents' actions, so
the stream can be generated ahead of time from a seed and saved as a CSV. Every
algorithm then faces exactly the same orders, which makes comparisons fair.

CSV format (one order per row):
    order_id,arrival_t,deadline_t,dishes
    0,3,98,salad+burger
`dishes` are joined with "+"; the same dish may appear more than once.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Order:
    oid: int
    arrival_t: int
    deadline_t: int
    dishes: list
    remaining: Counter = field(default=None)   # dishes not yet delivered
    bill: float = 0.0                           # total price of dishes delivered so far

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = Counter(self.dishes)

    def copy(self) -> "Order":
        return Order(self.oid, self.arrival_t, self.deadline_t, list(self.dishes))


def arrival_rate_at(order_cfg: dict, t: float) -> float:
    """Order arrival rate (orders/step) at step t; arrival_schedule is piecewise-linear."""
    sched = order_cfg.get("arrival_schedule")
    if not sched:
        return float(order_cfg["arrival_rate"])
    xs = [p[0] for p in sched]
    ys = [p[1] for p in sched]
    return float(np.interp(t, xs, ys))


def generate_orders(order_cfg: dict, menu: list, episode_length: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    probs = np.asarray(order_cfg["dishes_per_order_probs"], dtype=float)
    weights = order_cfg.get("dish_weights") or {}
    w = np.array([weights.get(d, 1.0) for d in menu], dtype=float)
    w = w / w.sum()
    orders = []
    for t in range(episode_length):
        for _ in range(rng.poisson(arrival_rate_at(order_cfg, t))):
            k = int(rng.choice(len(probs), p=probs)) + 1
            dishes = [str(d) for d in rng.choice(menu, size=k, p=w)]
            patience = int(rng.integers(order_cfg["patience_min"], order_cfg["patience_max"] + 1))
            orders.append(Order(len(orders), t, t + patience, dishes))
    return orders


def save_orders_csv(orders: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "arrival_t", "deadline_t", "dishes"])
        for o in orders:
            w.writerow([o.oid, o.arrival_t, o.deadline_t, "+".join(o.dishes)])


def load_orders_csv(path: str) -> list:
    orders = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orders.append(Order(int(row["order_id"]), int(row["arrival_t"]),
                                int(row["deadline_t"]), row["dishes"].split("+")))
    orders.sort(key=lambda o: o.arrival_t)
    return orders
