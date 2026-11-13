"""
How much can this kitchen actually serve?

This is a *lower bound* on the work a dish needs: the station visits of an ideal
recipe plan, with travel time taken from the layout, the best of the two agents'
skills, no walking to a station twice, no contention, and the cooking of a pot or an
oven overlapping with everything else. A real policy can only be slower, so

    steps of wall clock per dish  >=  (agent-steps per dish) / (number of agents)

and with the episode length that gives the kitchen's capacity in dishes. Comparing it
with the demand of a difficulty level is what "is this difficulty right?" means:

    capacity  >> demand   the level is relaxed
    capacity  ~  demand   the level is only completable by a near-optimal policy
    capacity  <  demand   the level is overloaded on purpose

Usage (from the project root)
-----------------------------
    python scripts/capacity.py --config configs/env/easy.yaml
    python scripts/capacity.py --config configs/env/medium.yaml configs/env/hard.yaml --seeds 0-19
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restaurant_env import KitchenConfig, KitchenCore  # noqa: E402
from restaurant_env.orders import generate_orders  # noqa: E402

WORK = {"take": 1, "take_plate": 1, "plate_from_counter": 1, "add_to_plate": 1,
        "cooker_add": 1, "cooker_collect": 1, "deliver": 1, "place": 1}


class Bound:
    """Ideal single-agent chains over the real layout."""

    def __init__(self, cfg: KitchenConfig):
        self.cfg = cfg
        self.core = KitchenCore(cfg)
        self.L = self.core.layout
        self.book = self.core.book
        self.start = self.L.id["counter_c"]
        # fastest hands in the team for each active process
        self.skill = {}
        for station_kind, (pname, _mapping) in self.book.processes.items():
            best = [a["skills"].get(pname) for a in cfg.agents if a["skills"].get(pname)]
            self.skill[station_kind] = int(min(best)) if best else 1

    def travel(self, i, j):
        return self.L.travel_time(i, j, max(a["move_speed"] for a in self.cfg.agents))

    def nearest(self, stations, frm):
        return min(stations, key=lambda s: self.travel(frm, s))

    def _bin(self, ingredient):
        for s, kind in enumerate(self.L.kinds):
            if kind == "dispenser" and self.L.stations[s].get("ingredient") == ingredient:
                return s
        raise KeyError(ingredient)

    def _stations(self, kind):
        return [s for s, k in enumerate(self.L.kinds) if k == kind]

    def component_chain(self, comp):
        """[(interaction, station)] that produces `comp` starting from empty hands."""
        procs, cur = [], comp
        while True:
            hit = None
            for station_kind, (_pname, mapping) in self.book.processes.items():
                for src, out in mapping.items():
                    if out == cur:
                        hit = (station_kind, src)
            if hit is None:
                break
            procs.append(hit)
            cur = hit[1]
        steps = [("take", self._bin(cur.split(":")[0]))]
        for station_kind, _src in reversed(procs):
            steps.append((station_kind, self._stations(station_kind)[0]))
        return steps

    def _cooker_for(self, comp):
        for r in self.book.cookers:
            if r["output"] == comp:
                return r
        return None

    def dish(self, dish: str) -> int:
        comps = self.book.dish_counters[dish]
        pos, total, staged = self.start, 0, []

        def do(interaction, station):
            nonlocal pos, total
            work = self.skill.get(interaction, WORK.get(interaction, 1))
            total += self.travel(pos, station) + work
            pos = station

        for comp, n in comps.items():
            for _ in range(n):
                recipe = self._cooker_for(comp)
                if recipe is not None:
                    for inp, m in recipe["_inputs"].items():
                        for _ in range(m):
                            for interaction, station in self.component_chain(inp):
                                do(interaction, station)
                            do("cooker_add", self._stations(recipe["tool"])[0])
                elif comp.endswith(":raw") and self.book.can_dispense(comp.split(":")[0]) \
                        and self.book.plate_valid([comp]):
                    continue                     # straight from the bin while plating
                else:
                    for interaction, station in self.component_chain(comp):
                        do(interaction, station)
                    counter = self.nearest(self._stations("counter"), pos)
                    do("place", counter)
                    staged.append(counter)

        do("take_plate", self.L.id["plate_rack"])
        for comp in comps:
            recipe = self._cooker_for(comp)
            if recipe is not None:
                do("cooker_collect", self._stations(recipe["tool"])[0])
            elif comp.endswith(":raw") and self.book.can_dispense(comp.split(":")[0]) \
                    and self.book.plate_valid([comp]):
                do("add_to_plate", self._bin(comp.split(":")[0]))
            else:
                do("plate_from_counter", staged.pop(0))
        do("deliver", self.L.id["pass"])
        return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", default=["configs/env/medium.yaml"])
    p.add_argument("--seeds", default="0-19", help="seeds used to average the demand")
    args = p.parse_args()

    lo, hi = args.seeds.split("-") if "-" in args.seeds else (args.seeds, args.seeds)
    seeds = list(range(int(lo), int(hi) + 1))

    for cfg_path in args.config:
        cfg = KitchenConfig.from_yaml(cfg_path)
        bound = Bound(cfg)
        n_agents = len(cfg.agents)
        print(f"\n{os.path.basename(cfg_path)}  ({n_agents} agents, {cfg.episode_length} steps)")
        print(f"  {'dish':<16}{'agent-steps':>12}{'wall steps':>12}")
        per_dish = {}
        for dish in cfg.menu:
            steps = bound.dish(dish)
            per_dish[dish] = steps / n_agents
            print(f"  {bound.book.dish_name(dish):<16}{steps:>12}{math.ceil(steps / n_agents):>12}")

        mean_steps = statistics.fmean(per_dish.values())
        capacity = cfg.episode_length / mean_steps
        demand = statistics.fmean(
            sum(len(o.dishes) for o in generate_orders(cfg.orders, cfg.menu, cfg.episode_length, s))
            for s in seeds)
        print(f"  mean per dish     {mean_steps * n_agents:>9.1f} agent-steps -> "
              f"{mean_steps:.1f} wall steps")
        print(f"  capacity          {capacity:>9.1f} dishes in {cfg.episode_length} steps")
        print(f"  demand            {demand:>9.1f} dishes (seeds {seeds[0]}-{seeds[-1]})")
        print(f"  demand/capacity   {demand / capacity:>9.0%}  "
              f"({'relaxed' if demand / capacity < 0.7 else 'at the limit' if demand / capacity <= 1.05 else 'overloaded'})")


if __name__ == "__main__":
    main()
