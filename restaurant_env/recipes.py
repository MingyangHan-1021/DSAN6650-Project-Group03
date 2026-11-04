"""
Recipes: loads data/recipes.json and answers "can these things be combined?" questions.

Terms
-----
- component: written as "ingredient:state", e.g. "tomato:raw", "tomato:chopped"
- process: turns one state into another at an active station, e.g. chop at the
  cutting board, fry at the pan. The agent has to stay at the station while processing.
- cooker: pot or oven. Put the inputs in; once a recipe is complete it cooks on its own,
  and burns if left too long after it is done. The agent can walk away meanwhile.
- dish: the set of components on a plate; it must match exactly to be served.
"""

from __future__ import annotations

import json
from collections import Counter


def _subset(a: Counter, b: Counter) -> bool:
    """Whether multiset a is contained in b (no item appears more often than in b)."""
    return all(b[k] >= v for k, v in a.items())


class RecipeBook:
    def __init__(self, data: dict, menu: list):
        self.ingredients = data["ingredients"]
        self.state_names = data.get("state_names", {})
        self.products = set(data.get("products", []))   # cooker outputs shown by name only

        # Processes: station kind -> (process name, {input: output})
        self.processes = {}
        for pname, p in data["processes"].items():
            self.processes[p["station"]] = (pname, p["map"])

        # Menu: only dishes on this episode's menu appear in orders
        unknown = [m for m in menu if m not in data["dishes"]]
        assert not unknown, f"menu contains dishes not in recipes.json: {unknown}"
        self.menu = list(menu)
        self.dishes = {k: data["dishes"][k] for k in menu}
        self.dish_counters = {k: Counter(v["components"]) for k, v in self.dishes.items()}

        # Cooker recipes: keep only those whose output the menu needs
        # (e.g. without pasta on the menu, the pot does not accept noodles)
        needed = set().union(*[set(c) for c in self.dish_counters.values()])
        self.cookers = [c for c in data["cookers"] if c["output"] in needed]
        for c in self.cookers:
            c["_inputs"] = Counter(c["inputs"])

        # Useful components: work backwards from what the menu needs (pre-process states,
        # cooker inputs). Dispensers only offer useful raw ingredients, and the cutting
        # board / pan only perform useful processes.
        useful = set(needed)
        changed = True
        while changed:
            changed = False
            for _, m in self.processes.values():
                for src, out in m.items():
                    if out in useful and src not in useful:
                        useful.add(src); changed = True
            for c in self.cookers:
                if c["output"] in useful:
                    for x in c["inputs"]:
                        if x not in useful:
                            useful.add(x); changed = True
        self.useful = useful

        # Every component that can ever appear, in a fixed order, for observation encoding
        vocab = set(needed)
        for ing in self.ingredients:
            vocab.add(f"{ing}:raw")
        for _, m in self.processes.values():
            vocab.update(m.keys()); vocab.update(m.values())
        for c in data["cookers"]:
            vocab.update(c["inputs"]); vocab.add(c["output"])
        self.vocab = sorted(vocab)
        self.vocab_index = {c: i for i, c in enumerate(self.vocab)}

    @classmethod
    def from_json(cls, path: str, menu: list) -> "RecipeBook":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), menu)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def process_output(self, station_kind: str, comp: str):
        """Result of processing `comp` at this kind of station; (None, None) if not possible."""
        if station_kind not in self.processes:
            return None, None
        pname, m = self.processes[station_kind]
        out = m.get(comp)
        return (pname, out) if out is not None and out in self.useful else (None, None)

    def can_dispense(self, ingredient: str) -> bool:
        return f"{ingredient}:raw" in self.useful

    def plate_valid(self, contents: list) -> bool:
        """Whether the plate's contents are part of some dish on the menu
        (plating is only allowed in the "right direction")."""
        c = Counter(contents)
        return any(_subset(c, d) for d in self.dish_counters.values())

    def dish_of(self, contents: list):
        """Which dish the plate contents form exactly; None if not a complete dish."""
        c = Counter(contents)
        for k, d in self.dish_counters.items():
            if c == d:
                return k
        return None

    def cooker_can_add(self, tool: str, contents: list, comp: str) -> bool:
        new = Counter(contents + [comp])
        return any(r["tool"] == tool and _subset(new, r["_inputs"]) for r in self.cookers)

    def cooker_match(self, tool: str, contents: list):
        """The cooker recipe whose inputs match the cooker contents exactly."""
        c = Counter(contents)
        for r in self.cookers:
            if r["tool"] == tool and c == r["_inputs"]:
                return r
        return None

    def price(self, dish: str) -> float:
        return float(self.dishes[dish]["price"])

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def comp_name(self, comp: str) -> str:
        ing, state = comp.split(":")
        name = self.ingredients.get(ing, ing)
        if ing in self.products:
            return name
        return f"{self.state_names.get(state, state)} {name}".strip()

    def dish_name(self, dish: str) -> str:
        return self.dishes[dish]["name"]
