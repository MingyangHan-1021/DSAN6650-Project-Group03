"""
Kitchen layout: loads data/layouts/*.json.

- Every station has a name, a kind and a 2D position (pos)
- Travel time between two stations = ceil(Manhattan distance / move speed)
- If zones is null, both agents can use every station; otherwise zones[i] lists the
  stations agent i is allowed to use

Station kinds
-------------
dispenser      ingredient bin: take one raw ingredient with empty hands; with a plate in
               hand, the ingredient can go straight onto the plate
counter        holds one thing (food or a plate); used for prepping ahead and handoffs
cutting_board  active process: chop
pan            active process: fry
pot / oven     passive cookers: cook on their own once the inputs are complete; burn if left too long
plate_rack     take one clean plate with empty hands
sink           wash one dirty plate with empty hands; the clean plate goes back to the rack
trash          throw away the food in hand, or empty the plate in hand
pass           hand in a finished dish for an order
"""

from __future__ import annotations

import json
import math

KINDS = ["dispenser", "counter", "cutting_board", "pan", "pot", "oven",
         "plate_rack", "sink", "trash", "pass"]


class Layout:
    def __init__(self, data: dict):
        self.name = data.get("name", "")
        self.stations = data["stations"]
        self.S = len(self.stations)
        self.names = [s["name"] for s in self.stations]
        self.kinds = [s["kind"] for s in self.stations]
        assert len(set(self.names)) == self.S, "station names must be unique"
        for k in self.kinds:
            assert k in KINDS, f"unknown station kind: {k}"
        self.id = {n: i for i, n in enumerate(self.names)}
        self.pos = [tuple(s["pos"]) for s in self.stations]

        self.agent_starts = [self.id[n] for n in data["agent_starts"]]
        assert self.agent_starts[0] != self.agent_starts[1], "the two agents must start at different stations"

        zones = data.get("zones")
        self.allowed = []
        for a in range(2):
            if zones is None:
                self.allowed.append([True] * self.S)
            else:
                z = set(zones[a])
                self.allowed.append([n in z for n in self.names])

        # Stations with state that goes into the observation: counters and cookers
        self.stateful = [i for i, k in enumerate(self.kinds) if k in ("counter", "pot", "oven")]

    @classmethod
    def from_json(cls, path: str) -> "Layout":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def distance(self, i: int, j: int) -> int:
        (x1, y1), (x2, y2) = self.pos[i], self.pos[j]
        return abs(x1 - x2) + abs(y1 - y2)

    def travel_time(self, i: int, j: int, speed: float) -> int:
        if i == j:
            return 0
        return max(1, math.ceil(self.distance(i, j) / speed))

    def first_of(self, kind: str) -> int:
        return self.kinds.index(kind)
