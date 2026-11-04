"""
Core simulation of the cooperative kitchen (independent of the Gym / PettingZoo API).

Two agents (agent 0 = chef, agent 1 = assistant by default) share one kitchen and work
together to fill orders:
take ingredients -> chop / fry / boil / bake -> plate -> hand in at the pass.

Rules vs. data
--------------
- data/recipes.json        recipes: ingredients, processes, cooker recipes, dishes
- data/layouts/*.json      kitchen layout: stations, coordinates, start positions, zones
- configs/env/*.yaml       everything else: menu, skills, plates, order stream, tips, reward
- order CSV (optional)     a pre-generated order stream
- this file                only advances the state: (state, action) -> new state

Actions
-------
Each agent picks one action per step: 0 = wait; s+1 = walk to station s and interact with it.
What "interact" means depends on what the agent is holding and on the station's state
(the same idea as Overcooked's interact).
An action takes (travel time + interaction time) steps; meanwhile the agent is locked and
can only choose 0.

- An agent holds at most one thing: one food item, or one plate (a plate can hold several items).
- A station can be used by only one agent at a time: while the other agent is walking to it
  or working at it, you cannot choose it. (After an action finishes, the agent stays where
  it is but no longer occupies that station.)
- On arrival the interaction is checked again (e.g. the item on a counter may be gone);
  if it no longer applies, the action "fails".

Order of updates in each step
-----------------------------
1. Agents 0 and 1 choose new actions in turn (agent 1 already sees what agent 0 just claimed)
2. Each agent's action advances one step (travel / arrive / work / finish)
3. Cookers tick (done, burnt)
4. Dirty plates return to the sink
5. Late orders expire, new orders arrive
6. Time +1, reward is computed
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

from .layout import Layout
from .orders import arrival_rate_at, generate_orders, load_orders_csv
from .recipes import RecipeBook
from .rewards import compute_reward

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cooker states
C_EMPTY, C_FILLING, C_COOKING, C_DONE, C_BURNT = 0, 1, 2, 3, 4
COOKER_STATE_NAMES = ["empty", "filling", "cooking", "done", "burnt"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _default_agents():
    # skills: number of steps each job takes; null means the agent cannot do it
    return [
        {"name": "chef", "move_speed": 2.0, "skills": {"chop": 3, "fry": 2, "wash": 3}},
        {"name": "assistant", "move_speed": 2.0, "skills": {"chop": 2, "fry": 4, "wash": 2}},
    ]


def _default_orders():
    return {
        "orders_file": None,          # path to an order CSV; null = generate from the seed
        "arrival_rate": 0.03,         # constant rate (orders/step), used when arrival_schedule is null
        "arrival_schedule": None,     # [[step, rate], ...], piecewise-linear interpolation
        "dishes_per_order_probs": [0.7, 0.3],  # probability of 1 / 2 / ... dishes per order
        "dish_weights": None,         # {dish: weight}; null = uniform
        "patience_min": 100,          # steps from arrival until the order expires
        "patience_max": 140,
    }


def _default_reward():
    return {
        "revenue_scale": 0.1,         # (price + tip) x scale
        "expire_penalty": 1.0,        # per expired order
        "burn_penalty": 0.5,          # per burnt cooker
        "waste_penalty": 0.1,         # per food item thrown away
        "invalid_action_penalty": 0.0,  # chose an action that is not currently allowed
        "failed_action_penalty": 0.0,   # arrived but the interaction no longer applies
        "step_penalty": 0.0,
        "shaping": {
            "enabled": True,
            "chop": 0.05,             # per item chopped
            "fry": 0.05,              # per item fried
            "cooker_start": 0.1,      # a cooker has all its inputs and starts cooking
            "cooker_collect": 0.1,    # finished soup / pizza / etc. moved onto a plate
            "plate_add": 0.05,        # per item added to a plate (not counting cooker_collect)
            "dish_delivered": 0.3,    # per dish handed in
        },
    }


@dataclass
class KitchenConfig:
    episode_length: int = 300
    recipes_file: str = "data/recipes.json"
    layout_file: str = "data/layouts/open.json"
    menu: list = field(default_factory=lambda: ["salad", "tomato_soup", "burger"])
    agents: list = field(default_factory=_default_agents)

    # Plates
    n_plates: int = 6
    initial_dirty_plates: int = 0
    dirty_return_delay: int = 5       # steps after a dish is served until its plate reaches the sink

    # Cookers
    burn_after: int = 8               # steps a finished cooker can wait before it burns
    cooker_clean_time: int = 2        # steps needed to clean a burnt cooker

    # Orders and tips
    orders: dict = field(default_factory=_default_orders)
    max_visible_orders: int = 5       # how many of the earliest open orders appear in the observation
    tip_max_rate: float = 0.20
    tip_grace_wait: float = 30.0      # no tip reduction if the order is finished within this many steps
    tip_zero_wait: float = 100.0      # tip drops to 0 at this many steps

    reward: dict = field(default_factory=_default_reward)

    @classmethod
    def from_yaml(cls, path: str) -> "KitchenConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "KitchenConfig":
        return cls(**data)

    def __post_init__(self):
        # Deep-merge orders / reward with the defaults, so a YAML file only lists what it changes
        orders = _default_orders()
        orders.update(self.orders or {})
        self.orders = orders
        reward = _default_reward()
        user = dict(self.reward or {})
        reward["shaping"].update(user.pop("shaping", {}) or {})
        reward.update(user)
        self.reward = reward
        assert len(self.agents) == 2, "only two agents are supported"
        assert 0 <= self.initial_dirty_plates <= self.n_plates
        assert self.tip_grace_wait < self.tip_zero_wait
        o = self.orders
        assert abs(sum(o["dishes_per_order_probs"]) - 1.0) < 1e-6
        assert o["patience_min"] <= o["patience_max"]


def resolve_path(p: Optional[str]) -> Optional[str]:
    """Look up a relative path in the current directory first, then in the project root."""
    if p is None or os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(PROJECT_ROOT, p)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------
@dataclass
class Item:
    """One thing in a hand or on a counter: a food item or a plate."""
    kind: str                      # "food" or "plate"
    comp: Optional[str] = None     # food state, e.g. "tomato:chopped" (when kind == food)
    contents: list = field(default_factory=list)  # items on the plate (when kind == plate)
    placed_by: int = -1            # last agent that put it on a counter (used to count handoffs)


@dataclass
class Cooker:
    tool: str                      # "pot" or "oven"
    contents: list = field(default_factory=list)
    status: int = C_EMPTY
    remaining: int = 0             # cooking steps left
    total: int = 0
    done_for: int = 0              # steps since it finished cooking
    output: Optional[str] = None


@dataclass
class Task:
    target: int                    # target station
    travel: int                    # travel steps left
    kind: str                      # interaction type
    work: int                      # interaction steps left
    arrived: bool = False


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------
class KitchenCore:
    def __init__(self, config: KitchenConfig):
        self.cfg = config
        self.book = RecipeBook.from_json(resolve_path(config.recipes_file), config.menu)
        self.layout = Layout.from_json(resolve_path(config.layout_file))
        self.S = self.layout.S
        self.n_actions = self.S + 1
        self.agent_names = [a["name"] for a in config.agents]
        sched = config.orders.get("arrival_schedule")
        self.max_rate = max([r for _, r in sched] if sched else [config.orders["arrival_rate"]])
        self.max_rate = max(self.max_rate, 1e-8)
        self._fixed_orders = None
        if config.orders.get("orders_file"):
            self._fixed_orders = load_orders_csv(resolve_path(config.orders["orders_file"]))

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, orders: Optional[list] = None) -> dict:
        """`orders` can be passed directly (for tests or replays); otherwise the CSV is used
        or orders are generated from the seed."""
        cfg, L = self.cfg, self.layout
        self.t = 0
        self.pos = list(L.agent_starts)
        self.holding: list = [None, None]
        self.task: list = [None, None]
        self.counter = {i: None for i, k in enumerate(L.kinds) if k == "counter"}
        self.cookers = {i: Cooker(k) for i, k in enumerate(L.kinds) if k in ("pot", "oven")}
        self.clean_plates = cfg.n_plates - cfg.initial_dirty_plates  # clean plates on the rack
        self.sink_dirty = cfg.initial_dirty_plates                    # dirty plates in the sink
        self.returning: list = []                                     # dirty plates on their way back (arrival times)

        if orders is not None:
            src = orders
        elif self._fixed_orders is not None:
            src = self._fixed_orders
        else:
            src = generate_orders(cfg.orders, cfg.menu, cfg.episode_length, seed or 0)
        self.future_orders = sorted((o.copy() for o in src), key=lambda o: o.arrival_t)
        self.open_orders: list = []

        self.stats = self._new_stats()
        self.step_log: list = []
        self._ev = self._new_events()
        self._release_orders(self._ev)
        return self.get_state()

    def step(self, actions):
        """actions = (agent 0 action, agent 1 action).
        Returns (state, reward, terminated, truncated, info)."""
        ev = self._new_events()
        self._ev = ev
        self.step_log = []
        acts = [int(actions[0]), int(actions[1])]

        for a in (0, 1):
            self._assign(a, acts[a], ev)
        for a in (0, 1):
            self._advance(a, ev)
        self._tick_cookers(ev)
        self._tick_plates()
        self._expire_orders(ev)
        self.t += 1
        self._release_orders(ev)
        self._update_step_stats()

        reward = compute_reward(ev, self.cfg.reward)
        truncated = self.t >= self.cfg.episode_length
        info = {"events": ev, "log": self.step_log, **self.summary()}
        return self.get_state(), reward, False, truncated, info

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def action_mask(self, a: int):
        m = [False] * self.n_actions
        m[0] = True
        if self.task[a] is not None:      # busy: can only continue
            return m
        for s in range(self.S):
            m[s + 1] = not self._claimed_by_other(a, s) and self._plan(a, s) is not None
        return m

    def _claimed_by_other(self, a: int, s: int) -> bool:
        task = self.task[1 - a]
        return task is not None and task.target == s

    # ------------------------------------------------------------------
    # Interaction rules: for (agent, station), what can be done now and how many steps it takes
    # ------------------------------------------------------------------
    def _skill(self, a: int, name: str):
        return self.cfg.agents[a]["skills"].get(name)

    def _plan(self, a: int, s: int):
        """Returns (interaction type, interaction steps), or None if no interaction applies."""
        L, book, h = self.layout, self.book, self.holding[a]
        if not L.allowed[a][s]:
            return None
        kind = L.kinds[s]

        if kind == "dispenser":
            comp = f"{L.stations[s]['ingredient']}:raw"
            if not book.can_dispense(L.stations[s]["ingredient"]):
                return None                # not used by this menu
            if h is None:
                return ("take", 1)
            if h.kind == "plate" and book.plate_valid(h.contents + [comp]):
                return ("add_to_plate", 1)

        elif kind == "plate_rack":
            if h is None and self.clean_plates > 0:
                return ("take_plate", 1)

        elif kind == "counter":
            it = self.counter[s]
            if h is None and it is not None:
                return ("pick", 1)
            if h is not None and it is None:
                return ("place", 1)
            if h is not None and it is not None:
                if h.kind == "plate" and it.kind == "food" and book.plate_valid(h.contents + [it.comp]):
                    return ("plate_from_counter", 1)
                if h.kind == "food" and it.kind == "plate" and book.plate_valid(it.contents + [h.comp]):
                    return ("food_onto_plate", 1)

        elif kind in ("cutting_board", "pan"):
            if h is not None and h.kind == "food":
                pname, out = book.process_output(kind, h.comp)
                t = self._skill(a, pname) if pname else None
                if out is not None and t is not None:
                    return (pname, int(t))

        elif kind in ("pot", "oven"):
            c = self.cookers[s]
            if c.status in (C_EMPTY, C_FILLING):
                if h is not None and h.kind == "food" and book.cooker_can_add(kind, c.contents, h.comp):
                    return ("cooker_add", 1)
            elif c.status == C_DONE:
                if h is not None and h.kind == "plate" and book.plate_valid(h.contents + [c.output]):
                    return ("cooker_collect", 1)
            elif c.status == C_BURNT:
                if h is None:
                    return ("cooker_clean", self.cfg.cooker_clean_time)

        elif kind == "sink":
            t = self._skill(a, "wash")
            if h is None and self.sink_dirty > 0 and t is not None:
                return ("wash", int(t))

        elif kind == "trash":
            if h is not None and (h.kind == "food" or h.contents):
                return ("trash", 1)

        elif kind == "pass":
            if h is not None and h.kind == "plate":
                dish = book.dish_of(h.contents)
                if dish is not None and self._order_for(dish) is not None:
                    return ("deliver", 1)
        return None

    # ------------------------------------------------------------------
    # Advancing actions
    # ------------------------------------------------------------------
    def _assign(self, a: int, act: int, ev: dict):
        if self.task[a] is not None or act == 0:
            return
        if not (0 < act < self.n_actions) or not self.action_mask(a)[act]:
            if 0 < act < self.n_actions and self._claimed_by_other(a, act - 1) \
                    and self._plan(a, act - 1) is not None:
                # Both agents picked the same station in the same step: agent 0 goes first,
                # agent 1's action is dropped for this step (not counted as invalid)
                ev["conflict"] += 1
                self.stats["conflicts"] += 1
                self._log(a, "conflict", act - 1, "")
                return
            ev["invalid"] += 1
            self.stats["invalid"][a] += 1
            self._log(a, "invalid", None, f"action {act}")
            return
        s = act - 1
        kind, work = self._plan(a, s)
        travel = self.layout.travel_time(self.pos[a], s, self.cfg.agents[a]["move_speed"])
        self.task[a] = Task(s, travel, kind, work)

    def _advance(self, a: int, ev: dict):
        task = self.task[a]
        if task is None:
            self.stats["idle_steps"][a] += 1
            return
        if not task.arrived:
            if task.travel > 0:            # walking
                self.stats["move_steps"][a] += 1
                task.travel -= 1
                if task.travel > 0:
                    return
                self._arrive(a, task)
                return                     # no interaction on the arrival step
            self._arrive(a, task)          # already at this station
            if self.task[a] is None:
                return
        # interacting
        self.stats["work_steps"][a] += 1
        task.work -= 1
        if task.work > 0:
            return
        plan = self._plan(a, task.target)
        if plan is None or plan[0] != task.kind:
            self._fail(a, task)
            return
        self._execute(a, task.target, task.kind, ev)
        self.task[a] = None

    def _arrive(self, a: int, task: Task):
        self.pos[a] = task.target
        task.arrived = True
        plan = self._plan(a, task.target)
        if plan is None or plan[0] != task.kind:
            self._fail(a, task)

    def _fail(self, a: int, task: Task):
        self.stats["failed"][a] += 1
        self._ev["failed"] += 1
        self._log(a, "failed", task.target, task.kind)
        self.task[a] = None

    # ------------------------------------------------------------------
    # Applying interactions
    # ------------------------------------------------------------------
    def _execute(self, a: int, s: int, kind: str, ev: dict):
        book, h, st = self.book, self.holding[a], self.stats
        detail = ""
        if kind == "take":
            comp = f"{self.layout.stations[s]['ingredient']}:raw"
            self.holding[a] = Item("food", comp)
            detail = book.comp_name(comp)
        elif kind == "add_to_plate":
            comp = f"{self.layout.stations[s]['ingredient']}:raw"
            h.contents.append(comp)
            ev["plate_add"] += 1
            detail = book.comp_name(comp)
        elif kind == "take_plate":
            self.clean_plates -= 1
            self.holding[a] = Item("plate")
        elif kind == "pick":
            it = self.counter[s]
            self.counter[s] = None
            self.holding[a] = it
            self._count_handoff(a, it, ev)
            detail = self._item_name(it)
        elif kind == "place":
            h.placed_by = a
            self.counter[s] = h
            self.holding[a] = None
            detail = self._item_name(h)
        elif kind == "plate_from_counter":
            it = self.counter[s]
            self.counter[s] = None
            h.contents.append(it.comp)
            self._count_handoff(a, it, ev)
            ev["plate_add"] += 1
            detail = book.comp_name(it.comp)
        elif kind == "food_onto_plate":
            it = self.counter[s]
            it.contents.append(h.comp)
            it.placed_by = a
            self.holding[a] = None
            ev["plate_add"] += 1
            detail = book.comp_name(h.comp)
        elif kind in ("chop", "fry"):
            _, out = book.process_output(self.layout.kinds[s], h.comp)
            detail = f"{book.comp_name(h.comp)} -> {book.comp_name(out)}"
            h.comp = out
            ev[kind] += 1
        elif kind == "cooker_add":
            c = self.cookers[s]
            c.contents.append(h.comp)
            detail = book.comp_name(h.comp)
            self.holding[a] = None
            r = book.cooker_match(c.tool, c.contents)
            if r is not None:
                c.status, c.remaining, c.total, c.output = C_COOKING, r["time"], r["time"], r["output"]
                ev["cooker_start"] += 1
            else:
                c.status = C_FILLING
        elif kind == "cooker_collect":
            c = self.cookers[s]
            h.contents.append(c.output)
            detail = book.comp_name(c.output)
            self.cookers[s] = Cooker(c.tool)
            ev["cooker_collect"] += 1
        elif kind == "cooker_clean":
            self.cookers[s] = Cooker(self.cookers[s].tool)
        elif kind == "wash":
            self.sink_dirty -= 1
            self.clean_plates += 1
            ev["plate_washed"] += 1
        elif kind == "trash":
            n = 1 if h.kind == "food" else len(h.contents)
            detail = self._item_name(h)
            if h.kind == "food":
                self.holding[a] = None
            else:
                h.contents = []
            ev["wasted"] += n
            st["wasted"] += n
        elif kind == "deliver":
            detail = self._deliver(a, ev)
        st["actions"][a][kind] = st["actions"][a].get(kind, 0) + 1
        self._log(a, kind, s, detail)

    def _count_handoff(self, a: int, it: Item, ev: dict):
        if it.placed_by not in (-1, a):
            ev["handoff"] += 1
            self.stats["handoffs"] += 1

    def _order_for(self, dish: str):
        """Among open orders that still need this dish, the one with the earliest deadline."""
        cands = [o for o in self.open_orders if o.remaining[dish] > 0]
        return min(cands, key=lambda o: o.deadline_t) if cands else None

    def _deliver(self, a: int, ev: dict):
        dish = self.book.dish_of(self.holding[a].contents)
        order = self._order_for(dish)
        price = self.book.price(dish)
        order.remaining[dish] -= 1
        order.bill += price
        self.holding[a] = None
        self.returning.append(self.t + self.cfg.dirty_return_delay)
        ev["bill"] += price
        ev["dish_delivered"] += 1
        st = self.stats
        st["bill_total"] += price
        st["dishes_delivered"] += 1
        detail = f"{self.book.dish_name(dish)} -> order {order.oid}"
        if sum(order.remaining.values()) == 0:
            wait = self.t + 1 - order.arrival_t
            rate = self.tip_rate(wait)
            tip = order.bill * rate
            ev["tip"] += tip
            ev["orders_completed"] += 1
            st["tip_total"] += tip
            st["orders_completed"] += 1
            st["sum_tip_rate"] += rate
            st["sum_completion_time"] += wait
            self.open_orders.remove(order)
            self._log(-1, "order_completed", None,
                      f"order {order.oid} took {wait} steps, tip {tip:.1f}")
        return detail

    def tip_rate(self, wait: float) -> float:
        cfg = self.cfg
        if wait <= cfg.tip_grace_wait:
            return cfg.tip_max_rate
        frac = (wait - cfg.tip_grace_wait) / (cfg.tip_zero_wait - cfg.tip_grace_wait)
        return cfg.tip_max_rate * max(0.0, 1.0 - frac)

    # ------------------------------------------------------------------
    # Time-driven updates
    # ------------------------------------------------------------------
    def _tick_cookers(self, ev: dict):
        for s, c in self.cookers.items():
            if c.status == C_COOKING:
                c.remaining -= 1
                if c.remaining <= 0:
                    c.status, c.done_for = C_DONE, 0
                    self._log(-1, "cooker_done", s, self.book.comp_name(c.output))
            elif c.status == C_DONE:
                c.done_for += 1
                if c.done_for > self.cfg.burn_after:
                    c.status = C_BURNT
                    ev["burnt"] += 1
                    self.stats["burnt"] += 1
                    self._log(-1, "burnt", s, self.book.comp_name(c.output))

    def _tick_plates(self):
        back = [t for t in self.returning if t <= self.t]
        self.returning = [t for t in self.returning if t > self.t]
        self.sink_dirty += len(back)

    def _expire_orders(self, ev: dict):
        kept = []
        for o in self.open_orders:
            if self.t + 1 >= o.deadline_t:
                ev["orders_expired"] += 1
                self.stats["orders_expired"] += 1
                self.stats["bill_lost"] += sum(self.book.price(d) * n for d, n in o.remaining.items())
                self._log(-1, "order_expired", None, f"order {o.oid}")
            else:
                kept.append(o)
        self.open_orders = kept

    def _release_orders(self, ev: dict):
        while self.future_orders and self.future_orders[0].arrival_t <= self.t:
            o = self.future_orders.pop(0)
            self.open_orders.append(o)
            self.stats["orders_arrived"] += 1
            self._log(-1, "order_arrived", None,
                      f"order {o.oid}: " + " + ".join(self.book.dish_name(d) for d in o.dishes))

    def _update_step_stats(self):
        st = self.stats
        if self.clean_plates == 0:
            st["no_clean_plate_steps"] += 1
        st["sum_open_orders"] += len(self.open_orders)
        # processed food sitting on counters, waiting to be used (prepping ahead)
        st["sum_prepped_on_counter"] += sum(
            1 for it in self.counter.values()
            if it is not None and it.kind == "food" and not it.comp.endswith(":raw"))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _new_events() -> dict:
        ev = {k: 0 for k in [
            "orders_completed", "orders_expired", "dish_delivered", "burnt", "wasted",
            "invalid", "failed", "chop", "fry", "cooker_start", "cooker_collect",
            "plate_add", "plate_washed", "handoff", "conflict",
        ]}
        ev["bill"] = 0.0
        ev["tip"] = 0.0
        return ev

    @staticmethod
    def _new_stats() -> dict:
        return {
            "bill_total": 0.0, "tip_total": 0.0, "bill_lost": 0.0,
            "orders_arrived": 0, "orders_completed": 0, "orders_expired": 0,
            "dishes_delivered": 0, "sum_tip_rate": 0.0, "sum_completion_time": 0.0,
            "burnt": 0, "wasted": 0, "handoffs": 0, "conflicts": 0,
            "invalid": [0, 0], "failed": [0, 0],
            "idle_steps": [0, 0], "move_steps": [0, 0], "work_steps": [0, 0],
            "actions": [{}, {}],              # count of each interaction per agent (division of labor)
            "no_clean_plate_steps": 0, "sum_open_orders": 0, "sum_prepped_on_counter": 0,
        }

    def summary(self) -> dict:
        s, n = self.stats, max(self.t, 1)
        done = s["orders_completed"]
        out = {
            "t": self.t,
            "revenue": s["bill_total"] + s["tip_total"],
            "bill_total": s["bill_total"],
            "tip_total": s["tip_total"],
            "avg_tip_rate": s["sum_tip_rate"] / done if done else 0.0,
            "orders_arrived": s["orders_arrived"],
            "orders_completed": done,
            "orders_expired": s["orders_expired"],
            "avg_completion_time": s["sum_completion_time"] / done if done else 0.0,
            "dishes_delivered": s["dishes_delivered"],
            "burnt": s["burnt"],
            "wasted": s["wasted"],
            "handoffs": s["handoffs"],
            "conflicts": s["conflicts"],
            "no_clean_plate_steps": s["no_clean_plate_steps"],
            "avg_open_orders": s["sum_open_orders"] / n,
            "avg_prepped_on_counter": s["sum_prepped_on_counter"] / n,
        }
        for a, name in enumerate(("a0", "a1")):
            out[f"{name}_idle_steps"] = s["idle_steps"][a]
            out[f"{name}_move_steps"] = s["move_steps"][a]
            out[f"{name}_work_steps"] = s["work_steps"][a]
            out[f"{name}_invalid"] = s["invalid"][a]
            out[f"{name}_failed"] = s["failed"][a]
            out[f"{name}_actions"] = dict(s["actions"][a])
        return out

    def plates_total(self) -> int:
        """Total number of plates (for the conservation check):
        rack + sink + returning + in hands + on counters."""
        n = self.clean_plates + self.sink_dirty + len(self.returning)
        n += sum(1 for h in self.holding if h is not None and h.kind == "plate")
        n += sum(1 for it in self.counter.values() if it is not None and it.kind == "plate")
        return n

    # ------------------------------------------------------------------
    # State and display
    # ------------------------------------------------------------------
    def _log(self, a: int, kind: str, s: Optional[int], detail: str):
        self.step_log.append({
            "t": self.t,
            "agent": self.agent_names[a] if a >= 0 else "",
            "event": kind,
            "station": self.layout.names[s] if s is not None else "",
            "detail": detail,
        })

    def _item_name(self, it: Optional[Item]) -> str:
        if it is None:
            return ""
        if it.kind == "food":
            return self.book.comp_name(it.comp)
        if not it.contents:
            return "empty plate"
        return "plate[" + ", ".join(self.book.comp_name(c) for c in it.contents) + "]"

    def get_state(self) -> dict:
        """Raw state dict (each interface turns it into its own observation vector)."""
        return {
            "t": self.t,
            "arrival_rate": arrival_rate_at(self.cfg.orders, self.t),
            "pos": list(self.pos),
            "holding": list(self.holding),
            "task": list(self.task),
            "counter": self.counter,
            "cookers": self.cookers,
            "clean_plates": self.clean_plates,
            "sink_dirty": self.sink_dirty,
            "returning": len(self.returning),
            "open_orders": sorted(self.open_orders, key=lambda o: o.deadline_t),
        }

    def snapshot(self) -> dict:
        """JSON-serializable snapshot of the state, for logs and replays."""
        L = self.layout
        agents = []
        for a in range(2):
            task = self.task[a]
            agents.append({
                "name": self.agent_names[a],
                "station": L.names[self.pos[a]],
                "holding": self._item_name(self.holding[a]),
                "task": None if task is None else {
                    "target": L.names[task.target], "kind": task.kind,
                    "travel": task.travel, "work": task.work},
            })
        return {
            "t": self.t,
            "agents": agents,
            "counters": {L.names[s]: self._item_name(it) for s, it in self.counter.items()},
            "cookers": {L.names[s]: {
                "status": COOKER_STATE_NAMES[c.status],
                "contents": [self.book.comp_name(x) for x in c.contents],
                "remaining": c.remaining, "done_for": c.done_for}
                for s, c in self.cookers.items()},
            "clean_plates": self.clean_plates,
            "sink_dirty": self.sink_dirty,
            "returning_plates": len(self.returning),
            "open_orders": [{
                "id": o.oid, "deadline_t": o.deadline_t,
                "remaining": {self.book.dish_name(d): n for d, n in o.remaining.items() if n > 0}}
                for o in sorted(self.open_orders, key=lambda o: o.deadline_t)],
        }

    def render_text(self) -> str:
        snap = self.snapshot()
        lines = [f"t={snap['t']:3d}  clean plates={snap['clean_plates']} "
                 f"dirty={snap['sink_dirty']} returning={snap['returning_plates']}"]
        for ag in snap["agents"]:
            task = ag["task"]
            tdesc = "idle" if task is None else f"{task['kind']}@{task['target']}"
            lines.append(f"  {ag['name']}: at {ag['station']}, holding [{ag['holding'] or 'nothing'}], {tdesc}")
        for name, it in snap["counters"].items():
            if it:
                lines.append(f"  {name}: {it}")
        for name, c in snap["cookers"].items():
            if c["status"] != "empty":
                extra = f" ({c['remaining']} left)" if c["status"] == "cooking" else ""
                lines.append(f"  {name}: {c['status']} {', '.join(c['contents'])}{extra}")
        for o in snap["open_orders"]:
            dishes = " ".join(f"{k} x{v}" for k, v in o["remaining"].items())
            lines.append(f"  order {o['id']} due {o['deadline_t']}: {dishes}")
        return "\n".join(lines)
