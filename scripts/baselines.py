"""
Baseline policies for the cooperative kitchen.

Two policies:

- random: each agent picks a uniformly random legal action (lower bound).
- rule:   a hand-written scheduler that turns the recipes into task chains and lets
          the two agents pick tasks off a shared list (see below).

The rule policy is also the difficulty calibration tool: if it cannot complete a
reasonable number of orders on medium, the difficulty parameters are wrong and the
config has to be fixed before anyone starts training RL.

How the rule policy works
-------------------------
The environment has no "combine" action: an agent walks to a station and the
interaction follows from what it holds, so one action = "go there and do it",
travel included. A dish is therefore a short sequence of station visits, and the
policy plans those sequences (a *task*) and hands them out. A plate is filled in one
of two ways:

park mode - dishes that need no cooker (salad, burger), open kitchen only
  park   burger: [plate_rack -> free counter]
  prep   burger: [beef_bin -> pan -> that counter]  [lettuce_bin -> board -> that counter]
  serve  burger: [counter(complete plate) -> pass]
  One counter for the whole dish, and no "put it down, walk back, pick it up again".

hand mode - dishes whose plate has to collect from a pot or the oven (soup, pasta, pizza)
  prep     soup: [tomato_bin -> board -> counter] x2  (and into the pot from there)
  assemble soup: [plate_rack -> pot(collect) -> park]
  serve    soup: [counter(complete plate) -> pass]

Rules on top of the chains:

* Two plans at a time, for the dishes whose deadlines are closest. When one of them
  needs a cooker, the second one is chosen among the dishes that do not, so the two
  agents are not left watching the pot.
* In hand mode a plate is only taken when every component is already on a counter,
  straight in a bin (the burger bun), or finished in a cooker: taking one earlier
  ties up one of the six plates.
* Plating and serving are separate tasks, so the split kitchen works too: there the
  assistant assembles on a counter and the chef carries the plate to the pass.
* Cookers are allocated per plan (both tomatoes of a soup go into the same pot) and a
  plate is fetched while the pot is nearly done, because a finished cooker burns after
  `burn_after` steps and "notice it, walk to the rack, walk to the pot" takes longer.
* A task is only claimed by an agent that may use every station in it, and the second
  agent never picks the station the first one just chose (the environment drops that
  action, which would otherwise leave the loser stuck forever).
* Every action is checked against its expected post-condition before the task moves
  on (did the take really put something in the hand? did the component really end up
  on the parked plate?). If it failed, the chain restarts from the bin, or the item in
  hand is parked, so nobody deadlocks holding something half-way through a recipe.
* A task that waits for a counter or a cooker for more than WAIT_LIMIT steps is
  dropped and the whole plan is rebuilt.

Usage (from the project root)
-----------------------------
    # one episode, print the state every 10 steps
    python scripts/baselines.py --policy rule --config configs/env/medium.yaml --seeds 0 --trace 10

    # the calibration table: both baselines, 20 seeds x 3 difficulties
    python scripts/baselines.py --policy both --config configs/env/easy.yaml \
        configs/env/medium.yaml configs/env/hard.yaml --seeds 0-19
    # -> results/csv/baselines_random.csv, baselines_rule.csv, baselines_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restaurant_env import KitchenConfig, KitchenCore  # noqa: E402
from restaurant_env.core import C_BURNT, C_COOKING, C_DONE, C_EMPTY, C_FILLING  # noqa: E402

# ---------------------------------------------------------------------------
# Small recipe helpers
# ---------------------------------------------------------------------------
def _cooker_recipe_for(book, output):
    """The cooker recipe that produces `output`, or None."""
    for r in book.cookers:
        if r["output"] == output:
            return r
    return None


def _process_for(book, comp):
    """(station kind, source component) of the process that produces `comp`."""
    for kind, (_pname, mapping) in book.processes.items():
        for src, out in mapping.items():
            if out == comp:
                return kind, src
    return None, None


# post-conditions of a step: what the agent must hold once the action is done
EMPTY = ("empty",)
PLATE = ("plate",)
ANY = ("any",)


def holds(comp):
    return ("comp", comp)


def plate_has(comp):
    return ("plate_has", comp)


def parked_has(comp):
    """The component is on the plate we parked on a counter (nobody holds that plate)."""
    return ("parked_has", comp)


# ---------------------------------------------------------------------------
# Tasks and plans
# ---------------------------------------------------------------------------
class Task:
    """A chain of station visits.

    Each step is (kind, payload, expected interaction, post-condition).
    """

    __slots__ = ("steps", "i", "agent", "label", "waited", "fails", "blocked_by")

    def __init__(self, steps, label="", blocked_by=None):
        self.steps = steps
        self.i = 0
        self.agent = None
        self.label = label
        self.blocked_by = blocked_by or []
        self.waited = 0        # consecutive steps this task could not start
        self.fails = 0         # how often an action of this task was rejected

    @property
    def done(self):
        return self.i >= len(self.steps)

    @property
    def step(self):
        return self.steps[self.i]


class Plan:
    """Everything needed to serve one dish: prep tasks, the plate, and the serving."""

    def __init__(self, dish: str, deadline: int):
        self.dish = dish
        self.deadline = deadline
        self.prep: list[Task] = []
        self.assemble: Task | None = None
        self.serve: Task | None = None
        self.cooker_kind = None     # "pot" / "oven" / None
        self.cooker_recipe = None
        self.cooker = None          # station index once allocated
        self.mode = "hand"          # "park": plate on a counter, components go straight onto it
        self.plate_counter = None   # counter the plate is parked on (mode == "park")

    def tasks(self):
        return self.prep + [self.assemble, self.serve]

    def abandoned(self):
        return all(t.agent is None for t in self.tasks())


class RulePolicy:
    """Recipe-driven task scheduler (see the module docstring)."""

    MAX_PLANS = 2
    WAIT_LIMIT = 14         # steps a task may wait for a counter / cooker before replanning
    FAIL_LIMIT = 3          # rejected actions before the plan is abandoned

    def __init__(self, max_plans: int = 2, agents=(0, 1)):
        self.MAX_PLANS = max_plans
        self.agents = tuple(agents)
        self.plans: list[Plan] = []
        self.task_of = [(None, None), (None, None)]   # (task, plan) per agent
        self.pending = [None, None]                   # (task, plan, step index) issued

    # -- public ----------------------------------------------------------
    def act(self, core: KitchenCore):
        self._sync_plans(core)
        acts = [0, 0]
        taken = set()
        for a in (0, 1):
            if a not in self.agents:           # solo reference: one agent does everything
                continue
            if core.task[a] is not None:       # busy: the mask only allows 0 anyway
                continue
            if self.pending[a] is not None:    # the previous action just finished
                self._verify(core, a)
            acts[a] = self._act_agent(core, a, taken)
            if acts[a] > 0:
                taken.add(acts[a] - 1)
        return acts

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _needs(self, core):
        """dish -> copies still to deliver, plus the closest deadline of each dish."""
        book = core.book
        need, deadline = Counter(), {}
        for order in core.open_orders:
            for dish, n in order.remaining.items():
                need[dish] += n
                deadline[dish] = min(deadline.get(dish, 10 ** 9), order.deadline_t)
        # a plate that already holds a complete dish will be served: it covers one
        for it in core.holding:
            if it is not None and it.kind == "plate":
                dish = book.dish_of(it.contents)
                if dish is not None and need[dish] > 0:
                    need[dish] -= 1
        for it in core.counter.values():
            if it is not None and it.kind == "plate":
                dish = book.dish_of(it.contents)
                if dish is not None and need[dish] > 0:
                    need[dish] -= 1
        # a cooker that is already cooking (or done) covers the dish it produces
        for c in core.cookers.values():
            if c.status in (C_COOKING, C_DONE) and c.output:
                for dish, data in book.dishes.items():
                    if c.output in data["components"]:
                        if need[dish] > 0:
                            need[dish] -= 1
                        break
        return Counter({d: n for d, n in need.items() if n > 0}), deadline

    @staticmethod
    def _plate_on_counter(core, dish):
        for it in core.counter.values():
            if it is not None and it.kind == "plate" and core.book.dish_of(it.contents) == dish:
                return True
        return False

    def _sync_plans(self, core):
        need, deadline = self._needs(core)
        # a plan lives until served; also while an agent is inside it or while a
        # finished plate for it waits on a counter (otherwise nobody serves it)
        self.plans = [p for p in self.plans
                      if not p.serve.done
                      and (p.dish in need or self._plate_on_counter(core, p.dish)
                           or any(t.agent is not None for t in p.tasks()))]
        for p in self.plans:
            self._allocate_cooker(core, p)

        active = [p for p in self.plans if not p.serve.done]
        planned = {p.dish for p in active}
        # While a cooker dish is in flight the agents would sit and watch the pot.
        # Prefer a dish that needs no cooker as the second plan, so there is always
        # something to do during the cooking time.
        book = core.book
        cooker_in_flight = any(p.cooker_kind is not None for p in active)

        def needs_cooker(dish):
            return any(_cooker_recipe_for(book, c) is not None for c in book.dish_counters[dish])

        candidates = sorted(need, key=lambda d: ((cooker_in_flight and needs_cooker(d)),
                                                 deadline.get(d, 10 ** 9)))
        for dish in candidates:
            if len(active) >= self.MAX_PLANS:
                break
            if dish in planned:
                continue
            plan = self._build_plan(core, dish, deadline.get(dish, core.t + 300))
            self.plans.append(plan)
            active.append(plan)
            planned.add(dish)
        self.plans.sort(key=lambda p: p.deadline)

    def _allocate_cooker(self, core, plan) -> bool:
        if plan.cooker_kind is None or plan.cooker is not None:
            return True
        recipe = plan.cooker_recipe
        empty, resumable = [], []
        for s, kind in enumerate(core.layout.kinds):
            if kind != plan.cooker_kind:
                continue
            c = core.cookers[s]
            if c.status == C_EMPTY:
                empty.append(s)
            elif c.status == C_FILLING and recipe is not None:
                have = Counter(c.contents)
                if all(recipe["_inputs"][x] >= n for x, n in have.items()):
                    resumable.append((len(c.contents), s))
        if empty:
            plan.cooker = empty[0]
            return True
        if resumable:
            plan.cooker = max(resumable)[1]     # finish the pot somebody left half full
            return True
        return False

    def _build_plan(self, core, dish: str, deadline: int) -> Plan:
        book, L = core.book, core.layout
        plan = Plan(dish, deadline)
        comps = book.dish_counters[dish]
        has_cooker = any(_cooker_recipe_for(book, c) is not None for c in comps)
        staged = [c for c in comps if not self._bin_addable(book, c)
                  and _cooker_recipe_for(book, c) is None]

        open_layout = all(L.allowed[0]) and all(L.allowed[1])
        if open_layout and not has_cooker and staged:
            # No cooker: park a plate on a counter and carry every component straight
            # onto it. That uses one counter per dish instead of two and removes the
            # "put it down on a counter, walk back later and pick it up" round trip.
            plan.mode = "park"
            park = Task([("st", "plate_rack", "take_plate", PLATE),
                         ("park_plate", None, "place", EMPTY)], label=f"{dish}: park")
            plan.prep.append(park)
            for comp in comps:
                for _ in range(comps[comp]):
                    steps = self._prep_steps(book, L, comp,
                                             ("our_plate_add", comp, "food_onto_plate",
                                              parked_has(comp)))
                    plan.prep.append(Task(steps, label=f"{dish}: prep {comp}", blocked_by=[park]))
            plan.assemble = park
            plan.serve = Task([("dish_plate", dish, "pick", PLATE),
                               ("st", "pass", "deliver", EMPTY)],
                              label=f"{dish}: serve",
                              blocked_by=[t for t in plan.prep if t is not park])
            return plan

        # Cooker dish: the plate travels with an agent (it has to collect from the
        # pot / oven), so components are staged on counters first.
        for comp in comps:
            recipe = _cooker_recipe_for(book, comp)
            if recipe is None:
                continue
            plan.cooker_kind = recipe["tool"]
            plan.cooker_recipe = recipe
            for _ in range(comps[comp]):
                for inp, m in recipe["_inputs"].items():
                    for _ in range(m):
                        plan.prep.extend(self._prep_tasks(
                            book, L, inp, ("cooker_fill", None, "cooker_add", EMPTY),
                            f"{dish}: prep {inp}"))

        for comp in comps:
            if _cooker_recipe_for(book, comp) is not None:
                continue
            if self._bin_addable(book, comp):
                continue                     # straight from the bin onto the plate
            for _ in range(comps[comp]):
                plan.prep.extend(self._prep_tasks(
                    book, L, comp, ("free_counter", None, "place", EMPTY),
                    f"{dish}: prep {comp}"))

        steps = [("plate_any", None, ("take_plate", "pick"), PLATE)]
        for comp in comps:
            if _cooker_recipe_for(book, comp) is not None:
                steps.append(("cooker_with", comp, "cooker_collect", plate_has(comp)))
            elif self._bin_addable(book, comp):
                steps.append(("st", self._bin_name(L, comp.split(":")[0]), "add_to_plate", plate_has(comp)))
            else:
                steps.append(("counter_with", comp, "plate_from_counter", plate_has(comp)))
        steps.append(("free_counter", None, "place", EMPTY))   # park the finished plate
        plan.assemble = Task(steps, label=f"{dish}: assemble")
        plan.serve = Task([("dish_plate", dish, "pick", PLATE),
                           ("st", "pass", "deliver", EMPTY)], label=f"{dish}: serve")
        return plan

    def _step_allowed(self, L, a, step) -> bool:
        """Can agent `a` perform this step (ignoring whether it is legal right now)?"""
        kind, payload = step[0], step[1]
        if kind == "st":
            s = L.id.get(payload)
            return s is not None and L.allowed[a][s]
        if kind in ("free_counter", "counter_with", "dish_plate", "plate_any",
                    "park_plate", "our_plate_add"):
            return any(L.allowed[a][s] for s in self._counter_ids(L))
        if kind in ("cooker_fill", "cooker_with"):
            for s, k in enumerate(L.kinds):
                if k in ("pot", "oven") and L.allowed[a][s]:
                    return True
            return False
        return True

    @staticmethod
    def _counter_ids(L):
        return [s for s, k in enumerate(L.kinds) if k == "counter"]

    def _prep_tasks(self, book, L, comp, end, label):
        """Build the prep task(s) for one component.

        In the split kitchen a single agent cannot walk the whole chain (the assistant
        may use the bins and the board, the chef the pan and the cookers), so the chain
        is cut at the zone boundary and the food is handed over on a counter:

            assistant: [tomato_bin -> cutting_board -> counter]
            chef:      [counter -> pan -> counter]
        """
        steps = self._prep_steps(book, L, comp, end)
        runs, cur, owner = [], [], None
        for step in steps:
            allowed = [a for a in (0, 1) if self._step_allowed(L, a, step)]
            if not allowed:
                allowed = [owner if owner is not None else 0]
            new_owner = owner if owner in allowed else allowed[0]
            if owner is not None and new_owner != owner and cur:
                carried = cur[-1][3][1] if cur[-1][3][0] == "comp" else None
                if carried is not None:
                    cur.append(("free_counter", None, "place", EMPTY))
                    runs.append(cur)
                    cur = [("counter_with", carried, "pick", holds(carried))]
            owner = new_owner
            cur.append(step)
        if cur:
            runs.append(cur)
        tasks, prev = [], None
        for run in runs:
            task = Task(run, label=label, blocked_by=[prev] if prev is not None else None)
            tasks.append(task)
            prev = task
        return tasks

    def _prep_steps(self, book, L, comp: str, end) -> list:
        """Steps that turn an empty hand into `comp` delivered to `end`."""
        chain, cur = [], comp
        while True:
            kind, src = _process_for(book, cur)
            if kind is None:
                break
            chain.append((self._station_of_kind(L, kind), book.processes[kind][0], cur))
            cur = src
        steps = [("st", self._bin_name(L, cur.split(":")[0]), "take", holds(cur))]
        for station, pname, out in reversed(chain):
            steps.append(("st", station, pname, holds(out)))
        steps.append(end)
        return steps

    def _bin_addable(self, book, comp: str) -> bool:
        """A raw ingredient that can go straight onto the plate (e.g. the burger bun)."""
        if not comp.endswith(":raw"):
            return False
        return book.can_dispense(comp.split(":")[0]) and book.plate_valid([comp])

    @staticmethod
    def _bin_name(L, ingredient: str) -> str:
        for s, kind in enumerate(L.kinds):
            if kind == "dispenser" and L.stations[s].get("ingredient") == ingredient:
                return L.names[s]
        raise KeyError(f"no bin for {ingredient}")

    @staticmethod
    def _station_of_kind(L, kind: str) -> str:
        return L.names[L.first_of(kind)]

    # ------------------------------------------------------------------
    # Running a task
    # ------------------------------------------------------------------
    @staticmethod
    def _counters(core):
        return [s for s, k in enumerate(core.layout.kinds) if k == "counter"]

    def _resolve(self, core, a, plan, step):
        """Which station this step means right now, or None if it has to wait."""
        L = core.layout
        kind, payload = step[0], step[1]
        if kind == "st":
            s = L.id.get(payload)
            return s if s is not None and L.allowed[a][s] else None
        if kind == "free_counter":
            free = [s for s in self._counters(core)
                    if core.counter[s] is None and L.allowed[a][s]]
            if not free:
                return None
            # the closest free counter: a placement is a pure detour otherwise
            speed = core.cfg.agents[a]["move_speed"]
            return min(free, key=lambda s: L.travel_time(core.pos[a], s, speed))
        if kind == "counter_with":
            for s in self._counters(core):
                it = core.counter[s]
                if it is not None and it.kind == "food" and it.comp == payload:
                    return s
            return None
        if kind == "park_plate":
            for s in self._counters(core):
                if core.counter[s] is None and L.allowed[a][s]:
                    return s
            return None
        if kind == "our_plate_add":
            s = plan.plate_counter
            if s is None:
                return None
            it = core.counter[s]
            return s if it is not None and it.kind == "plate" else None
        if kind == "plate_any":
            for s in self._counters(core):
                it = core.counter[s]
                if it is not None and it.kind == "plate" and not it.contents and L.allowed[a][s]:
                    return s
            s = L.id.get("plate_rack")
            return s if s is not None and core.clean_plates > 0 else None
        if kind == "dish_plate":
            for s in self._counters(core):
                it = core.counter[s]
                if it is not None and it.kind == "plate" and core.book.dish_of(it.contents) == payload:
                    return s
            return None
        if kind == "cooker_with":
            for s, c in core.cookers.items():
                if c.status == C_DONE and c.output == payload and L.allowed[a][s]:
                    return s
            return None
        if kind == "cooker_fill":
            return plan.cooker
        return None

    def _step_action(self, core, a, plan, task, taken):
        """1-based action, None if the step is impossible, -1 if it has to wait."""
        s = self._resolve(core, a, plan, task.step)
        if s is None or s in taken:
            return -1
        if not core.action_mask(a)[s + 1]:
            return -1          # the other agent is walking to / working at this station
        now = core._plan(a, s)
        if now is None:
            return None
        expected = task.step[2]
        if expected is not None:
            allowed = expected if isinstance(expected, tuple) else (expected,)
            if now[0] not in allowed:
                return None
        if task.step[0] == "park_plate":
            plan.plate_counter = s
        return s + 1

    @staticmethod
    def _post_ok(core, a, post, plan=None) -> bool:
        if post[0] == "parked_has":
            if plan is None or plan.plate_counter is None:
                return False
            it = core.counter[plan.plate_counter]
            return it is not None and it.kind == "plate" and post[1] in it.contents
        h = core.holding[a]
        if post[0] == "empty":
            return h is None
        if post[0] == "plate":
            return h is not None and h.kind == "plate"
        if post[0] == "plate_has":
            return h is not None and h.kind == "plate" and post[1] in h.contents
        if post[0] == "comp":
            return h is not None and h.kind == "food" and h.comp == post[1]
        return True

    def _verify(self, core, a):
        """Did the action we issued actually do what the step promised?"""
        task, plan, i = self.pending[a]
        self.pending[a] = None
        if self._post_ok(core, a, task.steps[i][3], plan):
            task.i = i + 1
            task.waited = 0
            if task.done:
                task.agent = None
                self.task_of[a] = (None, None)
            return
        # the action failed: restart the chain if the hands are empty, otherwise
        # park the item and give the task back
        task.agent = None
        self.task_of[a] = (None, None)
        task.fails += 1
        if task.fails > self.FAIL_LIMIT:
            self._abandon(plan)
        elif core.holding[a] is None and i > 0:
            task.i = 0

    def _abandon(self, plan):
        """Give up on a plan (a burnt cooker, a lost plate) so a fresh one can start."""
        if plan is None:
            return
        for t in plan.tasks():
            t.i = len(t.steps)
            t.agent = None
        if plan in self.plans:
            self.plans.remove(plan)
        for a in (0, 1):
            task, _plan = self.task_of[a]
            if task is not None and _plan is plan:
                self.task_of[a] = (None, None)
                self.pending[a] = None

    def _act_agent(self, core, a, taken) -> int:
        task, plan = self.task_of[a]
        if task is not None and task.done:
            task.agent = None
            self.task_of[a] = (None, None)
            task, plan = None, None
        if task is not None:
            act = self._step_action(core, a, plan, task, taken)
            if act == -1:
                task.waited += 1
                if task.waited > self.WAIT_LIMIT:     # hopeless: rebuild the plan
                    self._abandon(plan)
                    return self._fallback(core, a)
                return 0
            if act is None:
                task.waited += 1
                if task.waited > self.WAIT_LIMIT:
                    self._abandon(plan)
                    return self._fallback(core, a)
                self._drop(core, a, task, plan)
                return self._fallback(core, a)
            task.waited = 0
            self.pending[a] = (task, plan, task.i)
            return act

        task, plan = self._pick_task(core, a)
        if task is None:
            return self._fallback(core, a)
        act = self._step_action(core, a, plan, task, taken)
        if act == -1:
            return 0
        if act is None:
            return self._fallback(core, a)
        task.agent = a
        task.waited = 0
        self.task_of[a] = (task, plan)
        self.pending[a] = (task, plan, task.i)
        return act

    def _drop(self, core, a, task, plan):
        task.agent = None
        self.task_of[a] = (None, None)
        self.pending[a] = None
        if core.holding[a] is None and task.i > 0:
            task.i = 0                          # redo it from the bin
        task.fails += 1
        if task.fails > self.FAIL_LIMIT:
            self._abandon(plan)

    # ------------------------------------------------------------------
    # Choosing what to do next
    # ------------------------------------------------------------------
    def _can_do(self, core, a, task, plan) -> bool:
        """Can this agent use every station this task visits?"""
        L = core.layout
        for kind, payload, _expected, _post in task.steps:
            if kind == "st":
                s = L.id.get(payload)
                if s is None or not L.allowed[a][s]:
                    return False
            elif kind == "cooker_fill":
                if plan is None or plan.cooker is None or not L.allowed[a][plan.cooker]:
                    return False
        return True

    def _claimable(self, core, a, plan, task) -> bool:
        if task is None or task.done or task.agent is not None:
            return False
        for dep in task.blocked_by:
            if dep.done:
                continue
            # In a park plan the plate only has to be on the counter by the time the
            # component arrives, so the preps may start as soon as somebody is parking
            # it instead of waiting for that agent to finish.
            if plan is not None and plan.mode == "park" and dep.agent is not None:
                continue
            return False
        return self._can_do(core, a, task, plan)

    def _pick_task(self, core, a):
        # 1. a finished cooker outranks everything (it burns), and an almost finished
        #    one is worth waiting next to with a plate in hand: walking to the rack and
        #    back after it is done usually takes longer than `burn_after`
        for plan in self.plans:
            if plan.cooker is None:
                continue
            c = core.cookers[plan.cooker]
            soon = c.status == C_COOKING and c.remaining <= 4
            if (c.status == C_DONE or soon) and core.holding[a] is None:
                if self._claimable(core, a, plan, plan.assemble) and core.clean_plates > 0:
                    return plan.assemble, plan
        # 2. a complete plate waiting on a counter should be served
        for plan in self.plans:
            if core.holding[a] is None and self._claimable(core, a, plan, plan.serve):
                if self._resolve(core, a, plan, plan.serve.step) is not None:
                    return plan.serve, plan
        # 3. a burnt cooker is useless until it is cleaned
        if core.holding[a] is None:
            for s, c in core.cookers.items():
                if c.status == C_BURNT and core.layout.allowed[a][s]:
                    return Task([("st", core.layout.names[s], "cooker_clean", EMPTY)],
                                label="clean"), None
        # 4. no clean plates and dirty ones waiting -> wash
        if core.clean_plates <= 1 and core.sink_dirty > 0 and core.holding[a] is None:
            s = core.layout.id["sink"]
            if core.layout.allowed[a][s]:
                return Task([("st", "sink", "wash", EMPTY)], label="wash"), None
        # 5. plate a dish whose components are all available
        for plan in self.plans:
            if core.holding[a] is None and core.clean_plates > 0 \
                    and self._claimable(core, a, plan, plan.assemble) and self._ready(core, plan.dish):
                return plan.assemble, plan
        # 6. prep work, earliest deadline first
        for plan in self.plans:
            self._skip_satisfied_preps(core, plan)
            for task in plan.prep:
                if self._claimable(core, a, plan, task):
                    return task, plan
        return None, None

    @staticmethod
    def _task_comp(task):
        """The component a prep task produces (the step before the drop-off)."""
        post = task.steps[-2][3] if len(task.steps) >= 2 else task.steps[-1][3]
        return post[1] if post[0] == "comp" else None

    def _skip_satisfied_preps(self, core, plan):
        """Mark prep tasks done when the component is already on a counter / in the cooker."""
        if plan.mode == "park":
            if plan.plate_counter is None:
                return
            it = core.counter[plan.plate_counter]
            if it is None or it.kind != "plate":
                return
            on_plate = Counter(it.contents)
            for task in plan.prep:
                if task.done or task.agent is not None or task is plan.assemble:
                    continue
                comp = self._task_comp(task)
                if comp is not None and on_plate[comp] > 0:
                    on_plate[comp] -= 1
                    task.i = len(task.steps)
            return
        on_counter = Counter()
        for s in self._counters(core):
            it = core.counter[s]
            if it is not None and it.kind == "food":
                on_counter[it.comp] += 1
        in_cooker = Counter(plan.cooker_recipe["_inputs"]) if False else Counter()
        if plan.cooker is not None:
            in_cooker = Counter(core.cookers[plan.cooker].contents)
        for task in plan.prep:
            if task.done:
                continue
            comp = self._task_comp(task)
            if comp is None:
                continue
            if on_counter[comp] > 0:
                on_counter[comp] -= 1
                task.i = len(task.steps)
                task.agent = None
            elif task.steps[-1][0] == "cooker_fill" and in_cooker[comp] > 0:
                in_cooker[comp] -= 1
                task.i = len(task.steps)
                task.agent = None

    def _ready(self, core, dish: str) -> bool:
        """Can the dish be plated right now with what is already available?"""
        book = core.book
        avail = Counter()
        for s in self._counters(core):
            it = core.counter[s]
            if it is not None and it.kind == "food":
                avail[it.comp] += 1
        for comp, k in book.dish_counters[dish].items():
            short = k - avail[comp]
            if short <= 0:
                continue
            if comp.endswith(":raw") and book.can_dispense(comp.split(":")[0]):
                continue
            done = sum(1 for c in core.cookers.values() if c.status == C_DONE and c.output == comp)
            if done >= short:
                continue
            return False
        return True

    # ------------------------------------------------------------------
    # Fallbacks: never stand still holding something
    # ------------------------------------------------------------------
    def _fallback(self, core, a) -> int:
        h = core.holding[a]
        L = core.layout
        if h is not None and h.kind == "plate":
            dish = core.book.dish_of(h.contents)
            if dish is not None and any(o.remaining[dish] > 0 for o in core.open_orders):
                if core._plan(a, L.id["pass"]) is not None:
                    return L.id["pass"] + 1
            for s in self._counters(core):
                if core.counter[s] is None and core._plan(a, s) is not None:
                    return s + 1
            return L.id["trash"] + 1 if core._plan(a, L.id["trash"]) is not None else 0
        if h is not None:
            for s in self._counters(core):
                if core.counter[s] is None and core._plan(a, s) is not None:
                    return s + 1
            return L.id["trash"] + 1 if core._plan(a, L.id["trash"]) is not None else 0
        if core.clean_plates <= 1 and core.sink_dirty > 0 and core._plan(a, L.id["sink"]) is not None:
            return L.id["sink"] + 1
        for s, c in core.cookers.items():
            if c.status == C_BURNT and core._plan(a, s) is not None:
                return s + 1
        return 0


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------
class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, core: KitchenCore):
        return [int(self.rng.choice(np.flatnonzero(core.action_mask(a)))) for a in (0, 1)]


# ---------------------------------------------------------------------------
# Running and reporting
# ---------------------------------------------------------------------------
def run_episode(core: KitchenCore, policy, seed: int, trace: int = 0):
    core.reset(seed)
    # total demand of this episode: how many dishes the orders ask for
    demand_dishes = sum(len(o.dishes) for o in core.future_orders) + \
        sum(len(o.dishes) for o in core.open_orders)
    total, step, done = 0.0, 0, False
    info = {}
    while not done:
        actions = policy.act(core)
        _, reward, _, done, info = core.step(actions)
        total += reward
        step += 1
        if trace and step % trace == 0:
            print(f"\n--- step {step} | reward {total:.1f} ---")
            print(core.render_text())
    info["demand_dishes"] = demand_dishes
    return total, info


def flat_info(info: dict) -> dict:
    out = {}
    for k, v in info.items():
        if k in ("events", "log"):
            continue
        out[k] = json.dumps(v) if isinstance(v, dict) else v
    return out


SUMMARY_METRICS = [
    ("orders_arrived", "orders that arrived"),
    ("orders_completed", "orders completed"),
    ("orders_expired", "orders expired"),
    ("demand_dishes", "dishes ordered"),
    ("dishes_delivered", "dishes served"),
    ("revenue", "revenue"),
    ("avg_tip_rate", "average tip rate"),
    ("avg_completion_time", "average completion time"),
    ("wasted", "food thrown away"),
    ("burnt", "burnt cookers"),
    ("handoffs", "handoffs"),
    ("conflicts", "conflicts"),
    ("no_clean_plate_steps", "steps without a clean plate"),
    ("a0_idle_steps", "agent 0 idle steps"),
    ("a1_idle_steps", "agent 1 idle steps"),
    ("total_reward", "episode reward"),
]


def mean_sd(rows, config: str, policy: str, metric: str):
    values = [float(r[metric]) for r in rows
              if r["config"] == config and r["policy"] == policy]
    if not values:
        return float("nan"), 0.0
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def summarize(rows, config_name: str, policy: str):
    n = sum(1 for r in rows if r["config"] == config_name and r["policy"] == policy)
    print(f"\n{config_name} / {policy}  ({n} seeds)")
    for metric, label in SUMMARY_METRICS:
        m, sd = mean_sd(rows, config_name, policy, metric)
        print(f"  {label:<30s} {m:8.2f}  +- {sd:6.2f}")


def comparison_table(rows, configs, policies):
    """One row per (config, policy) so the two baselines can be read side by side."""
    head = (f"{'config':<8s}{'policy':<8s}{'completed':>12s}{'arrived':>9s}"
            f"{'demand':>8s}{'served':>8s}{'revenue':>9s}{'tip':>7s}{'wasted':>8s}{'burnt':>7s}")
    print("\n" + head)
    print("-" * len(head))
    for cfg in configs:
        for pol in policies:
            m, sd = mean_sd(rows, cfg, pol, "orders_completed")
            if m != m:      # nan: policy was not run for this config
                continue
            arr, _ = mean_sd(rows, cfg, pol, "orders_arrived")
            dem, _ = mean_sd(rows, cfg, pol, "demand_dishes")
            dis, _ = mean_sd(rows, cfg, pol, "dishes_delivered")
            rev, rev_sd = mean_sd(rows, cfg, pol, "revenue")
            tip, _ = mean_sd(rows, cfg, pol, "avg_tip_rate")
            was, _ = mean_sd(rows, cfg, pol, "wasted")
            bur, _ = mean_sd(rows, cfg, pol, "burnt")
            print(f"{cfg:<8s}{pol:<8s}{m:>7.2f}+-{sd:<4.2f}{arr:>9.2f}{dem:>8.2f}{dis:>8.2f}"
                  f"{rev:>9.1f}{tip:>7.3f}{was:>8.2f}{bur:>7.2f}")


def write_episodes(rows, path):
    keys = ["config", "policy", "seed", "tag", "total_reward"] + \
           sorted({k for r in rows for k in r} - {"config", "policy", "seed", "tag", "total_reward"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-episode results written to {path}")


def write_summary(rows, path, configs, policies):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "policy", "n_seeds", "metric", "mean", "std"])
        for cfg in configs:
            for pol in policies:
                n = sum(1 for r in rows if r["config"] == cfg and r["policy"] == pol)
                if n == 0:
                    continue
                for metric, _label in SUMMARY_METRICS:
                    m, sd = mean_sd(rows, cfg, pol, metric)
                    w.writerow([cfg, pol, n, metric, round(m, 4), round(sd, 4)])
    print(f"summary written to {path}")


def apply_overrides(cfg: KitchenConfig, overrides, arrival_scale: float = 1.0):
    """Change a few parameters at run time, without editing configs/env.

    `--override burn_after=12 --override orders.patience_min=130` are applied first,
    then `--arrival-scale` multiplies the order arrival rate (schedule or constant).
    """
    for item in overrides or []:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        *parents, leaf = key.strip().split(".")
        node = cfg
        for part in parents:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        if isinstance(node, dict):
            node[leaf] = value
        else:
            setattr(node, leaf, value)
    if arrival_scale != 1.0:
        schedule = cfg.orders.get("arrival_schedule")
        if schedule:
            cfg.orders["arrival_schedule"] = [[t, r * arrival_scale] for t, r in schedule]
        else:
            cfg.orders["arrival_rate"] = cfg.orders["arrival_rate"] * arrival_scale
    if overrides or arrival_scale != 1.0:
        cfg.__post_init__()          # re-run the merge / assertion logic
    return cfg


def parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def output_paths(policy_arg: str, out: str | None):
    """Where the per-policy CSVs go (one file per policy when both are run)."""
    policies = ["random", "rule"] if policy_arg == "both" else [policy_arg]
    if out is None:
        return policies, {p: f"results/csv/baselines_{p}.csv" for p in policies}
    if len(policies) == 1:
        return policies, {policies[0]: out}
    root, ext = os.path.splitext(out)
    return policies, {p: f"{root}_{p}{ext or '.csv'}" for p in policies}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["random", "rule", "both"], default="rule",
                   help='"both" runs the random and the rule baseline on the same seeds')
    p.add_argument("--config", nargs="+", default=["configs/env/medium.yaml"])
    p.add_argument("--seeds", default="0-19", help='e.g. "0-19" or "0,1,2"')
    p.add_argument("--out", default=None,
                   help="CSV path; default results/csv/baselines_<policy>.csv "
                        "(with --policy both the policy name is inserted before the extension)")
    p.add_argument("--trace", type=int, default=0, help="print the state every N steps (first seed only)")
    p.add_argument("--tag", default="", help="free-form label stored in the CSV")
    p.add_argument("--max-plans", type=int, default=2, help="how many dishes may be in flight")
    p.add_argument("--override", action="append", default=[],
                   help="runtime parameter override, e.g. --override burn_after=12")
    p.add_argument("--arrival-scale", type=float, default=1.0,
                   help="multiply the order arrival rate (difficulty experiment)")
    p.add_argument("--solo", action="store_true",
                   help="one agent does all the work (reference for what cooperation adds)")
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    configs = [os.path.splitext(os.path.basename(c))[0] + ("/solo" if args.solo else "")
               for c in args.config]
    policies, paths = output_paths(args.policy, args.out)
    for path in paths.values():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rows = []
    for policy_name in policies:
        for cfg_path in args.config:
            name = os.path.splitext(os.path.basename(cfg_path))[0] + ("/solo" if args.solo else "")
            cfg = apply_overrides(KitchenConfig.from_yaml(cfg_path), args.override,
                                  args.arrival_scale)
            for i, seed in enumerate(seeds):
                core = KitchenCore(cfg)
                if policy_name == "rule":
                    policy = RulePolicy(args.max_plans, agents=(0,) if args.solo else (0, 1))
                else:
                    policy = RandomPolicy(seed)
                trace = args.trace if i == 0 and policy_name == "rule" else 0
                total, info = run_episode(core, policy, seed, trace)
                label = name
                row = {"config": label, "policy": policy_name, "seed": seed,
                       "total_reward": round(total, 4), "tag": args.tag}
                row.update({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in flat_info(info).items()})
                rows.append(row)
                print(f"{name:8s} {policy_name:6s} seed {seed:3d} | reward {total:7.2f} | "
                      f"completed {info['orders_completed']:2d}/{info['orders_arrived']:2d} arrived, "
                      f"{info['orders_expired']:2d} expired | dishes {info['dishes_delivered']:2d}")

    for policy_name, path in paths.items():
        write_episodes([r for r in rows if r["policy"] == policy_name], path)
    summary_path = os.path.join(os.path.dirname(paths[policies[0]]) or ".", "baselines_summary.csv")
    write_summary(rows, summary_path, configs, policies)

    for name in configs:
        for policy_name in policies:
            summarize(rows, name, policy_name)
    if len(policies) > 1:
        comparison_table(rows, configs, policies)


if __name__ == "__main__":
    main()
