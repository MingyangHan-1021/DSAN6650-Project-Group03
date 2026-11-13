"""
Tests for the capacity bound (scripts/capacity.py).

The bound is the "how fast could this kitchen possibly be" reference used to judge
whether a difficulty level is calibrated. If the numbers here drift, the difficulty
argument in docs/baselines.md has to be re-checked, hence the tests.
"""
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baselines import RulePolicy, run_episode  # noqa: E402
from capacity import Bound  # noqa: E402
from restaurant_env import KitchenConfig, KitchenCore  # noqa: E402


def capacity(config_name: str):
    cfg = KitchenConfig.from_yaml(os.path.join(ROOT, "configs", "env", f"{config_name}.yaml"))
    bound = Bound(cfg)
    per_dish = [bound.dish(dish) for dish in cfg.menu]
    wall_steps = statistics.fmean(per_dish) / len(cfg.agents)
    return cfg, per_dish, cfg.episode_length / wall_steps


def test_every_dish_needs_a_plausible_number_of_steps():
    _cfg, per_dish, _capacity = capacity("medium")
    assert all(20 <= steps <= 60 for steps in per_dish), per_dish


def test_medium_capacity_matches_the_design_note():
    # README: "two agents should manage roughly one dish every 15-18 steps"
    _cfg, _per_dish, cap = capacity("medium")
    assert 15.0 <= cap <= 20.0


def test_capacity_shrinks_as_the_menu_gets_harder():
    # pizza needs the oven, so hard's menu must not be cheaper per dish than easy's
    _c1, d1, c1 = capacity("easy")
    _c2, d2, c2 = capacity("hard")
    assert statistics.fmean(d2) >= statistics.fmean(d1)
    assert c2 <= c1


def test_rule_policy_stays_below_the_bound():
    cfg, per_dish, cap = capacity("medium")
    core = KitchenCore(cfg)
    _total, info = run_episode(core, RulePolicy(), seed=0)
    assert info["dishes_delivered"] <= cap
