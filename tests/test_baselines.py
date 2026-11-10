"""
Smoke tests for the baseline policies (scripts/baselines.py).

They check that the rule policy actually serves dishes (so the environment is
solvable), that it is deterministic, and that neither policy crashes on any of the
three difficulties. They are not a performance guarantee: tuning the rule policy is
allowed to change the exact numbers, so only lower bounds are asserted.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baselines import RandomPolicy, RulePolicy, run_episode  # noqa: E402
from restaurant_env import KitchenConfig, KitchenCore  # noqa: E402


def make_core(name: str) -> KitchenCore:
    return KitchenCore(KitchenConfig.from_yaml(os.path.join(ROOT, "configs", "env", f"{name}.yaml")))


def test_rule_policy_serves_orders():
    core = make_core("easy")
    total, info = run_episode(core, RulePolicy(), seed=0)
    assert info["t"] == core.cfg.episode_length
    assert info["orders_arrived"] > 0
    assert info["dishes_delivered"] >= 4
    assert info["orders_completed"] >= 2
    assert info["wasted"] <= 3
    assert total > 0


def test_rule_policy_is_deterministic():
    first, _ = run_episode(make_core("easy"), RulePolicy(), seed=3)
    second, _ = run_episode(make_core("easy"), RulePolicy(), seed=3)
    assert first == second


def test_random_policy_does_not_crash():
    core = make_core("medium")
    _, info = run_episode(core, RandomPolicy(0), seed=0)
    assert info["t"] == core.cfg.episode_length


@pytest.mark.parametrize("name", ["easy", "medium", "hard"])
def test_rule_policy_runs_on_every_difficulty(name):
    core = make_core(name)
    _, info = run_episode(core, RulePolicy(), seed=0)
    assert info["t"] == core.cfg.episode_length
    assert info["orders_arrived"] > 0
    assert info["a0_invalid"] == 0 and info["a1_invalid"] == 0
