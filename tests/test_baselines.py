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

from baselines import (RandomPolicy, RulePolicy, apply_overrides, main,  # noqa: E402
                       mean_sd, output_paths, run_episode)
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


def test_apply_overrides_does_not_touch_the_config_files():
    cfg = KitchenConfig.from_yaml(os.path.join(ROOT, "configs", "env", "medium.yaml"))
    apply_overrides(cfg, ["burn_after=12", "orders.patience_min=130"], arrival_scale=0.5)
    assert cfg.burn_after == 12
    assert cfg.orders["patience_min"] == 130
    assert cfg.orders["arrival_schedule"][2] == [120, pytest.approx(0.035)]
    # the file on disk is unchanged (the override is a runtime thing)
    fresh = KitchenConfig.from_yaml(os.path.join(ROOT, "configs", "env", "medium.yaml"))
    assert fresh.burn_after == 8


def test_output_paths():
    policies, paths = output_paths("rule", None)
    assert policies == ["rule"] and paths["rule"].endswith("baselines_rule.csv")
    policies, paths = output_paths("both", "results/csv/run.csv")
    assert policies == ["random", "rule"]
    assert paths["rule"] == "results/csv/run_rule.csv"
    assert paths["random"] == "results/csv/run_random.csv"


def test_mean_sd():
    rows = [{"config": "easy", "policy": "rule", "x": 1.0},
            {"config": "easy", "policy": "rule", "x": 3.0},
            {"config": "easy", "policy": "random", "x": 9.0}]
    m, sd = mean_sd(rows, "easy", "rule", "x")
    assert m == 2.0 and sd == 1.0


def test_cli_writes_episode_and_summary_files(tmp_path, monkeypatch):
    out = tmp_path / "run.csv"
    monkeypatch.setattr("sys.argv", [
        "baselines.py", "--policy", "both",
        "--config", os.path.join(ROOT, "configs", "env", "easy.yaml"),
        "--seeds", "0-1", "--out", str(out),
    ])
    main()
    assert (tmp_path / "run_rule.csv").exists()
    assert (tmp_path / "run_random.csv").exists()
    summary = tmp_path / "baselines_summary.csv"
    assert summary.exists()
    header = summary.read_text().splitlines()[0]
    assert header == "config,policy,n_seeds,metric,mean,std"


@pytest.mark.parametrize("name", ["easy", "medium", "hard"])
def test_rule_policy_runs_on_every_difficulty(name):
    core = make_core(name)
    _, info = run_episode(core, RulePolicy(), seed=0)
    assert info["t"] == core.cfg.episode_length
    assert info["orders_arrived"] > 0
    assert info["a0_invalid"] == 0 and info["a1_invalid"] == 0
