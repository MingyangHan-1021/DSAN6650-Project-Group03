"""Rule tests for the environment: run `pytest tests` from the project root."""
import csv
import json

import numpy as np
import pytest

from restaurant_env import (
    CentralizedKitchenEnv, EpisodeLogger, KitchenConfig, KitchenCore,
    generate_orders, load_orders_csv, save_orders_csv,
)
from restaurant_env.core import C_BURNT, C_COOKING, C_DONE
from restaurant_env.orders import Order

CONFIGS = ["configs/env/easy.yaml", "configs/env/medium.yaml", "configs/env/hard.yaml"]
ALL_DISHES = ["salad", "tomato_soup", "burger", "pasta", "pizza"]


# ---------------------------------------------------------------- helpers
def make_core(menu=ALL_DISHES, layout="data/layouts/open.json", orders=None, **kw):
    cfg = KitchenConfig(menu=menu, layout_file=layout, **kw)
    core = KitchenCore(cfg)
    core.reset(0, orders=orders if orders is not None else [])
    return core


def order(oid, dishes, arrival=0, deadline=10_000):
    return Order(oid, arrival, deadline, list(dishes))


def do(core, a, station, expect_ok=True):
    """Send agent a to a station while the other agent waits, until the action finishes.
    Returns the number of steps it took."""
    sid = core.layout.id[station]
    acts = [0, 0]
    acts[a] = sid + 1
    assert core.action_mask(a)[sid + 1] == expect_ok, f"{core.agent_names[a]} @ {station}"
    if not expect_ok:
        return 0
    failed = core.stats["failed"][a]
    core.step(acts)
    n = 1
    while core.task[a] is not None:
        core.step([0, 0])
        n += 1
    assert core.stats["failed"][a] == failed, f"{station} failed"
    return n


def wait(core, n):
    for _ in range(n):
        core.step([0, 0])


def random_rollout(core, seed, use_mask=True):
    rng = np.random.default_rng(seed)
    core.reset(seed)
    done = False
    while not done:
        if use_mask:
            acts = [rng.choice(np.flatnonzero(core.action_mask(a))) for a in (0, 1)]
        else:
            acts = rng.integers(0, core.n_actions, size=2)
        _, _, _, done, info = core.step(acts)
        yield core, info


# ---------------------------------------------------------------- basic properties
@pytest.mark.parametrize("path", CONFIGS)
def test_configs_load(path):
    KitchenCore(KitchenConfig.from_yaml(path)).reset(0)


@pytest.mark.parametrize("path", CONFIGS)
@pytest.mark.parametrize("use_mask", [True, False])
def test_invariants(path, use_mask):
    """Plates are conserved; the two agents never target the same station."""
    cfg = KitchenConfig.from_yaml(path)
    core = KitchenCore(cfg)
    for seed in range(2):
        for c, _ in random_rollout(core, seed, use_mask):
            assert c.plates_total() == cfg.n_plates
            assert c.clean_plates >= 0 and c.sink_dirty >= 0
            if c.task[0] is not None and c.task[1] is not None:
                assert c.task[0].target != c.task[1].target


def test_masked_actions_never_invalid():
    core = KitchenCore(KitchenConfig.from_yaml("configs/env/medium.yaml"))
    *_, (_, info) = random_rollout(core, 0)
    assert info["a0_invalid"] == 0 and info["a1_invalid"] == 0


def test_determinism():
    core = KitchenCore(KitchenConfig.from_yaml("configs/env/medium.yaml"))
    a = [json.dumps(c.snapshot(), ensure_ascii=False) for c, _ in random_rollout(core, 7)]
    b = [json.dumps(c.snapshot(), ensure_ascii=False) for c, _ in random_rollout(core, 7)]
    assert a == b


def test_episode_length():
    core = KitchenCore(KitchenConfig(episode_length=40))
    assert sum(1 for _ in random_rollout(core, 0)) == 40


# ---------------------------------------------------------------- order CSV
def test_orders_csv_roundtrip(tmp_path):
    cfg = KitchenConfig.from_yaml("configs/env/medium.yaml")
    orders = generate_orders(cfg.orders, cfg.menu, cfg.episode_length, seed=3)
    path = tmp_path / "orders.csv"
    save_orders_csv(orders, path)
    loaded = load_orders_csv(path)
    assert [(o.oid, o.arrival_t, o.deadline_t, o.dishes) for o in orders] == \
           [(o.oid, o.arrival_t, o.deadline_t, o.dishes) for o in loaded]


def test_orders_file_used(tmp_path):
    path = tmp_path / "orders.csv"
    save_orders_csv([order(0, ["salad"], arrival=0, deadline=50)], path)
    core = KitchenCore(KitchenConfig(orders={"orders_file": str(path)}))
    core.reset(123)
    assert [o.oid for o in core.open_orders] == [0]


# ---------------------------------------------------------------- every dish can be made
def test_salad():
    core = make_core(orders=[order(0, ["salad"])])
    for st in ["lettuce_bin", "cutting_board", "counter_b", "tomato_bin", "cutting_board", "counter_c", "plate_rack", "counter_b", "counter_c", "pass"]:
        do(core, 1, st)
    assert core.stats["orders_completed"] == 1 and core.stats["bill_total"] == 10


def test_tomato_soup_and_burn():
    core = make_core(orders=[order(0, ["tomato_soup"])], burn_after=3)
    for _ in range(2):
        do(core, 1, "tomato_bin"); do(core, 1, "cutting_board"); do(core, 1, "pot_1")
    pot = core.cookers[core.layout.id["pot_1"]]
    assert pot.status == C_COOKING
    do(core, 1, "plate_rack")                       # grab a plate first and wait
    while pot.status != C_DONE:
        wait(core, 1)
    do(core, 1, "pot_1"); do(core, 1, "pass")
    assert core.stats["orders_completed"] == 1
    # cook another pot and leave it: it burns
    for _ in range(2):
        do(core, 1, "tomato_bin"); do(core, 1, "cutting_board"); do(core, 1, "pot_1")
    wait(core, 6 + 4)
    pot = core.cookers[core.layout.id["pot_1"]]
    assert pot.status == C_BURNT and core.stats["burnt"] == 1
    do(core, 0, "pot_1")                        # clean it with empty hands
    assert core.cookers[core.layout.id["pot_1"]].contents == []


def test_burger_with_handoff():
    """Assistant chops lettuce onto a counter; chef fries beef, takes a plate, adds the bun,
    picks up the lettuce and serves."""
    core = make_core(orders=[order(0, ["burger"])])
    do(core, 1, "lettuce_bin"); do(core, 1, "cutting_board"); do(core, 1, "counter_b")
    do(core, 0, "beef_bin"); do(core, 0, "pan"); do(core, 0, "counter_a")
    do(core, 0, "plate_rack"); do(core, 0, "bun_bin"); do(core, 0, "counter_a")
    do(core, 0, "counter_b"); do(core, 0, "pass")
    assert core.stats["orders_completed"] == 1 and core.stats["handoffs"] == 1


def test_pasta():
    core = make_core(orders=[order(0, ["pasta"])])
    do(core, 1, "tomato_bin"); do(core, 1, "cutting_board"); do(core, 1, "pan"); do(core, 1, "counter_b")
    do(core, 0, "noodle_bin"); do(core, 0, "pot_2"); do(core, 0, "plate_rack")
    wait(core, 5)
    do(core, 0, "pot_2"); do(core, 0, "counter_b"); do(core, 0, "pass")
    assert core.stats["orders_completed"] == 1


def test_pizza():
    core = make_core(orders=[order(0, ["pizza"])])
    do(core, 1, "dough_bin"); do(core, 1, "oven")
    do(core, 1, "tomato_bin"); do(core, 1, "cutting_board"); do(core, 1, "oven")
    do(core, 1, "cheese_bin"); do(core, 1, "cutting_board"); do(core, 1, "oven")
    wait(core, 8)
    do(core, 1, "plate_rack"); do(core, 1, "oven"); do(core, 1, "pass")
    assert core.stats["orders_completed"] == 1


def test_multi_dish_order_tip_only_when_complete():
    core = make_core(orders=[order(0, ["salad", "salad"])])
    for _ in range(2):
        for st in ["lettuce_bin", "cutting_board", "counter_b", "tomato_bin", "cutting_board", "counter_c", "plate_rack", "counter_b", "counter_c", "pass"]:
            do(core, 1, st)
        if core.stats["dishes_delivered"] == 1:
            assert core.stats["tip_total"] == 0 and core.stats["orders_completed"] == 0
    assert core.stats["orders_completed"] == 1 and core.stats["tip_total"] > 0


# ---------------------------------------------------------------- specific rules
def test_cannot_deliver_unordered_dish():
    core = make_core(orders=[order(0, ["burger"])])
    for st in ["lettuce_bin", "cutting_board", "counter_b", "tomato_bin", "cutting_board", "counter_c", "plate_rack", "counter_b", "counter_c"]:
        do(core, 1, st)
    do(core, 1, "pass", expect_ok=False)      # a salad is ready but nobody ordered one


def test_wrong_ingredient_not_allowed_on_plate():
    core = make_core()
    do(core, 1, "plate_rack")
    do(core, 1, "lettuce_bin", expect_ok=False)      # raw lettuce is not part of any dish
    do(core, 1, "bun_bin")                       # the bun can go straight onto the plate (burger)


def test_station_exclusive():
    """A station the other agent is walking to or using cannot be chosen; it is released when the action ends."""
    core = make_core()
    tomato = core.layout.id["tomato_bin"] + 1
    core.step([tomato, 0])
    assert not core.action_mask(1)[tomato]
    while core.task[0] is not None:
        core.step([0, 0])
    assert core.action_mask(1)[tomato]


def test_simultaneous_choice_second_is_invalid():
    core = make_core()
    s = core.layout.id["tomato_bin"] + 1
    core.step([s, s])
    assert core.stats["invalid"] == [0, 0] and core.stats["conflicts"] == 1


def test_order_expires():
    core = make_core(orders=[order(0, ["salad"], arrival=0, deadline=5)])
    wait(core, 5)
    assert core.stats["orders_expired"] == 1 and core.open_orders == []


def test_tip_rate():
    core = make_core()
    assert core.tip_rate(10) == pytest.approx(0.20)
    assert core.tip_rate(65) == pytest.approx(0.10)    # midpoint of 30 -> 100
    assert core.tip_rate(200) == 0.0


def test_dirty_plate_returns_and_wash():
    core = make_core(orders=[order(0, ["tomato_soup"])], dirty_return_delay=3)
    for _ in range(2):
        do(core, 1, "tomato_bin"); do(core, 1, "cutting_board"); do(core, 1, "pot_1")
    do(core, 1, "plate_rack")
    wait(core, 6)
    do(core, 1, "pot_1"); do(core, 1, "pass")
    wait(core, 3)
    assert core.sink_dirty == 1
    clean = core.clean_plates
    do(core, 1, "sink")
    assert core.clean_plates == clean + 1 and core.sink_dirty == 0


def test_skill_times_differ():
    core = make_core()
    do(core, 0, "tomato_bin")
    t_chef = do(core, 0, "cutting_board")
    do(core, 0, "counter_a")
    do(core, 1, "tomato_bin")
    t_commis = do(core, 1, "cutting_board")
    # both walk from the tomato bin to the cutting board: same travel time,
    # the only difference is the chop skill (3 vs 2)
    assert t_chef - t_commis == 1


def test_split_layout_zones():
    core = make_core(layout="data/layouts/split.json")
    chef, commis = 0, 1
    assert not core.action_mask(chef)[core.layout.id["tomato_bin"] + 1]
    assert not core.action_mask(chef)[core.layout.id["plate_rack"] + 1]
    assert not core.action_mask(commis)[core.layout.id["pass"] + 1]
    do(core, commis, "plate_rack"); do(core, commis, "counter_b")
    do(core, chef, "counter_b")                     # the plate can only reach the chef through a counter
    assert core.stats["handoffs"] == 1


def test_trash_penalized():
    core = make_core()
    do(core, 1, "tomato_bin")
    _, r, _, _, _ = core.step([0, core.layout.id["trash"] + 1])
    while core.task[1] is not None:
        _, r, _, _, _ = core.step([0, 0])
    assert core.stats["wasted"] == 1 and r < 0


# ---------------------------------------------------------------- API and logging
def test_gym_api():
    from gymnasium.utils.env_checker import check_env
    env = CentralizedKitchenEnv(config_path="configs/env/medium.yaml")
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_masks().shape[0] == sum(env.action_space.nvec)


def test_logger(tmp_path):
    core = KitchenCore(KitchenConfig.from_yaml("configs/env/easy.yaml"))
    rng = np.random.default_rng(0)
    core.reset(0)
    log = EpisodeLogger(core)
    done = False
    while not done:
        acts = [rng.choice(np.flatnonzero(core.action_mask(a))) for a in (0, 1)]
        _, r, _, done, info = core.step(acts)
        log.record(acts, r, info)
    log.save(tmp_path)
    with open(tmp_path / "actions.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == core.cfg.episode_length
    with open(tmp_path / "frames.jsonl", encoding="utf-8") as f:
        assert len(f.readlines()) == core.cfg.episode_length + 1
    assert (tmp_path / "events.csv").exists() and (tmp_path / "summary.json").exists()
