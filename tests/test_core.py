"""环境规则测试：在项目根目录运行 pytest tests"""
import numpy as np
import pytest

from restaurant_env.core import (
    DIRTY, DISH_READY, EATING, EMPTY, PHASE_DESSERT, WAIT_FOOD, WAIT_ORDER,
    Group, RestaurantConfig, RestaurantCore,
)
from restaurant_env.centralized_env import CentralizedRestaurantEnv

CONFIGS = ["configs/env/easy.yaml", "configs/env/medium.yaml", "configs/env/hard.yaml"]
ONE_DISH = [{"name": "x", "category": "main", "cook_time": 2, "price": 10.0}]


def random_rollout(core, seed, use_mask=True):
    rng = np.random.default_rng(seed)
    core.reset(seed)
    done = False
    while not done:
        if use_mask:
            c = rng.choice(np.flatnonzero(core.chef_mask()))
            s = rng.choice(np.flatnonzero(core.server_mask()))
        else:
            c = rng.integers(core.n_chef_actions)
            s = rng.integers(core.n_server_actions)
        _, _, _, done, info = core.step(c, s)
        yield core, info


def quiet_core(**kw):
    """没有随机到达的环境，便于手动构造场景。"""
    base = dict(arrival_rate=0.0, arrival_schedule=None, patience_min=100, patience_max=100)
    base.update(kw)
    if max(base.get("table_sizes", [4])) < 4:   # 小桌场景：最多 2 人一组
        base.setdefault("party_size_probs", [0.5, 0.5])
    core = RestaurantCore(RestaurantConfig(**base))
    core.reset(0)
    return core


def add_group(core, size, patience=100.0):
    g = Group(core._next_gid, size, patience, patience, core.t)
    core._next_gid += 1
    core.queue.append(g)
    return g


# ---------------------------------------------------------------- 基本性质
@pytest.mark.parametrize("path", CONFIGS)
def test_configs_load(path):
    RestaurantCore(RestaurantConfig.from_yaml(path)).reset(0)


@pytest.mark.parametrize("path", CONFIGS)
@pytest.mark.parametrize("use_mask", [True, False])
def test_invariants(path, use_mask):
    """盘子守恒、出餐口与灶台不超容量、桌上人数不超座位数"""
    cfg = RestaurantConfig.from_yaml(path)
    core = RestaurantCore(cfg)
    for seed in range(3):
        for c, _ in random_rollout(core, seed, use_mask):
            assert c.clean_plates + c.dirty_plates + c.plates_in_use() == cfg.n_plates
            assert c.clean_plates >= 0 and c.dirty_plates >= 0
            assert c._pass_count() + len(c.stoves) <= cfg.pass_capacity
            assert len(c.stoves) <= cfg.n_stoves
            for tb in c.tables:
                if tb.group is not None:
                    assert tb.group.size <= tb.capacity


def test_masked_actions_never_invalid():
    core = RestaurantCore(RestaurantConfig.from_yaml("configs/env/medium.yaml"))
    *_, (_, info) = random_rollout(core, 0, use_mask=True)
    assert info["invalid_chef"] == 0 and info["invalid_server"] == 0


def test_determinism():
    core = RestaurantCore(RestaurantConfig.from_yaml("configs/env/medium.yaml"))
    a = [i["revenue"] for _, i in random_rollout(core, 42)]
    b = [i["revenue"] for _, i in random_rollout(core, 42)]
    assert a == b


def test_episode_length():
    core = RestaurantCore(RestaurantConfig(episode_length=50))
    assert sum(1 for _ in random_rollout(core, 0)) == 50


# ---------------------------------------------------------------- 具体规则
def test_arrival_schedule_interpolation():
    core = RestaurantCore(RestaurantConfig(arrival_schedule=[[0, 0.0], [100, 0.2], [200, 0.2]]))
    assert core.arrival_rate_at(0) == 0.0
    assert core.arrival_rate_at(50) == pytest.approx(0.1)
    assert core.arrival_rate_at(150) == pytest.approx(0.2)
    assert core.arrival_rate_at(999) == pytest.approx(0.2)   # 超出范围保持端点值


def test_big_group_cannot_sit_at_small_table():
    core = quiet_core(table_sizes=[2, 4])
    add_group(core, 4)
    m = core.server_mask()
    assert not m[1]      # 0 号桌只有 2 座
    assert m[2]          # 1 号桌 4 座


def test_seat_picks_first_group_that_fits():
    core = quiet_core(table_sizes=[2, 4])
    add_group(core, 4)
    small = add_group(core, 2)
    core.step(0, 1)      # 带位到 2 人桌：跳过排在前面的 4 人组
    assert core.tables[0].group is small and len(core.queue) == 1


def test_cannot_cook_without_clean_plate():
    core = quiet_core(table_sizes=[2], n_plates=1, initial_dirty_plates=1, dishes=ONE_DISH)
    add_group(core, 1)
    core.step(0, 1); core.step(0, 2)       # 带位、点单
    assert core.tables[0].state == WAIT_FOOD
    assert not core.chef_mask()[1]
    core.step(1, 0)
    assert core.stats["invalid_chef"] == 1


def test_customer_walks_away_when_patience_runs_out():
    core = quiet_core()
    add_group(core, 3, patience=3)
    for _ in range(3):
        core.step(0, 0)
    assert core.stats["walkaway_queue_groups"] == 1 and core.stats["walkaway_people"] == 3


def test_tip_rate_formula():
    core = quiet_core()
    assert core.tip_rate(5, 0.0) == pytest.approx(0.20)    # 宽限期内
    assert core.tip_rate(35, 0.0) == pytest.approx(0.10)   # 10 → 60 的中点
    assert core.tip_rate(80, 0.0) == pytest.approx(0.0)
    assert core.tip_rate(5, 1.0) == pytest.approx(0.10)    # 全凉菜减半


def test_cold_dish_counted():
    core = quiet_core(table_sizes=[2], dishes=ONE_DISH, cold_after=2,
                      dessert_prob=0.0, eat_time_min=2, eat_time_max=2)
    add_group(core, 1)
    core.step(0, 1); core.step(0, 2)        # 带位、点单
    core.step(1, 0); core.step(0, 0)        # 做菜 2 步 → 出餐口
    for _ in range(3):                      # 在出餐口放 3 步
        core.step(0, 0)
    core.step(0, 3)                         # 上菜
    assert core.stats["cold_dishes"] == 1


def test_full_service_cycle_with_dessert():
    """完整流程：带位 → 点主菜 → 做 → 上 → 吃 → 点甜点 → 做 → 上 → 吃 → 付款 → 清桌 → 洗碗"""
    dishes = ONE_DISH + [{"name": "d", "category": "dessert", "cook_time": 1, "price": 5.0}]
    core = quiet_core(table_sizes=[2], dishes=dishes, dessert_prob=1.0,
                      eat_time_min=2, eat_time_max=2, dessert_eat_time_min=2, dessert_eat_time_max=2,
                      n_plates=8)
    add_group(core, 2)
    N = core.N
    core.step(0, 1)                                # 带位
    core.step(0, 1 + N)                            # 点主菜（2 道）
    assert len(core.tables[0].items) == 2
    core.step(1, 0); core.step(1, 0)               # 两道菜分别上灶
    core.step(0, 0)
    assert all(it.status == DISH_READY for it in core.tables[0].items)
    core.step(0, 1 + 2 * N)                        # 上菜
    assert core.tables[0].state == EATING and core.tables[0].plates == 2
    core.step(0, 0)                                # 吃完 → 进入甜点阶段
    tb = core.tables[0]
    assert tb.state == WAIT_ORDER and tb.phase == PHASE_DESSERT
    core.step(0, 1 + N)                            # 点甜点（每人一份）
    core.step(1, 0); core.step(1, 0)
    core.step(0, 1 + 2 * N)                        # 上甜点
    assert tb.state == EATING
    core.step(0, 0)                                # 吃完付款
    assert tb.state == DIRTY and tb.plates == 4
    assert core.stats["bill_total"] == 30.0 and core.stats["tip_total"] > 0
    core.step(0, 1 + 3 * N)                        # 清桌：4 个盘子进洗碗池
    assert tb.state == EMPTY and core.dirty_plates == 4
    core.step(N + 1, 0); core.step(0, 0)           # 洗一批（4 个，2 步）
    assert core.dirty_plates == 0 and core.clean_plates == 8


def test_dessert_abandon_still_pays_main():
    dishes = ONE_DISH + [{"name": "d", "category": "dessert", "cook_time": 1, "price": 5.0}]
    core = quiet_core(table_sizes=[2], dishes=dishes, dessert_prob=1.0, dessert_patience=2,
                      eat_time_min=1, eat_time_max=1)
    add_group(core, 1)
    N = core.N
    core.step(0, 1); core.step(0, 1 + N)
    core.step(1, 0); core.step(0, 0)
    core.step(0, 1 + 2 * N)
    core.step(0, 0)                                # 吃完，进入甜点阶段
    assert core.tables[0].phase == PHASE_DESSERT
    for _ in range(3):                             # 不理他们，耐心耗尽
        core.step(0, 0)
    assert core.stats["dessert_abandoned"] == 1
    assert core.stats["bill_total"] == 10.0        # 主菜照付
    assert core.tables[0].state == DIRTY


def test_gym_api():
    from gymnasium.utils.env_checker import check_env
    env = CentralizedRestaurantEnv(config_path="configs/env/medium.yaml")
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert env.action_masks().shape[0] == sum(env.action_space.nvec)


def test_obs_dim_independent_of_menu():
    a = CentralizedRestaurantEnv(RestaurantConfig())
    b = CentralizedRestaurantEnv(RestaurantConfig(dishes=ONE_DISH))
    assert a.observation_space.shape == b.observation_space.shape
