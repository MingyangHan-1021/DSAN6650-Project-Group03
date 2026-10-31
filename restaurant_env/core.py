"""
餐厅环境的底层逻辑（与 Gym / PettingZoo 接口无关）。

中心化版本（centralized_env.py）和多智能体版本（marl_env.py）都调用这里，
保证两种设定下的规则完全一致，对比才公平。

基本设定
--------
- 时间离散：1 步 ≈ 1 分钟。
- 两个角色：
    厨师（chef）：在灶台上做菜、洗盘子
    服务员（server）：带位、点单、上菜、清桌
- 顾客以"组"为单位到达，每组 1~4 人；桌子有不同大小（如两人桌、四人桌），
  一组顾客只能坐进容量足够的桌子。
- 每人点一道主菜；吃完主菜后，每人有一定概率追加甜点。
- 菜单分主菜和甜点，做菜时间和价格不同。

顾客流程
--------
到达排队 → 带位 → 点主菜 → 厨师做菜 → 出餐口 → 上菜（主菜上齐才开始吃）→ 用餐
  → [可能] 追加甜点：点甜点 → 做 → 上 → 吃
  → 付款（账单 + 小费）离开 → 桌子变脏 → 清桌 → 脏盘子进洗碗池 → 厨师洗碗

共享瓶颈
--------
- 干净盘子：每道菜开始做时占用一个干净盘子；盘子一直留在桌上，清桌时才送回洗碗池。
- 出餐口：做好的菜放在这里，容量有限。正在灶台上做的菜也预占出餐口位置，
  所以"出餐口上的菜 + 灶台上的菜"不能超过出餐口容量。
- 灶台：同时最多做 n_stoves 道菜。

小费与满意度
------------
小费 = 账单 × 小费率
小费率 = tip_max_rate × 等待系数 × (1 - cold_tip_penalty × 凉菜比例)
- 等待时间 = 到达 → 主菜上齐，若吃了甜点再加上 甜点阶段开始 → 甜点上齐
- 等待系数：等待 ≤ tip_grace_wait 时为 1，之后线性下降，到 tip_zero_wait 时为 0
- 凉菜：在出餐口放了超过 cold_after 步才被送到桌上的菜

动作的时间语义
--------------
- 服务员动作：选择后立即生效，然后锁定若干步（模拟耗时）。
- 厨师开始做菜：只锁定 prep_time 步（备菜），之后菜在灶台上自己计时，
  厨师可以去做别的事（开始另一道菜或洗碗）。
- 厨师洗碗：一次洗最多 wash_batch 个，锁定 wash_time 步，完成时盘子变干净。
- 被锁定的智能体只能选 0（继续），其他动作会被忽略，且不算非法。

每步结算顺序
------------
1. 厨师动作  2. 服务员动作  3. 灶台计时（做好的菜进出餐口）与厨师锁定计时
4. 服务员锁定计时  5. 出餐口上的菜放凉计时  6. 用餐计时（吃完 → 甜点或付款）
7. 耐心衰减与顾客离开  8. 新顾客到达  9. 时间 +1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

from .rewards import compute_reward

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 桌子状态
EMPTY = 0       # 空桌，可以带位
WAIT_ORDER = 1  # 已入座，等待点单（主菜或甜点）
WAIT_FOOD = 2   # 已点单，菜还没上齐
EATING = 3      # 用餐中
DIRTY = 4       # 顾客已离开，桌上有脏盘子，需要清理
TABLE_STATE_NAMES = ["空桌", "等点单", "等上菜", "用餐中", "待清理"]
N_TABLE_STATES = 5

# 用餐阶段
PHASE_MAIN = 0
PHASE_DESSERT = 1

# 单道菜的状态
DISH_PENDING = 0    # 已下单，还没开始做
DISH_COOKING = 1    # 在灶台上
DISH_READY = 2      # 在出餐口
DISH_DELIVERED = 3  # 已送到桌上


# ---------------------------------------------------------------------------
# 配置默认值
# ---------------------------------------------------------------------------
def _default_dishes():
    # category: main 主菜 / dessert 甜点；weight: 被点到的相对概率（同类之内）
    return [
        {"name": "沙拉",     "category": "main",    "cook_time": 2,  "price": 9.0,  "weight": 1.0},
        {"name": "例汤",     "category": "main",    "cook_time": 2,  "price": 7.0,  "weight": 1.0},
        {"name": "汉堡",     "category": "main",    "cook_time": 4,  "price": 14.0, "weight": 1.5},
        {"name": "意面",     "category": "main",    "cook_time": 5,  "price": 16.0, "weight": 1.5},
        {"name": "披萨",     "category": "main",    "cook_time": 6,  "price": 18.0, "weight": 1.2},
        {"name": "烤鸡",     "category": "main",    "cook_time": 8,  "price": 22.0, "weight": 1.0},
        {"name": "牛排",     "category": "main",    "cook_time": 10, "price": 28.0, "weight": 0.8},
        {"name": "海鲜饭",   "category": "main",    "cook_time": 12, "price": 32.0, "weight": 0.5},
        {"name": "冰淇淋",   "category": "dessert", "cook_time": 1,  "price": 6.0,  "weight": 1.0},
        {"name": "布朗尼",   "category": "dessert", "cook_time": 2,  "price": 8.0,  "weight": 1.0},
        {"name": "提拉米苏", "category": "dessert", "cook_time": 3,  "price": 9.0,  "weight": 0.8},
    ]


def _default_reward():
    return {
        "revenue_scale": 0.05,          # (账单 + 小费) × 系数 = 奖励
        "walkaway_penalty": 0.5,        # 主菜阶段耐心耗尽离开，每人惩罚
        "balk_penalty": 0.3,            # 门口排满直接离开，每人惩罚
        "dessert_abandon_penalty": 0.0,  # 甜点阶段等不及离开（主菜照付），每组惩罚
        "invalid_action_penalty": 0.0,  # 每个非法动作的惩罚
        "step_penalty": 0.0,            # 每步固定惩罚
        "shaping": {                    # 中间奖励（奖励塑形），可整体关闭做消融
            "enabled": True,
            "seated": 0.1,              # 每次带位
            "order_taken": 0.1,         # 每次点单（主菜或甜点）
            "dish_cooked": 0.02,        # 每道菜做好
            "dish_delivered": 0.05,     # 每道菜送到桌上
            "plate_washed": 0.01,       # 每洗好一个盘子
            "table_cleared": 0.1,       # 每次清桌
        },
    }


@dataclass
class RestaurantConfig:
    # ---- 回合 ----
    episode_length: int = 240           # 一回合多少步（240 步 ≈ 4 小时）

    # ---- 桌子 ----
    table_sizes: list = field(default_factory=lambda: [2, 2, 4, 4])  # 每张桌子的座位数

    # ---- 顾客到达 ----
    party_size_probs: list = field(default_factory=lambda: [0.2, 0.4, 0.15, 0.25])
    # 一组人数为 1,2,3,4 的概率（列表长度即最大人数）
    arrival_rate: float = 0.08          # arrival_schedule 为 null 时使用的固定到达率（组/步）
    arrival_schedule: Optional[list] = None
    # 时变到达率：[[步数, 到达率], ...]，中间线性插值，两端外保持不变
    # 例：[[0, 0.05], [60, 0.05], [90, 0.18], [150, 0.18], [180, 0.05]] 表示午餐高峰
    max_queue: int = 5                  # 门口最多排几组，满了新顾客直接离开（balk）

    # ---- 耐心 ----
    patience_min: float = 25.0          # 初始耐心在 [min, max] 均匀采样，单位：步
    patience_max: float = 40.0
    patience_decay_queue: float = 1.0   # 门口排队时每步耐心减少量
    patience_decay_order: float = 1.0   # 入座等点单时
    patience_decay_food: float = 0.5    # 点单后等上菜时
    dessert_patience: float = 15.0      # 进入甜点阶段时耐心重置为这个值

    # ---- 用餐 ----
    eat_time_min: int = 8               # 主菜用餐时长（步）
    eat_time_max: int = 15
    dessert_prob: float = 0.25          # 吃完主菜后，每人点甜点的概率
    dessert_eat_time_min: int = 3       # 甜点用餐时长（步）
    dessert_eat_time_max: int = 6

    # ---- 菜单 ----
    dishes: list = field(default_factory=_default_dishes)

    # ---- 厨房 ----
    n_stoves: int = 3                   # 灶台数：同时最多做几道菜
    prep_time: int = 1                  # 开始做一道菜时厨师被占用的步数
    n_plates: int = 16                  # 盘子总数
    initial_dirty_plates: int = 0       # 开局时脏盘子数
    wash_time: int = 2                  # 洗一批盘子需要几步
    wash_batch: int = 4                 # 一批最多洗几个
    pass_capacity: int = 6              # 出餐口容量（含灶台上预占的位置）
    cold_after: int = 3                 # 菜在出餐口放超过这么多步，送到时算凉菜

    # ---- 服务员动作耗时（步）----
    seat_time: int = 1
    order_time: int = 1
    deliver_time: int = 1
    clear_time: int = 1

    # ---- 小费 ----
    tip_max_rate: float = 0.20          # 最高小费率
    tip_grace_wait: float = 10.0        # 等待不超过这么多步，小费率不打折
    tip_zero_wait: float = 60.0         # 等待达到这么多步，小费率降为 0
    cold_tip_penalty: float = 0.5       # 全部是凉菜时，小费率再打几折（0.5 = 减半）

    # ---- 奖励 ----
    reward: dict = field(default_factory=_default_reward)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "RestaurantConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "RestaurantConfig":
        data = dict(data)
        # 奖励部分做深合并：YAML 里只写想改的项即可
        reward = _default_reward()
        user_reward = dict(data.pop("reward", {}) or {})
        user_shaping = user_reward.pop("shaping", {}) or {}
        reward.update(user_reward)
        reward["shaping"].update(user_shaping)
        cfg = cls(**data, reward=reward)
        cfg.validate()
        return cfg

    def __post_init__(self):
        self.validate()

    def validate(self):
        max_party = len(self.party_size_probs)
        assert len(self.table_sizes) >= 1
        assert max(self.table_sizes) >= max_party, "至少要有一张桌子能坐下最大的一组"
        assert abs(sum(self.party_size_probs) - 1.0) < 1e-6
        assert 0 <= self.initial_dirty_plates <= self.n_plates
        assert self.patience_min <= self.patience_max
        assert self.eat_time_min <= self.eat_time_max
        assert self.dessert_eat_time_min <= self.dessert_eat_time_max
        assert 0.0 <= self.dessert_prob <= 1.0
        assert self.tip_grace_wait < self.tip_zero_wait
        assert self.n_stoves >= 1 and self.prep_time >= 1 and self.wash_batch >= 1
        assert any(d["category"] == "main" for d in self.dishes), "至少要有一道主菜"
        for d in self.dishes:
            assert d["category"] in ("main", "dessert")
            assert d["cook_time"] >= 1


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------
@dataclass
class OrderItem:
    """一道菜。"""
    dish: int
    status: int = DISH_PENDING
    pass_wait: int = 0             # 在出餐口上已经放了几步


@dataclass
class Group:
    """一组顾客。"""
    gid: int
    size: int
    patience: float
    patience_init: float
    arrival_t: int
    seated_t: int = -1
    ordered_t: int = -1
    main_wait: float = 0.0         # 到达 → 主菜上齐
    dessert_start_t: int = -1
    dessert_wait: float = 0.0      # 甜点阶段开始 → 甜点上齐
    dessert_count: int = 0
    bill: float = 0.0              # 已送到桌上的菜的价格之和
    n_delivered: int = 0
    n_cold: int = 0


@dataclass
class Table:
    capacity: int
    state: int = EMPTY
    group: Optional[Group] = None
    phase: int = PHASE_MAIN
    items: list = field(default_factory=list)   # 当前阶段的菜
    plates: int = 0                # 桌上的盘子数（清桌时送回洗碗池）
    eat_remaining: int = 0


@dataclass
class StoveTask:
    item: OrderItem
    table: int
    gid: int
    remaining: int


# ---------------------------------------------------------------------------
# 核心环境
# ---------------------------------------------------------------------------
class RestaurantCore:
    def __init__(self, config: RestaurantConfig):
        self.cfg = config
        self.N = len(config.table_sizes)
        self.max_party = len(config.party_size_probs)
        self.n_dishes = len(config.dishes)
        self.main_ids = [i for i, d in enumerate(config.dishes) if d["category"] == "main"]
        self.dessert_ids = [i for i, d in enumerate(config.dishes) if d["category"] == "dessert"]
        self.main_p = self._weights(self.main_ids)
        self.dessert_p = self._weights(self.dessert_ids) if self.dessert_ids else None
        self.max_cook_time = max(d["cook_time"] for d in config.dishes)
        self.max_price = max(d["price"] for d in config.dishes)
        if config.arrival_schedule:
            self.max_rate = max(r for _, r in config.arrival_schedule)
        else:
            self.max_rate = config.arrival_rate
        self.max_rate = max(self.max_rate, 1e-8)
        self.rng = np.random.default_rng()

    def _weights(self, ids):
        w = np.array([self.cfg.dishes[i].get("weight", 1.0) for i in ids], dtype=float)
        return w / w.sum()

    # ------------------------------------------------------------------
    # 动作编号
    # ------------------------------------------------------------------
    # 厨师：0 空闲/继续 | 1..N 开始做第 k 号桌的一道菜 | N+1 洗一批盘子
    @property
    def n_chef_actions(self) -> int:
        return self.N + 2

    # 服务员：0 空闲/继续 | 带位 1..N | 点单 N+1..2N | 上菜 2N+1..3N | 清桌 3N+1..4N
    @property
    def n_server_actions(self) -> int:
        return 4 * self.N + 1

    def chef_action_name(self, a: int) -> str:
        if a == 0:
            return "空闲"
        if a <= self.N:
            return f"做{a - 1}号桌的菜"
        return "洗盘子"

    def server_action_name(self, a: int) -> str:
        if a == 0:
            return "空闲"
        kind, i = divmod(a - 1, self.N)
        return f"{['带位到', '点单', '上菜到', '清理'][kind]}{i}号桌"

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.t = 0
        self.tables = [Table(capacity=c) for c in cfg.table_sizes]
        self.queue: list[Group] = []
        self.clean_plates = cfg.n_plates - cfg.initial_dirty_plates
        self.dirty_plates = cfg.initial_dirty_plates   # 洗碗池里的脏盘子
        self.stoves: list[StoveTask] = []
        self.chef_busy = 0           # 厨师剩余锁定步数（备菜或洗碗）
        self.washing = 0             # 正在洗的盘子数
        self.server_busy = 0
        self._next_gid = 0
        self.stats = {
            "revenue": 0.0, "bill_total": 0.0, "tip_total": 0.0,
            "groups_paid": 0, "customers_paid": 0,
            "sum_tip_rate": 0.0,
            "walkaway_groups": 0, "walkaway_people": 0,
            "walkaway_queue_groups": 0, "walkaway_table_groups": 0,
            "balked_groups": 0, "balked_people": 0,
            "arrived_groups": 0, "arrived_people": 0,
            "dessert_phases": 0,      # 进入甜点阶段的组数
            "dessert_served": 0,      # 吃完甜点的组数
            "dessert_abandoned": 0,   # 甜点阶段等不及离开的组数
            "dishes_delivered": 0, "cold_dishes": 0, "wasted_dishes": 0,
            "seats_offered": 0, "seats_used": 0,   # 入座时的桌子容量与实际人数，算座位利用率
            "invalid_chef": 0, "invalid_server": 0,
            "plate_shortage_steps": 0, "pass_full_steps": 0, "stoves_full_steps": 0,
            "chef_idle_steps": 0, "server_idle_steps": 0,
            "sum_wait_queue": 0.0, "n_wait_queue": 0,   # 到达 → 入座
            "sum_wait_order": 0.0, "n_wait_order": 0,   # 入座 → 点主菜
            "sum_wait_main": 0.0, "n_wait_main": 0,     # 到达 → 主菜上齐（小费用的等待）
        }
        self._arrivals(self._new_events())  # 开局先来一批顾客，避免前几步空转
        return self.get_state()

    def step(self, chef_action: int, server_action: int):
        """返回 (state, reward, terminated, truncated, info)。
        terminated 恒为 False（餐厅不会"失败"），时间到了 truncated=True。"""
        ev = self._new_events()
        self._apply_chef(int(chef_action), ev)
        self._apply_server(int(server_action), ev)
        self._tick_kitchen(ev)
        self._tick_server()
        self._tick_pass()
        self._tick_eating(ev)
        self._tick_patience(ev)
        self._arrivals(ev)
        self._update_step_stats()

        self.t += 1
        reward = compute_reward(ev, self.cfg.reward)
        truncated = self.t >= self.cfg.episode_length
        info = {"events": ev, **self.summary()}
        return self.get_state(), reward, False, truncated, info

    @staticmethod
    def _new_events() -> dict:
        ev = {k: 0 for k in [
            "groups_paid", "walkaway_groups", "walkaway_people", "balked_groups",
            "balked_people", "dessert_abandoned", "invalid_chef", "invalid_server",
            "seated", "order_taken", "dish_cooked", "dish_delivered", "cold_delivered",
            "plate_washed", "table_cleared", "wasted",
        ]}
        ev["bill"] = 0.0
        ev["tip"] = 0.0
        return ev

    # ------------------------------------------------------------------
    # 到达率
    # ------------------------------------------------------------------
    def arrival_rate_at(self, t: float) -> float:
        sched = self.cfg.arrival_schedule
        if not sched:
            return self.cfg.arrival_rate
        xs = [p[0] for p in sched]
        ys = [p[1] for p in sched]
        return float(np.interp(t, xs, ys))

    # ------------------------------------------------------------------
    # 合法动作
    # ------------------------------------------------------------------
    def chef_mask(self) -> np.ndarray:
        m = np.zeros(self.n_chef_actions, dtype=bool)
        m[0] = True
        if self.chef_busy > 0:      # 忙碌中只能继续
            return m
        for k in range(self.N):
            m[1 + k] = self._can_cook(k)
        m[self.N + 1] = self.dirty_plates > 0
        return m

    def server_mask(self) -> np.ndarray:
        m = np.zeros(self.n_server_actions, dtype=bool)
        m[0] = True
        if self.server_busy > 0:
            return m
        N = self.N
        for i, tb in enumerate(self.tables):
            m[1 + i] = tb.state == EMPTY and self._first_fitting_group(tb.capacity) is not None
            m[1 + N + i] = tb.state == WAIT_ORDER
            m[1 + 2 * N + i] = tb.state == WAIT_FOOD and any(
                it.status == DISH_READY for it in tb.items)
            m[1 + 3 * N + i] = tb.state == DIRTY
        return m

    def _can_cook(self, k: int) -> bool:
        tb = self.tables[k]
        return (
            tb.state == WAIT_FOOD
            and any(it.status == DISH_PENDING for it in tb.items)
            and self.clean_plates > 0
            and len(self.stoves) < self.cfg.n_stoves
            and self._pass_count() + len(self.stoves) < self.cfg.pass_capacity
        )

    def _pass_count(self) -> int:
        return sum(it.status == DISH_READY for tb in self.tables for it in tb.items)

    def _first_fitting_group(self, capacity: int) -> Optional[int]:
        """队列里最早到达、且能坐进这张桌子的一组（返回队列下标）。"""
        for idx, g in enumerate(self.queue):
            if g.size <= capacity:
                return idx
        return None

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------
    def _apply_chef(self, a: int, ev: dict):
        self._chef_active = self.chef_busy > 0   # 本步厨师是否在干活（用于统计空闲）
        if self.chef_busy > 0 or a == 0:
            return
        if not self.chef_mask()[a]:
            ev["invalid_chef"] += 1
            self.stats["invalid_chef"] += 1
            return
        cfg = self.cfg
        self._chef_active = True
        if a <= self.N:
            k = a - 1
            tb = self.tables[k]
            # 同一桌里先做最费时的菜，让一桌的菜尽量同时出
            pending = [it for it in tb.items if it.status == DISH_PENDING]
            item = max(pending, key=lambda it: cfg.dishes[it.dish]["cook_time"])
            item.status = DISH_COOKING
            self.clean_plates -= 1
            self.stoves.append(StoveTask(item, k, tb.group.gid, cfg.dishes[item.dish]["cook_time"]))
            self.chef_busy = cfg.prep_time
        else:
            n = min(cfg.wash_batch, self.dirty_plates)
            self.dirty_plates -= n
            self.washing = n
            self.chef_busy = cfg.wash_time

    def _apply_server(self, a: int, ev: dict):
        self._server_active = self.server_busy > 0
        if self.server_busy > 0 or a == 0:
            return
        if not self.server_mask()[a]:
            ev["invalid_server"] += 1
            self.stats["invalid_server"] += 1
            return
        self._server_active = True
        kind, i = divmod(a - 1, self.N)
        tb = self.tables[i]
        cfg = self.cfg
        if kind == 0:    # 带位：队列中最早到达、且坐得下的一组
            g = self.queue.pop(self._first_fitting_group(tb.capacity))
            g.seated_t = self.t
            self._record_wait("queue", self.t - g.arrival_t)
            self.stats["seats_offered"] += tb.capacity
            self.stats["seats_used"] += g.size
            tb.state, tb.group, tb.phase = WAIT_ORDER, g, PHASE_MAIN
            ev["seated"] += 1
            self.server_busy = cfg.seat_time
        elif kind == 1:  # 点单：主菜阶段每人一道主菜；甜点阶段按之前决定的份数点甜点
            g = tb.group
            if tb.phase == PHASE_MAIN:
                dishes = self.rng.choice(self.main_ids, size=g.size, p=self.main_p)
                g.ordered_t = self.t
                self._record_wait("order", self.t - g.seated_t)
            else:
                dishes = self.rng.choice(self.dessert_ids, size=g.dessert_count, p=self.dessert_p)
            tb.items = [OrderItem(int(d)) for d in dishes]
            tb.state = WAIT_FOOD
            ev["order_taken"] += 1
            self.server_busy = cfg.order_time
        elif kind == 2:  # 上菜：把这桌在出餐口上的菜全部端走
            self._deliver(tb, ev)
            self.server_busy = cfg.deliver_time
        else:            # 清桌：桌上的盘子全部送回洗碗池
            self.dirty_plates += tb.plates
            self._reset_table(tb)
            ev["table_cleared"] += 1
            self.server_busy = cfg.clear_time

    def _deliver(self, tb: Table, ev: dict):
        cfg, g = self.cfg, tb.group
        for it in tb.items:
            if it.status != DISH_READY:
                continue
            it.status = DISH_DELIVERED
            tb.plates += 1
            g.bill += cfg.dishes[it.dish]["price"]
            g.n_delivered += 1
            ev["dish_delivered"] += 1
            self.stats["dishes_delivered"] += 1
            if it.pass_wait > cfg.cold_after:
                g.n_cold += 1
                ev["cold_delivered"] += 1
                self.stats["cold_dishes"] += 1
        if all(it.status == DISH_DELIVERED for it in tb.items):  # 上齐了才开始吃
            tb.state = EATING
            if tb.phase == PHASE_MAIN:
                g.main_wait = self.t - g.arrival_t
                self._record_wait("main", g.main_wait)
                lo, hi = cfg.eat_time_min, cfg.eat_time_max
            else:
                g.dessert_wait = self.t - g.dessert_start_t
                lo, hi = cfg.dessert_eat_time_min, cfg.dessert_eat_time_max
            tb.eat_remaining = int(self.rng.integers(lo, hi + 1))

    # ------------------------------------------------------------------
    # 时间推进
    # ------------------------------------------------------------------
    def _tick_kitchen(self, ev: dict):
        # 灶台
        still = []
        for task in self.stoves:
            task.remaining -= 1
            if task.remaining > 0:
                still.append(task)
                continue
            tb = self.tables[task.table]
            alive = (tb.group is not None and tb.group.gid == task.gid
                     and any(it is task.item for it in tb.items))
            if alive:
                task.item.status = DISH_READY
                task.item.pass_wait = 0
                ev["dish_cooked"] += 1
            else:  # 顾客已经走了，菜作废，盘子变脏
                self.dirty_plates += 1
                ev["wasted"] += 1
                self.stats["wasted_dishes"] += 1
        self.stoves = still
        # 厨师锁定
        if self.chef_busy > 0:
            self.chef_busy -= 1
            if self.chef_busy == 0 and self.washing > 0:
                self.clean_plates += self.washing
                ev["plate_washed"] += self.washing
                self.washing = 0

    def _tick_server(self):
        if self.server_busy > 0:
            self.server_busy -= 1

    def _tick_pass(self):
        for tb in self.tables:
            for it in tb.items:
                if it.status == DISH_READY:
                    it.pass_wait += 1

    def _tick_eating(self, ev: dict):
        cfg = self.cfg
        for tb in self.tables:
            if tb.state != EATING:
                continue
            tb.eat_remaining -= 1
            if tb.eat_remaining > 0:
                continue
            g = tb.group
            if tb.phase == PHASE_MAIN:
                n_dessert = int(self.rng.binomial(g.size, cfg.dessert_prob)) if self.dessert_ids else 0
                if n_dessert > 0:   # 进入甜点阶段，等服务员来点甜点
                    tb.phase, tb.state, tb.items = PHASE_DESSERT, WAIT_ORDER, []
                    g.dessert_count = n_dessert
                    g.dessert_start_t = self.t
                    g.patience = g.patience_init = cfg.dessert_patience
                    self.stats["dessert_phases"] += 1
                    continue
                self._pay(tb, ev, include_dessert=False)
            else:
                self.stats["dessert_served"] += 1
                self._pay(tb, ev, include_dessert=True)

    def _tick_patience(self, ev: dict):
        cfg = self.cfg
        kept = []
        for g in self.queue:    # 门口排队的顾客
            g.patience -= cfg.patience_decay_queue
            if g.patience <= 0:
                self._count_walkaway(g, ev)
                self.stats["walkaway_queue_groups"] += 1
            else:
                kept.append(g)
        self.queue = kept
        for tb in self.tables:  # 已入座的顾客
            if tb.state == WAIT_ORDER:
                tb.group.patience -= cfg.patience_decay_order
            elif tb.state == WAIT_FOOD:
                tb.group.patience -= cfg.patience_decay_food
            else:
                continue
            if tb.group.patience > 0:
                continue
            self._discard_ready(tb, ev)  # 出餐口上这桌的菜作废；灶台上的做完时作废
            if tb.phase == PHASE_DESSERT:
                # 甜点等不及：付主菜的钱离开（小费只按主菜阶段的等待算）
                ev["dessert_abandoned"] += 1
                self.stats["dessert_abandoned"] += 1
                self._pay(tb, ev, include_dessert=False)
            else:
                self._count_walkaway(tb.group, ev)
                self.stats["walkaway_table_groups"] += 1
                if tb.plates > 0:  # 已经上了部分菜，桌上有盘子要清理
                    tb.state, tb.group, tb.items = DIRTY, None, []
                else:
                    self._reset_table(tb)

    def _arrivals(self, ev: dict):
        cfg = self.cfg
        n = self.rng.poisson(self.arrival_rate_at(self.t))
        for _ in range(n):
            size = int(self.rng.choice(self.max_party, p=cfg.party_size_probs)) + 1
            self.stats["arrived_groups"] += 1
            self.stats["arrived_people"] += size
            if len(self.queue) >= cfg.max_queue:
                ev["balked_groups"] += 1
                ev["balked_people"] += size
                self.stats["balked_groups"] += 1
                self.stats["balked_people"] += size
                continue
            p = float(self.rng.uniform(cfg.patience_min, cfg.patience_max))
            self.queue.append(Group(self._next_gid, size, p, p, self.t))
            self._next_gid += 1

    def _update_step_stats(self):
        s = self.stats
        if self.clean_plates == 0 and any(
            it.status == DISH_PENDING for tb in self.tables for it in tb.items
        ):
            s["plate_shortage_steps"] += 1
        if self._pass_count() >= self.cfg.pass_capacity:
            s["pass_full_steps"] += 1
        if len(self.stoves) >= self.cfg.n_stoves:
            s["stoves_full_steps"] += 1
        if not self._chef_active:      # 本步既没在忙、也没开始新动作
            s["chef_idle_steps"] += 1
        if not self._server_active:
            s["server_idle_steps"] += 1

    # ------------------------------------------------------------------
    # 付款与小费
    # ------------------------------------------------------------------
    def tip_rate(self, wait: float, cold_frac: float) -> float:
        cfg = self.cfg
        if wait <= cfg.tip_grace_wait:
            w = 1.0
        else:
            w = max(0.0, 1.0 - (wait - cfg.tip_grace_wait) / (cfg.tip_zero_wait - cfg.tip_grace_wait))
        return cfg.tip_max_rate * w * (1.0 - cfg.cold_tip_penalty * cold_frac)

    def _pay(self, tb: Table, ev: dict, include_dessert: bool):
        g = tb.group
        wait = g.main_wait + (g.dessert_wait if include_dessert else 0.0)
        cold_frac = g.n_cold / g.n_delivered if g.n_delivered else 0.0
        rate = self.tip_rate(wait, cold_frac)
        tip = g.bill * rate
        ev["bill"] += g.bill
        ev["tip"] += tip
        ev["groups_paid"] += 1
        s = self.stats
        s["revenue"] += g.bill + tip
        s["bill_total"] += g.bill
        s["tip_total"] += tip
        s["groups_paid"] += 1
        s["customers_paid"] += g.size
        s["sum_tip_rate"] += rate
        tb.state, tb.group, tb.items = DIRTY, None, []   # 盘子留在桌上，等服务员清理

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _count_walkaway(self, g: Group, ev: dict):
        ev["walkaway_groups"] += 1
        ev["walkaway_people"] += g.size
        self.stats["walkaway_groups"] += 1
        self.stats["walkaway_people"] += g.size

    def _discard_ready(self, tb: Table, ev: dict):
        for it in tb.items:
            if it.status == DISH_READY:
                self.dirty_plates += 1
                ev["wasted"] += 1
                self.stats["wasted_dishes"] += 1

    @staticmethod
    def _reset_table(tb: Table):
        tb.state, tb.group, tb.phase, tb.items = EMPTY, None, PHASE_MAIN, []
        tb.plates, tb.eat_remaining = 0, 0

    def _record_wait(self, name: str, dt: float):
        self.stats[f"sum_wait_{name}"] += dt
        self.stats[f"n_wait_{name}"] += 1

    def plates_in_use(self) -> int:
        """灶台、正在洗、出餐口、桌上的盘子数（用于检查盘子守恒）。"""
        return len(self.stoves) + self.washing + self._pass_count() + sum(
            tb.plates for tb in self.tables)

    def summary(self) -> dict:
        """评测用的汇总指标（本回合累计）。"""
        s = self.stats
        avg = lambda k: s[f"sum_wait_{k}"] / s[f"n_wait_{k}"] if s[f"n_wait_{k}"] else 0.0
        return {
            "t": self.t,
            "revenue": s["revenue"],
            "bill_total": s["bill_total"],
            "tip_total": s["tip_total"],
            "avg_tip_rate": s["sum_tip_rate"] / s["groups_paid"] if s["groups_paid"] else 0.0,
            "groups_paid": s["groups_paid"],
            "customers_paid": s["customers_paid"],
            "walkaway_groups": s["walkaway_groups"],
            "walkaway_people": s["walkaway_people"],
            "walkaway_queue_groups": s["walkaway_queue_groups"],
            "walkaway_table_groups": s["walkaway_table_groups"],
            "balked_groups": s["balked_groups"],
            "balked_people": s["balked_people"],
            "lost_people": s["walkaway_people"] + s["balked_people"],
            "arrived_groups": s["arrived_groups"],
            "arrived_people": s["arrived_people"],
            "dessert_phases": s["dessert_phases"],
            "dessert_served": s["dessert_served"],
            "dessert_abandoned": s["dessert_abandoned"],
            "dishes_delivered": s["dishes_delivered"],
            "cold_dishes": s["cold_dishes"],
            "cold_rate": s["cold_dishes"] / s["dishes_delivered"] if s["dishes_delivered"] else 0.0,
            "wasted_dishes": s["wasted_dishes"],
            "seat_utilization": s["seats_used"] / s["seats_offered"] if s["seats_offered"] else 0.0,
            "invalid_chef": s["invalid_chef"],
            "invalid_server": s["invalid_server"],
            "plate_shortage_steps": s["plate_shortage_steps"],
            "pass_full_steps": s["pass_full_steps"],
            "stoves_full_steps": s["stoves_full_steps"],
            "chef_idle_steps": s["chef_idle_steps"],
            "server_idle_steps": s["server_idle_steps"],
            "avg_wait_queue": avg("queue"),
            "avg_wait_order": avg("order"),
            "avg_wait_main": avg("main"),
        }

    # ------------------------------------------------------------------
    # 状态（原始字典，由各接口自行转成观测向量）
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        cfg = self.cfg
        tables = []
        for tb in self.tables:
            g = tb.group
            st = [it.status for it in tb.items]
            pending_time = sum(cfg.dishes[it.dish]["cook_time"]
                               for it in tb.items if it.status == DISH_PENDING)
            ready_waits = [it.pass_wait for it in tb.items if it.status == DISH_READY]
            if g is None:
                wait_so_far = 0.0
            elif tb.phase == PHASE_MAIN:
                wait_so_far = self.t - g.arrival_t if tb.state in (WAIT_ORDER, WAIT_FOOD) else g.main_wait
            else:
                wait_so_far = g.main_wait + (self.t - g.dessert_start_t
                                             if tb.state in (WAIT_ORDER, WAIT_FOOD) else g.dessert_wait)
            tables.append({
                "capacity": tb.capacity,
                "state": tb.state,
                "phase": tb.phase,
                "party_size": g.size if g else 0,
                "patience_frac": g.patience / g.patience_init if g else 0.0,
                "eat_remaining": tb.eat_remaining,
                "n_items": len(tb.items),
                "n_pending": st.count(DISH_PENDING),
                "n_cooking": st.count(DISH_COOKING),
                "n_ready": st.count(DISH_READY),
                "n_delivered": st.count(DISH_DELIVERED),
                "pending_cook_time": pending_time,
                "max_pass_wait": max(ready_waits) if ready_waits else 0,
                "wait_so_far": wait_so_far,
                "plates": tb.plates,
            })
        return {
            "t": self.t,
            "arrival_rate": self.arrival_rate_at(self.t),
            "tables": tables,
            "queue": [{"size": g.size, "patience_frac": g.patience / g.patience_init}
                      for g in self.queue],
            "clean_plates": self.clean_plates,
            "dirty_plates": self.dirty_plates,
            "pass_count": self._pass_count(),
            "stoves": [{"table": s.table, "remaining": s.remaining} for s in self.stoves],
            "chef_busy": self.chef_busy,
            "chef_washing": self.washing > 0,
            "server_busy": self.server_busy,
        }

    def render_text(self) -> str:
        cfg = self.cfg
        q = " ".join(f"{g.size}人" for g in self.queue) or "无"
        lines = [
            f"t={self.t:3d}  到达率={self.arrival_rate_at(self.t):.2f}  排队: {q}",
            f"  干净盘={self.clean_plates} 脏盘={self.dirty_plates} 洗碗中={self.washing}  "
            f"出餐口={self._pass_count()}/{cfg.pass_capacity}  "
            f"灶台={len(self.stoves)}/{cfg.n_stoves}",
            f"  厨师: {'忙(' + str(self.chef_busy) + ')' if self.chef_busy else '空闲'}   "
            f"服务员: {'忙(' + str(self.server_busy) + ')' if self.server_busy else '空闲'}",
        ]
        sym = {DISH_PENDING: "待", DISH_COOKING: "做", DISH_READY: "出", DISH_DELIVERED: "上"}
        for i, tb in enumerate(self.tables):
            s = f"  桌{i}({tb.capacity}座): {TABLE_STATE_NAMES[tb.state]}"
            if tb.group:
                s += f" {tb.group.size}人 耐心={tb.group.patience:.1f}"
                if tb.phase == PHASE_DESSERT:
                    s += " [甜点]"
            if tb.items:
                s += " " + " ".join(f"{cfg.dishes[it.dish]['name']}({sym[it.status]})"
                                    for it in tb.items)
            if tb.plates:
                s += f" 盘={tb.plates}"
            lines.append(s)
        return "\n".join(lines)
