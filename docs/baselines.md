# 阶段 1：基线（随机 + 规则策略）

对应文档里的"阶段 1：基线"。两个基线现在都可以用一条命令跑完三档难度 × 20 个种子，
结果写入 `results/csv/`，并附一张汇总表。

```bash
# 两个基线一起跑，20 种子 × easy / medium / hard
python scripts/baselines.py --policy both \
    --config configs/env/easy.yaml configs/env/medium.yaml configs/env/hard.yaml --seeds 0-19
```

产出：

| 文件 | 内容 |
|---|---|
| `results/csv/baselines_random.csv` | 每个种子的完整指标（34 列） |
| `results/csv/baselines_rule.csv` | 同上 |
| `results/csv/baselines_summary.csv` | 长表：config / policy / metric / mean / std |

评测种子统一用 0–19，和文档"规则"一节一致；规则策略完全确定性（同种子两次跑结果相同，
`tests/test_baselines.py` 里有一条测试盯着这点）。

## 两个策略是什么

**random**（`--policy random`）：每一步从合法动作里均匀随机选。它只用来当下界——
如果随机都解得出，说明环境太简单。

**rule**（`--policy rule`）：手写的菜谱任务链调度器，不是打分器。一道菜被拆成站点访问链：

```
prep      burger: [beef_bin -> pan -> counter]   [lettuce_bin -> board -> counter]
assemble  burger: [plate_rack -> bun_bin -> counter(beef) -> counter(lettuce) -> park]
serve     burger: [counter(做好的盘子) -> pass]
```

两个智能体从共享任务池领活，规则如下：

- 同时最多推进 2 道菜（按 deadline 排序；试过 3、4，更差——柜台只有 3 个，铺开只会互相堵）
- 只有"所有配料都已就位（在柜台 / 在出餐口 / 能从料箱直接上盘）"时才取盘子；提前取盘会占死 6 个盘子里的一个，灶台也会因为没人收而烧糊
- 灶台按 plan 分配（一碗汤的两个番茄进同一口锅），完成的锅优先级最高，因为 8 步后就糊
- split 厨房里一条链跨两个区域，会在柜台处切成两段交给不同的人（助手做到柜台、厨师接手），
  所以 hard 的 handoff 是 10.35 次/局
- 每个动作执行完**先校验后置条件再推进**（`take` 真的拿到了吗？`cooker_collect` 真的收到盘子里了吗？），
  失败就从料箱重来或把手里的东西放到柜台——这条是避免死锁的关键
- 加了超时重规划：一个任务等柜台/灶台超过 14 步就放弃整条 plan，重新排

`--solo` 会把所有活交给智能体 0（open 厨房里一人能做完一道菜），用来衡量"协作到底带来多少收益"。

## 结果（seeds 0–19，均值 ± 标准差）

| 难度 | 策略 | 完成订单 | 到达订单 | 需求菜品 | 出菜 | 收入 | 小费率 |
|---|---|---|---|---|---|---|---|
| easy | random | 0.00 ± 0.00 | 7.20 | 8.60 | 0.00 | 0.0 | 0.000 |
| easy | rule | 3.30 ± 1.95 | 7.20 | 8.60 | 4.40 | 64.5 | 0.155 |
| medium | random | 0.10 ± 0.30 | 13.45 | 17.55 | 0.15 | 2.3 | 0.009 |
| medium | rule | 3.90 ± 1.84 | 13.45 | 17.55 | 5.85 | 94.4 | 0.114 |
| hard | random | 0.00 ± 0.00 | 18.20 | 29.20 | 0.00 | 0.0 | 0.000 |
| hard | rule | 0.75 ± 0.94 | 18.20 | 29.20 | 2.05 | 31.4 | 0.057 |

其他指标（medium / rule）：浪费 0.65、烧糊 0.30、handoff 7.10、冲突 0.00、
没有干净盘子的步数 1.4、平均完成时间 56.5 步、agent 0/1 空闲步数 116 / 182（共 600 步中的 50%）。

**能确定的两件事**

1. 随机策略在三档上都约等于 0 单 → 环境不能被瞎蒙解开，这个下界是有效的。
2. 规则策略能稳定出菜（medium 5.85 盘、easy 最好一局 8 盘），环境"在合理配合下做得出菜"这一点成立。

**单人参照（medium）**：`--solo` 只有 2.50 单 / 3.70 盘，比双人（3.90 / 5.85）低。
协作确实带来约 1.6 倍收益，远不到 2 倍——说明当前策略的协调开销不小。

## 难度校准：现在还测不出来

文档要求用规则策略校准三档难度。做了一组敏感性实验（medium，20 种子）：

| 设置 | 完成订单 | 出菜 |
|---|---|---|
| 基线（medium 原配置） | 3.90 | 5.85 |
| 到达率 ×0.7 | 3.60 | 4.95 |
| `burn_after` 8 → 12 | 3.95 | 5.90 |
| 耐心 100–140 → 130–170 | 3.85 | 5.90 |

复现方式（不改 `configs/env/`，只在运行时覆盖）：

```bash
python scripts/baselines.py --policy rule --config configs/env/medium.yaml --seeds 0-19 \
    --arrival-scale 0.7
python scripts/baselines.py --policy rule --config configs/env/medium.yaml --seeds 0-19 \
    --override burn_after=12
python scripts/baselines.py --policy rule --config configs/env/medium.yaml --seeds 0-19 \
    --override orders.patience_min=130 --override orders.patience_max=170
```

**读法**：把需求砍掉 30% 只让出菜少 15%（完成率 33% → 40%），把耐心加长、把烧糊窗口放宽
基本没有效果。也就是说，当前瓶颈是**策略自己的调度效率**（medium 约 51 步/盘，而 README
的设计目标是两人 15–18 步/盘；单人 81 步/盘 vs 设计目标 30–37 步/盘），而不是难度参数。
策略还有约 50% 的 agent-step 在空转。

所以结论是：**在参照策略达到设计节奏之前，调难度参数是测不出来的**。
现在不能据此说"medium 太难"或"太简单"——这也是阶段 1 目前唯一没有闭环的部分。

## 下一步（按优先级）

1. **把规则策略的效率提上去**（它同时是难度标尺和之后"人类水平参照"的底稿），
   已知的最大浪费是空转 50%，其次是一盘菜要经过"放柜台 → 再取回"的双重搬运。
   具体可试：让空闲的智能体去帮下一条 plan 备料、备料直接落到停好的盘子（`food_onto_plate`）、
   减少 plan 之间的等待。
2. 规则策略接近设计节奏后，再把上面三个旋钮重跑一遍，才能给出"三档是否合适"的结论。
3. 之后才进阶段 2（中心化 MaskablePPO / Double DQN）——`scripts/evaluate.py` 会复用
   `baselines.py` 的 CSV 格式和种子约定。

## 已知局限

- 规则策略是**下界参照**，不是上限；它没有做任何搜索或学习。
- `--solo` 只在 open 厨房有意义；split 厨房一个人做不完一道菜，`--solo` 的数字不能直接比较。
- 随机策略的浪费很高（每局 26–46 个食材），这是随机动作的正常表现，不是 bug。
