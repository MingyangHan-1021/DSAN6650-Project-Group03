# Cooperative Kitchen: a Multi-Agent RL Environment

Two agents, a **chef** and an **assistant**, share one kitchen and work together to fill orders: take ingredients from the bins, chop, fry, boil or bake them, plate them, and hand the dishes in at the pass. They share a single team reward and have to divide the work around shared tools, counters and plates.

## Installation

```bash
pip install -r requirements.txt
pytest                # should report 32 passed
```

## Quick start

```python
from restaurant_env import CentralizedKitchenEnv

env = CentralizedKitchenEnv(config_path="configs/env/medium.yaml")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([a0, a1])   # one action per agent
mask = env.action_masks()     # legal actions, e.g. for MaskablePPO
print(env.render())           # text view of the kitchen
```

```bash
# Generate order CSV files from a config
python scripts/generate_orders.py --config configs/env/medium.yaml --seeds 0 1 2 --out data/orders/medium

# Run one episode with random legal actions and save the logs
python scripts/run_random.py --config configs/env/easy.yaml --out results/logs/random_easy
```

## Baselines

```bash
# both baselines, 20 seeds x 3 difficulties -> results/csv/
python scripts/baselines.py --policy both --config configs/env/easy.yaml \
    configs/env/medium.yaml configs/env/hard.yaml --seeds 0-19

# one agent only, as a reference for what cooperation adds (open kitchen)
python scripts/baselines.py --policy rule --config configs/env/medium.yaml --seeds 0-19 --solo

# difficulty experiments: parameters are overridden at run time, configs/env stays untouched
python scripts/baselines.py --policy rule --config configs/env/medium.yaml --seeds 0-19 \
    --arrival-scale 0.7 --override burn_after=12

# theoretical capacity of each difficulty (how many dishes 300 steps can serve at best)
python scripts/capacity.py --config configs/env/easy.yaml configs/env/medium.yaml configs/env/hard.yaml
```

Writes one row per episode (`baselines_random.csv`, `baselines_rule.csv`) plus a long-format
summary (`baselines_summary.csv`: config / policy / metric / mean / std). The rule policy is a
recipe scheduler: it turns a dish into a chain of station visits, keeps two dishes in flight,
hands the chains out to the two agents, and - when the dish needs no cooker - parks a plate on a
counter and carries the components straight onto it. See [docs/baselines.md](docs/baselines.md)
for the method, the numbers and the difficulty calibration.

Seeds 0-19:

| | easy | medium | hard |
|---|---|---|---|
| orders completed (random / rule) | 0.00 / 3.65 | 0.10 / 5.10 | 0.00 / 1.40 |
| orders arrived | 7.20 | 13.45 | 18.20 |
| dishes ordered | 8.60 | 17.55 | 29.20 |
| dishes served (rule) | 4.75 | 7.25 | 3.00 |
| revenue (rule) | 70.7 | 114.9 | 43.6 |
| avg tip rate (rule) | 0.174 | 0.128 | 0.072 |
| wasted / burnt (rule) | 0.00 / 0.25 | 0.15 / 0.35 | 2.45 / 0.80 |

The random policy completes about 0 orders everywhere, so the environment is not solvable by
chance. `scripts/capacity.py` puts the best possible pace at about 16.9 wall steps per dish
(17.8 dishes in 300 steps): easy then demands 47% of that capacity, medium 99% and hard 172%.
Medium is therefore designed right at the limit - only a near-optimal policy keeps up - and hard
is overloaded on purpose. The rule policy runs at 41 steps per dish (about 2.4x the bound), so it
is a lower reference, not an upper bound.

## Project structure

```
restaurant_env/
├── core.py              # simulator core: (state, action) -> new state
├── recipes.py           # loads recipes; checks processing, plating and serving
├── layout.py            # loads the kitchen layout; computes travel times
├── orders.py            # order generation, CSV read/write
├── rewards.py           # events -> reward
├── logger.py            # episode logs (CSV / JSON)
└── centralized_env.py   # Gymnasium interface (centralized version)
data/
├── recipes.json         # recipes
└── layouts/
    ├── open.json        # open kitchen
    └── split.json       # split kitchen
configs/env/             # easy / medium / hard
scripts/
├── baselines.py         # random and rule-based baselines (evaluation + CSV)
├── capacity.py          # theoretical capacity of a difficulty (lower bound)
├── generate_orders.py   # generate order CSV files
└── run_random.py        # run a random episode and save the logs
tests/test_core.py       # rule tests
pytest.ini               # lets pytest import restaurant_env from the project root
```

Rules and data are kept apart: recipes and layouts live in JSON, orders can come from a CSV, and every other parameter is in YAML. `core.py` only advances time.

## Flow

```
order arrives -> take ingredients -> process (chop / fry) -> into pot or oven (cooks on its own)
              -> take a plate and plate the dish -> hand it in at the pass -> tip once the whole order is served
served plates come back dirty to the sink a few steps later -> wash them -> back on the plate rack
```

## Mechanics

**Actions**: each agent picks one action per step. 0 means wait; s+1 means "walk to station s and interact with it". What the interaction does depends on what the agent holds and on the station's state. An action takes travel time + interaction time, and the agent can only choose 0 until it finishes.

**Hands**: an agent holds at most one thing: one food item, or one plate (a plate can hold several items).

**Occupancy**: you cannot choose a station the other agent is walking to or working at. After an action finishes, the agent stays where it is but no longer occupies the station. If both agents choose the same station in the same step, the chef goes first and the assistant's action is dropped for that step (counted as a conflict).

**Movement**: travel time = ⌈Manhattan distance ÷ move speed⌉. On arrival the interaction is checked again; if it no longer applies, the action fails.

**Processing**: the cutting board (chop) and the pan (fry) are active stations; the agent stays there while processing.

**Cookers**: the pot and the oven are passive. Once all inputs are in, they cook on their own and the agent can walk away. If a finished cooker is left too long it **burns**, and someone has to clean it with empty hands.

**Plating**: only items that belong to some dish can go on a plate. Food and a plate can be combined on a counter; with a plate in hand you can also take a bun straight from its bin, or collect the finished dish from a pot or the oven.

**Counters**: each counter holds one thing. Counters are used to prep ahead and to pass things between agents. One agent putting something down and the other picking it up counts as a handoff.

**Orders**: an order has 1–3 dishes and is complete only when all of them are served. It expires at its deadline. Each dish's price is paid as soon as it is served.

**Tips**: when an order is completed, tip = order price × tip rate. The rate is 20% if the order took at most 30 steps, then falls linearly to 0 at 100 steps.

**Plates**: limited in number. A served plate returns to the sink 5 steps later and must be washed before reuse.

**Menu filtering**: the bins only offer ingredients the current menu uses, and the cutting board and pan only perform useful processes.

**Heterogeneity**: the two agents have different skills (see below); a skill set to `null` means that agent cannot do the job. In the split layout each agent can only use the stations in its own zone.

## Actions and observations

There are 19 stations, so each agent has 20 actions and the centralized action space is `MultiDiscrete([20, 20])`. Action k (k ≥ 1) targets station `env.core.layout.names[k-1]`.

| Interaction | Station | Condition |
|---|---|---|
| take ingredient | bin | empty hands |
| ingredient straight onto plate | bin | holding a plate, and the result is a valid partial dish (e.g. the bun for a burger) |
| take plate | plate_rack | empty hands, a clean plate is available |
| pick up / put down | counter | pick up with empty hands, or put down on an empty counter |
| plate | counter | one side is a plate, the other is food, and the result is valid |
| chop / fry | cutting_board / pan | holding a processable item, and the agent has the skill |
| put in | pot / oven | holding food that belongs to one of its recipes |
| collect | pot / oven | finished, holding a plate |
| clean | pot / oven | burnt, empty hands |
| wash plate | sink | empty hands, a dirty plate is waiting, and the agent has the skill |
| throw away | trash | holding food, or holding a non-empty plate (empties it) |
| serve | pass | the plate is exactly a dish that an open order still needs |

The observation is a vector in [0, 1] (300 values for medium) containing:

- each agent's position, target, remaining busy time and item in hand
- the contents of the counters, pots and oven, plus each cooker's status and progress
- the number of plates on the rack, in the sink and on their way back
- the 5 open orders with the earliest deadlines: dishes still needed, time left, time waited
- episode progress and the current order arrival rate

## Recipes (data/recipes.json)

| Dish | How to make it | Tools | Price |
|---|---|---|---|
| Salad | chopped lettuce + chopped tomato, plated | cutting board | 10 |
| Tomato Soup | 2 chopped tomatoes in a pot for 6 steps, collected | cutting board, pot | 12 |
| Burger | fried beef + chopped lettuce + bun, plated | pan, cutting board | 16 |
| Tomato Pasta | noodles in a pot for 5 steps + chopped then fried tomato, plated | pot, cutting board, pan | 18 |
| Pizza | dough + chopped tomato + chopped cheese in the oven for 8 steps, collected | cutting board, oven | 20 |

Tomatoes are in almost every dish, so the cutting board is the main shared bottleneck.

## Kitchen layout (data/layouts)

```
y=0  lettuce_bin  tomato_bin  beef_bin   bun_bin    noodle_bin  dough_bin  cheese_bin
y=2  cutting_board            counter_a  counter_b  counter_c              pan
y=4  plate_rack   sink        trash      pass       oven        pot_1      pot_2
     x=0          x=1         x=2        x=3        x=4         x=5        x=6
```

- **open**: both agents can use every station. The chef starts at counter_c and the assistant at counter_a.
- **split**: the chef can only use pan, pot_1, pot_2, oven and pass; the assistant can only use the bins, cutting_board, plate_rack and sink. Both share the three counters and the trash, so every handoff goes through a counter.

## The two agents

| | Move speed (cells/step) | Chop | Fry | Wash |
|---|---|---|---|---|
| chef | 2 | 3 steps | 2 steps | 3 steps |
| assistant | 2 | 2 steps | 4 steps | 2 steps |

## Parameters (medium)

### Episode and data files

| Parameter | Value | Meaning |
|---|---|---|
| episode_length | 300 | steps per episode |
| recipes_file | data/recipes.json | recipes |
| layout_file | data/layouts/open.json | kitchen layout |
| menu | salad, tomato_soup, burger, pasta | dishes in this episode |

### Plates and cookers

| Parameter | Value | Meaning |
|---|---|---|
| n_plates | 6 | total plates |
| initial_dirty_plates | 0 | dirty plates at the start |
| dirty_return_delay | 5 | steps until a served plate reaches the sink |
| burn_after | 8 | steps a finished cooker can wait before burning |
| cooker_clean_time | 2 | steps to clean a burnt cooker |

### Orders (under `orders:`)

| Parameter | Value | Meaning |
|---|---|---|
| orders_file | null | path to an order CSV; null = generate from the seed |
| arrival_rate | 0.03 | constant rate (orders/step), used when arrival_schedule is null |
| arrival_schedule | see below | time-varying rate |
| dishes_per_order_probs | [0.7, 0.3] | probability of 1 / 2 dishes per order |
| dish_weights | null | per-dish weights; null = uniform |
| patience_min / max | 100 / 140 | steps from arrival until an order expires |
| max_visible_orders | 5 | orders shown in the observation |

Arrival rate:

| Step | 0 | 80 | 120 | 200 | 240 | 300 |
|---|---|---|---|---|---|---|
| Rate (orders/step) | 0.02 | 0.03 | 0.07 | 0.07 | 0.03 | 0.02 |

Steps 120–200 are the rush.

### Tips

| Parameter | Value | Meaning |
|---|---|---|
| tip_max_rate | 0.20 | maximum tip rate |
| tip_grace_wait | 30 | full tip if the order takes at most this many steps |
| tip_zero_wait | 100 | no tip at this many steps or more |

### Reward (under `reward:`)

Shared by both agents:

```
reward = 0.1 × (price of served dishes + tips)
       − 1.0 × expired orders − 0.5 × burnt cookers − 0.1 × food items thrown away
       + shaping rewards
```

| Parameter | Value | Meaning |
|---|---|---|
| revenue_scale | 0.1 | revenue-to-reward scale |
| expire_penalty | 1.0 | per expired order |
| burn_penalty | 0.5 | per burnt cooker |
| waste_penalty | 0.1 | per food item thrown away |
| invalid_action_penalty | 0 | chose an action that is not currently allowed |
| failed_action_penalty | 0 | arrived but the interaction no longer applies |
| step_penalty | 0 | per step |
| shaping.enabled | true | turn shaping rewards on or off |
| shaping.chop / fry | 0.05 / 0.05 | per item chopped / fried |
| shaping.cooker_start | 0.1 | a cooker has all its inputs and starts cooking |
| shaping.cooker_collect | 0.1 | finished dish collected from a cooker |
| shaping.plate_add | 0.05 | per item added to a plate |
| shaping.dish_delivered | 0.3 | per dish served |

The waste penalty (0.1) is larger than any single shaping reward (0.05), so a loop of "put it on the plate, then throw it away" has a negative net reward.

In YAML, `orders` and `reward` only need the entries you want to change; the rest use the defaults.

## Difficulty levels

| | easy | medium | hard |
|---|---|---|---|
| Menu | salad, tomato soup, burger | + tomato pasta | + pizza |
| Layout | open | open | split |
| Arrival rate | constant 0.025 | rush 0.07 | rush 0.09 |
| Dishes per order | 1 / 2: 0.8 / 0.2 | 1 / 2: 0.7 / 0.3 | 1 / 2 / 3: 0.5 / 0.35 / 0.15 |
| Order patience | 120–160 | 100–140 | 90–120 |
| Plates | 8 | 6 | 5 (2 dirty at the start) |
| Dishes per episode (approx.) | 9 | 17 | 29 |

One agent working alone needs about 30–37 steps per dish; two agents working together should manage roughly one dish every 15–18 steps. So easy is relaxed, medium builds a backlog during the rush, and hard is clearly overloaded.

## Order CSV

```
order_id,arrival_t,deadline_t,dishes
0,26,154,burger
1,70,186,pasta+salad
```

`dishes` are joined with `+`. Set `orders.orders_file` in a config and the environment uses that order stream in every episode.

## Logs

`EpisodeLogger` saves one episode into a folder:

| File | Contents |
|---|---|
| events.csv | one row per event: t, agent, event, station, detail (take, chop, handoff, serve, order completed / expired, burnt, ...) |
| actions.csv | both agents' actions and the reward at every step |
| frames.jsonl | one full state snapshot per step, for replay |
| summary.json | episode summary metrics |

## Evaluation metrics (returned in `info`, cumulative over the episode)

| Metric | Meaning |
|---|---|
| revenue / bill_total / tip_total | total revenue / dish prices / tips |
| avg_tip_rate | average tip rate |
| orders_arrived / completed / expired | orders that arrived / were completed / expired |
| avg_completion_time | average time to complete an order |
| dishes_delivered | dishes served |
| burnt / wasted | burnt cookers / food items thrown away |
| handoffs | handoffs (one agent puts something on a counter, the other picks it up) |
| conflicts | times both agents chose the same station in the same step |
| avg_prepped_on_counter | average number of processed items waiting on counters (prepping ahead) |
| no_clean_plate_steps | steps with no clean plate on the rack |
| avg_open_orders | average number of open orders (backlog) |
| a0_ / a1_ idle / move / work_steps | idle / walking / interacting steps per agent |
| a0_ / a1_ actions | count of each interaction per agent (division of labor) |
| a0_ / a1_ invalid / failed | invalid actions / failed actions per agent |

`info["events"]` holds the event counts for the current step only, which is useful for plotting how things change over time.
