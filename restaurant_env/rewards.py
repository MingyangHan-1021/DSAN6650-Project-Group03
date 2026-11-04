"""
Reward function.

core.step() collects event counts `ev` for each step; this turns them into one scalar.
Both agents share the same team reward.

The intermediate (shaping) rewards are event-based and can be exploited, e.g. by
repeatedly adding food to a plate and throwing it away. To prevent this, every food
item thrown in the trash costs `waste_penalty`, which by default is larger than any
single shaping reward, so such loops have a negative net reward.
"""


def compute_reward(ev: dict, rcfg: dict) -> float:
    r = 0.0

    # Main reward: price of delivered dishes + tip when an order is completed
    r += rcfg["revenue_scale"] * (ev["bill"] + ev["tip"])

    # Penalties
    r -= rcfg["expire_penalty"] * ev["orders_expired"]
    r -= rcfg["burn_penalty"] * ev["burnt"]
    r -= rcfg["waste_penalty"] * ev["wasted"]
    r -= rcfg["invalid_action_penalty"] * ev["invalid"]
    r -= rcfg["failed_action_penalty"] * ev["failed"]
    r -= rcfg["step_penalty"]

    # Intermediate (shaping) rewards
    sh = rcfg["shaping"]
    if sh.get("enabled", False):
        r += sh["chop"] * ev["chop"]
        r += sh["fry"] * ev["fry"]
        r += sh["cooker_start"] * ev["cooker_start"]
        r += sh["cooker_collect"] * ev["cooker_collect"]
        r += sh["plate_add"] * ev["plate_add"]
        r += sh["dish_delivered"] * ev["dish_delivered"]

    return float(r)
