"""
奖励函数。

core.step() 每一步收集一份事件计数 ev（账单、小费、上菜数、顾客流失人数等），
这里把事件换算成一个标量奖励。两个智能体共享同一份团队奖励。

单独放一个文件的好处：做消融实验时只改这里或改 YAML，不动环境规则。

注意：这里的中间奖励是"事件型"的，不是 Ng et al. (1999) 的基于势能的塑形，
理论上可能改变最优策略（例如智能体学会只点单不做菜）。
如果需要理论保证，可以改成 F = gamma * Phi(s') - Phi(s) 的形式，
Phi 可以取"已点单桌数 + 已出餐数"之类的进度量。
"""


def compute_reward(ev: dict, rcfg: dict) -> float:
    r = 0.0

    # 主奖励：顾客付款（账单 + 小费）。小费随等待时间和凉菜比例下降，
    # 所以这一项同时鼓励"多接待"和"快服务"
    r += rcfg["revenue_scale"] * (ev["bill"] + ev["tip"])

    # 惩罚：按人数计，大桌流失损失更大
    r -= rcfg["walkaway_penalty"] * ev["walkaway_people"]
    r -= rcfg["balk_penalty"] * ev["balked_people"]
    r -= rcfg["dessert_abandon_penalty"] * ev["dessert_abandoned"]
    r -= rcfg["invalid_action_penalty"] * (ev["invalid_chef"] + ev["invalid_server"])
    r -= rcfg["step_penalty"]

    # 中间奖励（奖励塑形）
    sh = rcfg["shaping"]
    if sh.get("enabled", False):
        r += sh["seated"] * ev["seated"]
        r += sh["order_taken"] * ev["order_taken"]
        r += sh["dish_cooked"] * ev["dish_cooked"]
        r += sh["dish_delivered"] * ev["dish_delivered"]
        r += sh["plate_washed"] * ev["plate_washed"]
        r += sh["table_cleared"] * ev["table_cleared"]

    return float(r)
