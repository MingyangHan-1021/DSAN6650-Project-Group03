"""Checks for the joint-action adapter used by the centralized DQN."""

import numpy as np

from restaurant_env import CentralizedKitchenEnv
from scripts.train_ddqn import decode_action, joint_mask


def test_joint_mask_matches_environment_and_excludes_conflicts():
    env = CentralizedKitchenEnv(config_path="configs/env/medium.yaml")
    env.reset(seed=0)
    n = int(env.action_space.nvec[0])
    mask = joint_mask(env.action_masks(), n)
    assert mask.shape == (n * n,)
    assert mask[0]  # both wait
    assert not mask[1 * n + 1]  # same station claimed twice
    for action in np.flatnonzero(mask):
        a0, a1 = decode_action(action, n)
        assert env.core.action_mask(0)[a0]
        assert env.core.action_mask(1)[a1]
        assert a0 == 0 or a0 != a1


def test_joint_mask_respects_busy_agent():
    env = CentralizedKitchenEnv(config_path="configs/env/medium.yaml")
    env.reset(seed=0)
    tomato = env.core.layout.id["tomato_bin"] + 1
    env.step([tomato, 0])
    n = int(env.action_space.nvec[0])
    legal = np.flatnonzero(joint_mask(env.action_masks(), n))
    assert all(decode_action(a, n)[0] == 0 for a in legal)
