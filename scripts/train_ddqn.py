"""Masked centralized Double DQN for the cooperative kitchen.

Run from the project root::

    python scripts/train_ddqn.py --config configs/env/medium.yaml --seed 0

The online network selects the best legal next action; the target network
evaluates it. Joint actions are flattened as ``a0 * n_actions_per_agent + a1``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restaurant_env import CentralizedKitchenEnv  # noqa: E402
from scripts.evaluate import evaluate  # noqa: E402


def joint_mask(action_masks: np.ndarray, n: int) -> np.ndarray:
    """Expand two per-agent masks into the legal flattened joint actions."""
    masks = np.asarray(action_masks, dtype=bool)
    if masks.shape != (2 * n,):
        raise ValueError(f"expected {2 * n} mask entries, got {masks.shape}")
    legal = masks[:n, None] & masks[None, n:]
    indices = np.arange(1, n)
    legal[indices, indices] = False  # avoid dropping agent 1's action on conflicts
    return legal.reshape(-1)


def decode_action(action: int, n: int) -> tuple[int, int]:
    return divmod(int(action), n)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.capacity = capacity
        self.states = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.next_masks = np.empty((capacity, n_actions), dtype=bool)
        self.size = 0
        self.cursor = 0

    def add(self, state, action, reward, next_state, done, next_mask):
        i = self.cursor
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.next_masks[i] = next_mask
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(self.size, size=batch_size)
        return (self.states[indices], self.actions[indices], self.rewards[indices],
                self.next_states[indices], self.dones[indices], self.next_masks[indices])


def main() -> None:
    import torch
    from torch import nn
    from torch.nn import functional as F

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env/medium.yaml")
    parser.add_argument("--algo-config", default="configs/algo/ddqn.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    with open(args.algo_config, encoding="utf-8") as handle:
        hp = yaml.safe_load(handle)
    total = args.timesteps or int(hp["total_timesteps"])
    eval_seeds = [int(s) for s in hp["eval_seeds"]]
    if not eval_seeds:
        raise ValueError("eval_seeds must contain at least one seed")
    name = f"ddqn_{Path(args.config).stem}_seed{args.seed}"
    model_path = Path("models") / f"{name}.pt"
    run_dir = Path("runs") / name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    env = CentralizedKitchenEnv(config_path=args.config)
    env.action_space.seed(args.seed)
    n = int(env.action_space.nvec[0])
    obs_dim = int(env.observation_space.shape[0])
    hidden = int(hp["hidden_size"])

    def make_network():
        return nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                             nn.Linear(hidden, hidden), nn.ReLU(),
                             nn.Linear(hidden, n * n)).to(device)

    online = make_network()
    target = make_network()
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=float(hp["learning_rate"]))
    replay = ReplayBuffer(int(hp["buffer_size"]), obs_dim, n * n)
    gamma = float(hp["gamma"])
    learn_at = int(hp["learning_starts"])

    def greedy(obs, mask):
        with torch.no_grad():
            values = online(torch.as_tensor(obs, device=device, dtype=torch.float32))
            values = values.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf)
            return int(values.argmax().item())

    def policy(obs, action_masks):
        return decode_action(greedy(obs, joint_mask(action_masks, n)), n)

    def save_model():
        torch.save({"state_dict": online.state_dict(), "obs_dim": obs_dim,
                    "n": n, "hidden_size": hidden, "train_seed": args.seed}, model_path)

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_count = 0
    best_reward = -np.inf
    curve_path = run_dir / "curve.csv"
    try:
        with curve_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "step", "episode", "episode_reward", "orders_completed", "td_loss",
                "epsilon", "eval_mean_reward",
            ])
            writer.writeheader()
            last_loss = float("nan")
            for step in range(1, total + 1):
                mask = joint_mask(env.action_masks(), n)
                anneal = min(1.0, step / max(1, total * float(hp["exploration_fraction"])))
                epsilon = float(hp["epsilon_start"]) + anneal * (
                    float(hp["epsilon_end"]) - float(hp["epsilon_start"]))
                if rng.random() < epsilon:
                    action = int(rng.choice(np.flatnonzero(mask)))
                else:
                    action = greedy(obs, mask)
                next_obs, reward, terminated, truncated, info = env.step(decode_action(action, n))
                done = terminated or truncated
                next_mask = joint_mask(env.action_masks(), n)
                replay.add(obs, action, reward, next_obs, done, next_mask)
                episode_reward += reward
                obs = next_obs

                if step >= learn_at and step % int(hp["train_frequency"]) == 0 \
                        and replay.size >= int(hp["batch_size"]):
                    states, actions, rewards, next_states, dones, next_masks = replay.sample(
                        int(hp["batch_size"]), rng)
                    s = torch.as_tensor(states, device=device)
                    a = torch.as_tensor(actions, device=device).unsqueeze(1)
                    r = torch.as_tensor(rewards, device=device)
                    ns = torch.as_tensor(next_states, device=device)
                    d = torch.as_tensor(dones, device=device)
                    nm = torch.as_tensor(next_masks, device=device)
                    with torch.no_grad():
                        # Double DQN: online selects, target evaluates. Mask before argmax.
                        next_actions = online(ns).masked_fill(~nm, -torch.inf).argmax(1, keepdim=True)
                        next_values = target(ns).gather(1, next_actions).squeeze(1)
                        targets = r + gamma * (1.0 - d) * next_values
                    current = online(s).gather(1, a).squeeze(1)
                    loss = F.smooth_l1_loss(current, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    optimizer.step()
                    last_loss = float(loss.item())

                if step % int(hp["target_update_frequency"]) == 0:
                    target.load_state_dict(online.state_dict())

                if done:
                    episode_count += 1
                    writer.writerow({"step": step, "episode": episode_count,
                                     "episode_reward": episode_reward,
                                     "orders_completed": info["orders_completed"],
                                     "td_loss": last_loss, "epsilon": epsilon})
                    print(f"step {step}: episode reward {episode_reward:.2f}, "
                          f"orders {info['orders_completed']}/{info['orders_arrived']}")
                    obs, _ = env.reset(seed=args.seed + episode_count)
                    episode_reward = 0.0

                if step % int(hp["eval_freq"]) == 0:
                    rows = evaluate(policy, args.config, "ddqn", args.seed, eval_seeds)
                    mean_reward = float(np.mean([row["total_reward"] for row in rows]))
                    writer.writerow({"step": step, "episode": episode_count,
                                     "td_loss": last_loss, "epsilon": epsilon,
                                     "eval_mean_reward": mean_reward})
                    handle.flush()
                    print(f"step {step}: eval reward {mean_reward:.2f}")
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        save_model()

        if not model_path.exists():
            save_model()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        online.load_state_dict(checkpoint["state_dict"])
        rows = evaluate(policy, args.config, "ddqn", args.seed, eval_seeds)
        print(f"best model: {model_path} | mean eval reward: "
              f"{np.mean([row['total_reward'] for row in rows]):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
