"""Train the centralized kitchen with sb3-contrib MaskablePPO.

Run from the project root::

    python scripts/train_centralized.py --config configs/env/medium.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restaurant_env import CentralizedKitchenEnv  # noqa: E402
from scripts.evaluate import evaluate  # noqa: E402


def main() -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/env/medium.yaml")
    parser.add_argument("--algo-config", default="configs/algo/maskppo.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    with open(args.algo_config, encoding="utf-8") as handle:
        hp = yaml.safe_load(handle)
    total = args.timesteps or int(hp["total_timesteps"])
    name = f"maskppo_{Path(args.config).stem}_seed{args.seed}"
    model_path = Path("models") / name
    run_dir = Path("runs") / name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = [int(s) for s in hp["eval_seeds"]]
    if not eval_seeds:
        raise ValueError("eval_seeds must contain at least one seed")

    class KitchenEvalCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_reward = -np.inf

        def _on_step(self) -> bool:
            if self.num_timesteps % int(hp["eval_freq"]):
                return True
            rows = evaluate(
                lambda obs, mask: self.model.predict(
                    obs, action_masks=mask, deterministic=True
                )[0],
                args.config, "maskppo", args.seed, eval_seeds,
            )
            mean_reward = float(np.mean([r["total_reward"] for r in rows]))
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_orders_completed", float(np.mean(
                [r["orders_completed"] for r in rows]
            )))
            print(f"step {self.num_timesteps}: eval reward {mean_reward:.2f}")
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(str(model_path))
            return True

    env = CentralizedKitchenEnv(config_path=args.config)
    try:
        model = MaskablePPO(
            "MlpPolicy", env,
            learning_rate=float(hp["learning_rate"]),
            n_steps=int(hp["n_steps"]),
            batch_size=int(hp["batch_size"]),
            n_epochs=int(hp["n_epochs"]),
            gamma=float(hp["gamma"]),
            gae_lambda=float(hp["gae_lambda"]),
            ent_coef=float(hp["ent_coef"]),
            clip_range=float(hp["clip_range"]),
            policy_kwargs={"net_arch": list(hp["net_arch"])},
            seed=args.seed,
            tensorboard_log=str(run_dir),
            verbose=1,
        )
        model.learn(total_timesteps=total, callback=KitchenEvalCallback())
        if not model_path.with_suffix(".zip").exists():
            model.save(str(model_path))
        best = MaskablePPO.load(str(model_path), env=env)
        rows = evaluate(
            lambda obs, mask: best.predict(obs, action_masks=mask, deterministic=True)[0],
            args.config, "maskppo", args.seed, eval_seeds,
        )
        print(f"best model: {model_path}.zip | mean eval reward: "
              f"{np.mean([r['total_reward'] for r in rows]):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
