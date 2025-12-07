"""Train PPO agents on MiniGrid environments with configurable reward wrappers."""
from __future__ import annotations

import argparse
import ast
import os
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from train_env import make_env

REWARD_WRAPPERS = ["goal_only", "goal_decay", "subgoal", "subgoal_decay", "exploration"]


def build_env(
    env_name: str,
    reward_wrapper: str | None,
    reward_wrapper_kwargs: dict[str, Any] | None,
    seed: int | None,
    render_mode: str | None = None,
    **env_kwargs,
):
    def _thunk():
        env = make_env(
            env_name=env_name,
            reward_wrapper=reward_wrapper,
            reward_wrapper_kwargs=reward_wrapper_kwargs,
            render_mode=render_mode,
            **env_kwargs,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def parse_kwargs(kvs: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for pair in kvs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid kv '{pair}'. Expected format key=value"
            )
        key, value = pair.split("=", 1)
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="door_key", choices=["door_key", "four_rooms"])
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument(
        "--reward-wrapper",
        type=str,
        choices=REWARD_WRAPPERS + ["none"],
        default="none",
        help="Reward wrapper to apply (or 'none').",
    )
    parser.add_argument("--reward-wrapper-kw", nargs="*", default=())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Optional path to an existing PPO checkpoint to resume training from.",
    )
    args = parser.parse_args()

    reward_kwargs = parse_kwargs(list(args.reward_wrapper_kw)) if args.reward_wrapper_kw else None
    reward_wrapper = None if args.reward_wrapper == "none" else args.reward_wrapper

    env_fns = [
        build_env(
            env_name=args.env,
            reward_wrapper=reward_wrapper,
            reward_wrapper_kwargs=reward_kwargs,
            seed=None if args.seed is None else args.seed + i,
            render_mode="human" if args.render else None,
        )
        for i in range(args.n_envs)
    ]

    vec_env: VecEnv = DummyVecEnv(env_fns)

    if args.init_model:
        model = PPO.load(args.init_model, env=vec_env, device=args.device)
        model.set_env(vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            device=args.device,
            seed=args.seed,
        )

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path=args.save_dir,
        name_prefix=f"ppo_{args.env}_{reward_wrapper or 'base'}",
    )

    model.learn(total_timesteps=args.total_steps, callback=checkpoint_callback)
    model.save(
        os.path.join(
            args.save_dir,
            f"ppo_{args.env}_{reward_wrapper or 'base'}_final",
        )
    )


if __name__ == "__main__":
    main()
