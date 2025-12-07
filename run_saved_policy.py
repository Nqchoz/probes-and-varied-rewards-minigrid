"""Render rollouts of a saved PPO policy inside the MiniGrid environments."""
from __future__ import annotations

import argparse
import ast
from typing import Any, Dict

from stable_baselines3 import PPO

from train_env import make_env

REWARD_WRAPPERS = {None, "subgoal", "subgoal_decay", "exploration"}


def parse_kwargs(pairs: list[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid kwarg '{pair}'. Expected format key=value."
            )
        key, value = pair.split("=", 1)
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the Stable-Baselines3 checkpoint (e.g. runs/ppo_env_wrapper_final.zip).",
    )
    parser.add_argument(
        "--env",
        dest="env_name",
        choices=["door_key", "four_rooms"],
        default="door_key",
        help="Base environment to instantiate for rollouts.",
    )
    parser.add_argument(
        "--reward-wrapper",
        choices=sorted(filter(None, REWARD_WRAPPERS)),
        default=None,
        help="Optional reward wrapper to mirror training conditions.",
    )
    parser.add_argument(
        "--reward-wrapper-kw",
        nargs="*",
        default=(),
        metavar="key=value",
        help="Override reward wrapper kwargs (literal-evaluated).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max_steps override passed to make_env.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to roll out for visualization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed applied to each episode (offset per episode).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions (default samples from the policy).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable environment rendering for headless rollouts.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to Stable-Baselines3 when loading the model (e.g. 'cpu').",
    )
    args = parser.parse_args()

    reward_kw = parse_kwargs(list(args.reward_wrapper_kw)) if args.reward_wrapper_kw else None
    model = PPO.load(args.model_path, device=args.device)

    render_mode = None if args.no_render else "human"
    env = make_env(
        env_name=args.env_name,
        reward_wrapper=args.reward_wrapper,
        reward_wrapper_kwargs=reward_kw,
        max_steps=args.max_steps,
        render_mode=render_mode,
    )

    try:
        for episode_idx in range(args.episodes):
            seed = args.seed + episode_idx if args.seed is not None else None
            obs, _ = env.reset(seed=seed)
            done = False
            episode_return = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, _info = env.step(action)
                episode_return += float(reward)
                steps += 1
                done = terminated or truncated

            print(
                f"Episode {episode_idx + 1}/{args.episodes}: return={episode_return:.2f}, steps={steps}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
