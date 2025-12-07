import argparse
import json
from typing import Any, Optional

import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import Door, Goal, Key
from stable_baselines3 import PPO

from minigrid_envs.door_key import DoorKeyEnv7x7
from train_env import make_env


def _find_objects(env: Any):
    door_info = {"pos": None, "obj": None}
    key_info = {"pos": None, "obj": None}
    goal_info = {"pos": None, "obj": None}

    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, Door):
                door_info = {"pos": (x, y), "obj": obj}
            elif isinstance(obj, Key):
                key_info = {"pos": (x, y), "obj": obj}
            elif isinstance(obj, Goal):
                goal_info = {"pos": (x, y), "obj": obj}

    return door_info, key_info, goal_info


def _visible_flags(obs_image: np.ndarray, object_idx: int) -> tuple[bool, list[list[int]]]:
    matches = np.argwhere(obs_image[:, :, 0] == object_idx)
    return bool(len(matches)), matches.tolist()


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def generate_dataset(
    num_samples: int = 2000,
    num_episodes: int = 200,
    model_path: Optional[str] = None,
    random_ratio: float = 0.3,
    output_obs: str = 'eval_obs.npy',
    output_meta: str = 'eval_meta.json',
):
    """Generate evaluation dataset with diverse observations.
    
    Parameters
    ----------
    num_samples : int
        Target number of observations to collect.
    num_episodes : int
        Number of episodes to run.
    model_path : str, optional
        Path to trained PPO model. If provided, uses policy actions mixed with random.
        If None, uses only random actions.
    random_ratio : float
        Ratio of random actions when using a policy (0.0 = all policy, 1.0 = all random).
    output_obs : str
        Output path for observations array.
    output_meta : str
        Output path for metadata JSON.
    """
    # Load model if provided
    model = None
    wrapped_env = None
    if model_path:
        print(f"Loading policy from {model_path}...")
        model = PPO.load(model_path)
        wrapped_env = make_env(env_name="door_key", reward_wrapper=None)
        print("Policy loaded. Will use mixed policy/random actions.")
    
    env = DoorKeyEnv7x7()
    KEY_IDX = OBJECT_TO_IDX["key"]
    DOOR_IDX = OBJECT_TO_IDX["door"]
    GOAL_IDX = OBJECT_TO_IDX["goal"]
    dataset = []
    meta = []
    
    print(f"Generating {num_samples} observations across {num_episodes} episodes...")
    print(f"Action strategy: {'Random only' if model is None else f'{int((1-random_ratio)*100)}% policy, {int(random_ratio*100)}% random'}")
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        if wrapped_env is not None:
            wrapped_obs, _ = wrapped_env.reset(seed=ep)
        
        done = False
        visited = set()
        
        while not done and len(dataset) < num_samples:
            # Choose action: policy or random
            if model is not None and np.random.random() > random_ratio:
                # Use policy action
                action, _ = model.predict(wrapped_obs, deterministic=False)
                action = int(action)
            else:
                # Use random action
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            if wrapped_env is not None:
                wrapped_obs, _, _, _, _ = wrapped_env.step(action)
            
            done = terminated or truncated
            door_info, key_info, goal_info = _find_objects(env)
            
            # Extract easily-labeled meta
            agent_pos = tuple(env.agent_pos)
            has_key = env.carrying is not None
            goal_pos = goal_info["pos"]
            dist_to_goal = (
                abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
                if goal_pos is not None
                else None
            )
            visited_flag = agent_pos in visited
            visited.add(agent_pos)
            obs_image = obs['image']
            key_visible, key_coords = _visible_flags(obs_image, KEY_IDX)
            door_visible, door_coords = _visible_flags(obs_image, DOOR_IDX)
            goal_visible, goal_coords = _visible_flags(obs_image, GOAL_IDX)

            door_obj = door_info["obj"]
            dataset.append(obs_image)
            meta.append(
                _to_serializable(
                    {
                        'agent_pos': agent_pos,
                        'has_key': has_key,
                        'dist_to_goal': dist_to_goal,
                        'visited': visited_flag,
                        'key_visible': key_visible,
                        'door_visible': door_visible,
                        'goal_visible': goal_visible,
                        'visible_key_coords': key_coords,
                        'visible_door_coords': door_coords,
                        'visible_goal_coords': goal_coords,
                        'door_open': bool(door_obj and door_obj.is_open),
                        'door_locked': bool(door_obj and door_obj.is_locked),
                        'door_pos': door_info['pos'],
                        'key_pos': key_info['pos'],
                        'goal_pos': goal_info['pos'],
                    }
                )
            )
        
        if len(dataset) >= num_samples:
            break
    
    # Save dataset
    print(f"\nCollected {len(dataset)} observations")
    np.save(output_obs, np.array(dataset))
    with open(output_meta, 'w') as f:
        json.dump(meta, f)
    
    # Print diversity statistics
    print(f"\nDataset diversity:")
    for concept in ['has_key', 'key_visible', 'door_open', 'door_visible', 'goal_visible']:
        values = [m[concept] for m in meta]
        unique, counts = np.unique(values, return_counts=True)
        print(f"  {concept}: {dict(zip(unique, counts))}")
    
    print(f"\n✓ Saved to {output_obs} and {output_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation dataset for concept probing")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of observations to collect",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=200,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained PPO model (if None, uses random actions only)",
    )
    parser.add_argument(
        "--random-ratio",
        type=float,
        default=0.3,
        help="Ratio of random actions when using policy (0.0-1.0)",
    )
    parser.add_argument(
        "--output-obs",
        type=str,
        default="eval_obs.npy",
        help="Output path for observations",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="eval_meta.json",
        help="Output path for metadata",
    )
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.num_samples,
        num_episodes=args.num_episodes,
        model_path=args.model_path,
        random_ratio=args.random_ratio,
        output_obs=args.output_obs,
        output_meta=args.output_meta,
    )
