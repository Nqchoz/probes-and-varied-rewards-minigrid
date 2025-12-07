"""Factories for available MiniGrid environments."""

from gymnasium.wrappers import FilterObservation, FlattenObservation

from minigrid_envs.door_key import DoorKeyEnv7x7
from minigrid_envs.four_rooms import FOUR_ROOMS_ID, make_env as make_four_rooms_env
from minigrid_envs.reward_wrappers import (
    ExplorationRewardWrapper,
    SubgoalDecayRewardWrapper,
    SubgoalRewardWrapper,
)


def make_env(
    env_name: str = "door_key",
    reward_wrapper: str | None = None,
    reward_wrapper_kwargs: dict | None = None,
    **kwargs,
):
    """Instantiate the requested MiniGrid environment.

    Parameters
    ----------
    env_name: str
        Either "door_key" (custom fixed 7x7 layout) or "four_rooms" (stock
        ``MiniGrid-FourRooms-v0``).
    **kwargs:
        Forwarded to the underlying environment constructor.
    """

    max_steps = kwargs.pop("max_steps", None)

    def _apply_max_steps(source_kwargs: dict) -> dict:
        env_kwargs = dict(source_kwargs)
        if max_steps is not None:
            env_kwargs["max_steps"] = max_steps
        return env_kwargs

    env_name = env_name.lower()
    if env_name == "door_key":
        env = DoorKeyEnv7x7(**_apply_max_steps(kwargs))
    elif env_name == "four_rooms":
        env = make_four_rooms_env(**_apply_max_steps(kwargs))
    else:
        raise ValueError(
            f"Unknown env_name '{env_name}'. Choose 'door_key' or 'four_rooms'."
        )

    if reward_wrapper is not None:
        reward_wrapper = reward_wrapper.lower()
        if reward_wrapper == "subgoal":
            wrapper_kwargs = reward_wrapper_kwargs or {}
            env = SubgoalRewardWrapper(env, **wrapper_kwargs)
        elif reward_wrapper == "subgoal_decay":
            wrapper_kwargs = reward_wrapper_kwargs or {}
            env = SubgoalDecayRewardWrapper(env, **wrapper_kwargs)
        elif reward_wrapper == "exploration":
            wrapper_kwargs = reward_wrapper_kwargs or {}
            env = ExplorationRewardWrapper(env, **wrapper_kwargs)
        else:
            raise ValueError(
                "Unknown reward_wrapper '"
                f"{reward_wrapper}'. Currently supports 'subgoal', 'subgoal_decay', "
                "and 'exploration'."
            )

    env = FilterObservation(env, ["image", "direction"])
    env = FlattenObservation(env)
    return env


__all__ = [
    "make_env",
    "DoorKeyEnv7x7",
    "FOUR_ROOMS_ID",
    "SubgoalRewardWrapper",
    "SubgoalDecayRewardWrapper",
    "ExplorationRewardWrapper",
]
