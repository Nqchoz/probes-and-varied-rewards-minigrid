"""Factory helpers for the stock MiniGrid FourRooms layout."""
from __future__ import annotations

import gymnasium as gym
from gymnasium.core import Env

FOUR_ROOMS_ID = "MiniGrid-FourRooms-v0"


def make_env(**kwargs) -> Env:
    """Instantiate the canonical MiniGrid FourRooms environment."""
    return gym.make(FOUR_ROOMS_ID, **kwargs)
