"""Reward shaping wrappers for MiniGrid experiments."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType
from minigrid.core.world_object import Door


class SubgoalRewardWrapper(gym.Wrapper):
    """Reward key pickup (+0.3), door open(+0.3), and goal completion (0.4) milestones."""

    def __init__(
        self,
        env: gym.Env,
        *,
        key_reward: float = 0.3,
        door_reward: float = 0.3,
        goal_reward: float = 0.4,
    ) -> None:
        super().__init__(env)
        self.key_reward = key_reward
        self.door_reward = door_reward
        self.goal_reward = goal_reward
        self._rewarded_key = False
        self._rewarded_door = False

    def reset(self, **kwargs: Any):
        self._rewarded_key = False
        self._rewarded_door = False
        return self.env.reset(**kwargs)

    def _has_key(self) -> bool:
        carrying = getattr(self.unwrapped, "carrying", None)
        return getattr(carrying, "type", None) == "key"

    def _door_is_open(self) -> bool:
        grid = getattr(self.unwrapped, "grid", None)
        if grid is None or grid.grid is None:
            return False
        for obj in grid.grid:  # type: ignore[attr-defined]
            if isinstance(obj, Door) and obj.is_open:
                return True
        return False

    def step(self, action: ActType):
        had_key = self._has_key()
        door_open = self._door_is_open()
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped = 0.0 #cumulative sum of reward for this step
        if not self._rewarded_key: #ensure key reward is only given upon first pick-up
            has_key = self._has_key()
            if has_key and not had_key:
                shaped += self.key_reward
                self._rewarded_key = True
        if not self._rewarded_door: #ensure door reward is only given upon first door opening
            now_open = self._door_is_open()
            if now_open and not door_open:
                shaped += self.door_reward
                self._rewarded_door = True
        if terminated: #reaches goal square
            shaped += self.goal_reward

        return obs, shaped, terminated, truncated, info


class SubgoalDecayRewardWrapper(gym.Wrapper):
    """Reward key pickup (+0.3), door open(+0.3), and goal completion (+0.4) milestones.
       Subtract 0.01 per total number of steps from final reward"""

    def __init__(
        self,
        env: gym.Env,
        *,
        key_reward: float = 0.3,
        door_reward: float = 0.3,
        goal_reward: float = 0.4,
        step_penalty: float = 0.01,
    ) -> None:
        super().__init__(env)
        self.key_reward = key_reward
        self.door_reward = door_reward
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self._rewarded_key = False
        self._rewarded_door = False

    def reset(self, **kwargs: Any):
        self._rewarded_key = False
        self._rewarded_door = False
        return self.env.reset(**kwargs)

    def _has_key(self) -> bool:
        carrying = getattr(self.unwrapped, "carrying", None)
        return getattr(carrying, "type", None) == "key"

    def _door_is_open(self) -> bool:
        grid = getattr(self.unwrapped, "grid", None)
        if grid is None or grid.grid is None:
            return False
        for obj in grid.grid:  # type: ignore[attr-defined]
            if isinstance(obj, Door) and obj.is_open:
                return True
        return False

    def step(self, action: ActType):
        had_key = self._has_key()
        door_open = self._door_is_open()
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped = -self.step_penalty
        if not self._rewarded_key:
            has_key = self._has_key()
            if has_key and not had_key:
                shaped += self.key_reward
                self._rewarded_key = True
        if not self._rewarded_door:
            now_open = self._door_is_open()
            if now_open and not door_open:
                shaped += self.door_reward
                self._rewarded_door = True
        if terminated:
            shaped += self.goal_reward

        return obs, shaped, terminated, truncated, info

class ExplorationRewardWrapper(gym.Wrapper):
    """Provide bonuses for visiting new tiles (+0.08) and for reaching the goal (+0.5)."""

    def __init__(
        self,
        env: gym.Env,
        *,
        visit_reward: float = 0.08,
        goal_reward: float = 0.5,
        step_penalty: float = 0.00,
        count_start: bool = False,
    ) -> None:
        super().__init__(env)
        self.visit_reward = visit_reward
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.count_start = count_start
        self._visited: set[tuple[int, int]] = set()

    def reset(self, **kwargs: Any):
        obs = self.env.reset(**kwargs)
        self._visited = set()
        if self.count_start:
            pos = self._agent_pos()
            if pos is not None:
                self._visited.add(pos)
        return obs

    def _agent_pos(self) -> tuple[int, int] | None:
        pos = getattr(self.unwrapped, "agent_pos", None)
        if pos is None:
            return None
        return (int(pos[0]), int(pos[1]))

    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = -self.step_penalty if self.step_penalty else 0.0
        pos = self._agent_pos()
        if pos is not None and pos not in self._visited:
            self._visited.add(pos)
            shaped += self.visit_reward
        if terminated:
            shaped += self.goal_reward
        return obs, shaped, terminated, truncated, info


__all__ = [
    "SubgoalRewardWrapper",
    "SubgoalDecayRewardWrapper",
    "ExplorationRewardWrapper",
]
