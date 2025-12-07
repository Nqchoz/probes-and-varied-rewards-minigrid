"""Custom MiniGrid DoorKey environment with a fixed 7x7 layout."""
from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv


class DoorKeyEnv7x7(MiniGridEnv):
    """Fixed-layout DoorKey environment sized 7x7. Based on the stock MiniGrid 6x6 DoorKey environment"""

    def __init__(
        self,
        agent_start_pos: tuple[int, int] | None = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = 64,
        **kwargs,
    ) -> None:
        mission_space = MissionSpace(
            mission_func=lambda: "use the key to unlock the door and reach the goal"
        )
        super().__init__(
            mission_space=mission_space,
            width=7,
            height=7,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.door_color = "yellow"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Middle partition with a locked door
        partition_x = width // 2
        self.grid.vert_wall(partition_x, 1, height - 2)
        door_y = height // 2
        door = Door(self.door_color, is_locked=True)
        self.put_obj(door, partition_x, door_y)

        # Key placement on the agent side of the door
        key_pos = (partition_x - 2, height - 3)
        self.put_obj(Key(color=self.door_color), *key_pos)

        # Goal beyond the door
        goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *goal_pos)

        if self.agent_start_pos is None:
            self.agent_start_pos = self.place_agent()
        else:
            # Place agent deterministically at the requested tile.
            self.place_agent(
                top=self.agent_start_pos,
                size=(1, 1),
                rand_dir=False,
            )
            self.agent_dir = self.agent_start_dir

        self.mission = "use the key to unlock the door and reach the goal"

    @staticmethod
    def available_door_colors() -> list[str]:
        """Return door colors supported by MiniGrid."""
        return list(COLOR_NAMES)
