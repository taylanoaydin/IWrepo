import os
from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import lax
from jax import numpy as jp
import mujoco
import xml.etree.ElementTree as ET

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = 'r'
GOAL = G = 'g'

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                 [1, R, 0, 1, 1, G, G, 1],
                 [1, 0, 0, 1, 0, G, G, 1],
                 [1, 1, 0, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 0, 1],
                 [1, 0, 1, G, 0, 1, G, 1],
                 [1, 0, G, G, 1, G, G, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE1 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, G, 1, 1, G, 1, 1],
             [1, G, G, R, 1, G, 1, 1],
             [1, 1, 1, G, 1, G, G, 1],
             [1, G, G, G, 1, G, 1, 1],
             [1, 1, G, 1, 1, G, 1, 1],
             [1, G, G, G, R, G, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE2 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, 1, 1, 1, R, 1, 1],
             [1, G, 1, 1, 1, G, G, 1],
             [1, G, 1, G, G, G, 1, 1],
             [1, G, G, G, 1, R, G, 1],
             [1, G, G, 1, 1, 1, G, 1],
             [1, G, 1, 1, G, G, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE3 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, 1, 1, 1, R, 1, 1],
             [1, G, 1, 1, 1, G, G, 1],
             [1, G, 1, G, G, G, 1, 1],
             [1, G, G, G, 1, R, G, 1],
             [1, G, G, 1, 1, 1, G, 1],
             [1, G, 1, 1, G, G, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE4 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, G, 1, G, 1, 1, 1],
             [1, R, 1, G, G, G, 1, 1],
             [1, G, G, G, 1, G, 1, 1],
             [1, G, 1, 1, 1, G, G, 1],
             [1, G, G, G, G, G, G, 1],
             [1, G, 1, G, R, 1, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE5 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, 1, G, G, G, 1, 1],
             [1, R, 1, G, 1, G, G, 1],
             [1, G, G, G, 1, 1, R, 1],
             [1, 1, G, 1, 1, 1, 1, 1],
             [1, G, G, G, G, G, G, 1],
             [1, G, R, 1, 1, G, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE6 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, 1, 1, G, 1, 1, 1],
             [1, G, 1, G, R, G, 1, 1],
             [1, R, G, G, 1, G, G, 1],
             [1, G, G, 1, 1, G, 1, 1],
             [1, 1, G, 1, 1, G, 1, 1],
             [1, 1, 1, 1, R, G, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE7 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, R, G, G, 1, 1, 1],
             [1, G, 1, 1, G, G, 1, 1],
             [1, G, 1, 1, R, G, 1, 1],
             [1, 1, 1, G, G, 1, 1, 1],
             [1, R, G, G, G, 1, 1, 1],
             [1, G, 1, G, G, G, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE8 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, G, G, 1, R, 1, 1, 1],
             [1, 1, G, G, G, G, G, 1],
             [1, G, 1, G, 1, R, 1, 1],
             [1, G, G, G, G, G, 1, 1],
             [1, G, 1, 1, 1, G, G, 1],
             [1, G, G, 1, R, G, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE9 = [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, R, 1, 1, G, G, 1, 1],
             [1, G, G, 1, 1, G, 1, 1],
             [1, G, G, G, R, G, G, 1],
             [1, G, 1, 1, G, 1, 1, 1],
             [1, G, G, 1, G, G, G, 1],
             [1, 1, G, G, G, 1, G, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE10 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, G, 1, G, G, G, G, 1],
              [1, R, G, G, 1, G, 1, 1],
              [1, 1, 1, 1, 1, G, G, 1],
              [1, G, G, G, R, G, 1, 1],
              [1, G, G, 1, 1, G, G, 1],
              [1, R, 1, 1, 1, 1, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE11 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, G, G, R, G, 1, 1, 1],
              [1, G, 1, 1, G, 1, G, 1],
              [1, R, 1, G, G, G, G, 1],
              [1, G, G, G, 1, 1, G, 1],
              [1, 1, G, 1, 1, R, G, 1],
              [1, G, G, G, 1, G, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE12 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, R, 1, 1, G, G, 1, 1],
              [1, G, 1, 1, G, 1, 1, 1],
              [1, G, G, 1, R, G, G, 1],
              [1, 1, G, G, G, G, 1, 1],
              [1, 1, G, 1, 1, 1, 1, 1],
              [1, G, G, G, R, G, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE13 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, G, G, 1, 1, 1, 1, 1],
              [1, R, 1, G, 1, G, G, 1],
              [1, G, G, G, G, R, 1, 1],
              [1, 1, G, G, 1, 1, 1, 1],
              [1, G, 1, G, 1, 1, G, 1],
              [1, G, G, G, G, G, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE14 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, R, 1, 1, 1, G, G, 1],
              [1, G, 1, 1, 1, G, 1, 1],
              [1, G, G, G, R, G, 1, 1],
              [1, G, G, G, 1, G, G, 1],
              [1, 1, G, 1, 1, 1, 1, 1],
              [1, G, R, G, G, G, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE15 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, G, 1, R, 1, G, G, 1],
              [1, G, 1, G, 1, G, 1, 1],
              [1, G, G, G, G, G, G, 1],
              [1, R, 1, 1, G, 1, G, 1],
              [1, G, G, G, 1, 1, G, 1],
              [1, G, 1, R, 1, G, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE16 = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, G, G, G, G, G, 1, 1],
              [1, R, 1, 1, R, 1, 1, 1],
              [1, 1, 1, G, G, G, G, 1],
              [1, 1, 1, 1, G, 1, 1, 1],
              [1, R, G, G, G, 1, G, 1],
              [1, G, 1, G, G, G, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

EMPTY_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, G, G, G, G, G, G, 1],
              [1, R, G, G, R, G, G, 1],
              [1, G, G, G, G, G, G, 1],
              [1, G, G, G, G, G, G, 1],
              [1, R, G, G, G, G, G, 1],
              [1, G, G, G, G, R, G, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5

def combine_16_mazes(mazes):
    """
    Combines 16 individual 8x8 mazes into a single 4x4 grid maze (32x32).

    Args:
        mazes: A list of 16 individual 8x8 mazes (2D lists).

    Returns:
        A 2D list representing the combined maze grid.
    """
    if len(mazes) != 16:
        raise ValueError("You must provide exactly 16 mazes.")
    
    maze_size = len(mazes[0])  # Assumes all mazes are square and of the same size
    combined_size = 4 * maze_size  # 4x4 grid

    # Initialize the combined maze with walls
    COMBINED_MAZE = [[1] * combined_size for _ in range(combined_size)]

    # Populate the combined maze with individual mazes
    for grid_row in range(4):
        for grid_col in range(4):
            maze_index = grid_row * 4 + grid_col
            maze = mazes[maze_index]
            for i in range(maze_size):
                for j in range(maze_size):
                    COMBINED_MAZE[grid_row * maze_size + i][grid_col * maze_size + j] = maze[i][j]

    return COMBINED_MAZE

# Define or load the 16 individual 8x8 mazes (BIG_MAZE1 to BIG_MAZE16)
BIG_MAZES = [BIG_MAZE1, BIG_MAZE2, BIG_MAZE3, BIG_MAZE4,
             BIG_MAZE5, BIG_MAZE6, BIG_MAZE7, BIG_MAZE8,
             BIG_MAZE9, BIG_MAZE10, BIG_MAZE11, BIG_MAZE12,
             BIG_MAZE13, BIG_MAZE14, BIG_MAZE15, BIG_MAZE16]

EMPTY_MAZES = [EMPTY_MAZE.copy() for _ in range(16)]

# Combine the 16 mazes into a 4x4 grid
COMBINED_MAZE = combine_16_mazes(BIG_MAZES)

EM_COMBINED_MAZE = combine_16_mazes(EMPTY_MAZES)

def find_starts_jax(structure, maze_size, size_scaling):
    """
    Finds all start positions ('r') in the grid and represents them as a JAX-compatible structure.
    
    Args:
        structure: The combined 2D grid maze.
        maze_size: A tuple (rows, cols) specifying the size of individual mazes.
        size_scaling: Scaling factor for positions.

    Returns:
        A tuple of:
            - starts: JAX array of start positions with an additional column for maze indices.
            - maze_indices: JAX array of unique maze indices.
    """
    starts_list = []
    maze_indices = []

    num_mazes_vertical = len(structure) // maze_size[0]
    num_mazes_horizontal = len(structure[0]) // maze_size[1]

    for grid_row in range(num_mazes_vertical):
        for grid_col in range(num_mazes_horizontal):
            maze_index = grid_row * num_mazes_horizontal + grid_col
            maze_offset_x = grid_row * maze_size[0]
            maze_offset_y = grid_col * maze_size[1]

            for i in range(maze_size[0]):
                for j in range(maze_size[1]):
                    x, y = maze_offset_x + i, maze_offset_y + j
                    if structure[x][y] == RESET:
                        starts_list.append([x * size_scaling, y * size_scaling, maze_index])

            if starts_list:
                maze_indices.append(maze_index)

    starts = jp.array(starts_list) if starts_list else jp.array([])
    maze_indices = jp.array(maze_indices)

    return starts, maze_indices


def find_goals_jax(structure, maze_size, size_scaling):
    """
    Finds all goal positions ('g') in the grid and represents them as a JAX-compatible structure.
    
    Args:
        structure: The combined 2D grid maze.
        maze_size: A tuple (rows, cols) specifying the size of individual mazes.
        size_scaling: Scaling factor for positions.

    Returns:
        A tuple of:
            - goals: JAX array of goal positions with an additional column for maze indices.
            - maze_indices: JAX array of unique maze indices.
    """
    goals_list = []
    maze_indices = []

    num_mazes_vertical = len(structure) // maze_size[0]
    num_mazes_horizontal = len(structure[0]) // maze_size[1]

    for grid_row in range(num_mazes_vertical):
        for grid_col in range(num_mazes_horizontal):
            maze_index = grid_row * num_mazes_horizontal + grid_col
            maze_offset_x = grid_row * maze_size[0]
            maze_offset_y = grid_col * maze_size[1]

            for i in range(maze_size[0]):
                for j in range(maze_size[1]):
                    x, y = maze_offset_x + i, maze_offset_y + j
                    if structure[x][y] == GOAL:
                        goals_list.append([x * size_scaling, y * size_scaling, maze_index])

            if goals_list:
                maze_indices.append(maze_index)

    goals = jp.array(goals_list) if goals_list else jp.array([])
    maze_indices = jp.array(maze_indices)

    return goals, maze_indices

def make_maze(maze_size_scaling):
    """
    Creates a maze XML and identifies possible starts and goals.

    Args:
        COMBINED_MAZE: A 2D list representing the combined maze structure (4x4 grid of 8x8 mazes).
        maze_size_scaling: Scaling factor for the maze.

    Returns:
        xml_string: The XML representation of the maze.
        starts: A JAX array of start positions with an additional column for maze indices.
        goals: A JAX array of goal positions with an additional column for maze indices.
        start_indices: A JAX array of unique maze indices corresponding to starts.
        goal_indices: A JAX array of unique maze indices corresponding to goals.
    """
    maze_size = (8, 8)  # Each maze is 8x8

    # Find starts and goals with JAX-compatible structures
    starts, start_indices = find_starts_jax(COMBINED_MAZE, maze_size, maze_size_scaling)
    goals, goal_indices = find_goals_jax(COMBINED_MAZE, maze_size, maze_size_scaling)

    # Load the Mujoco XML template
    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_maze2.xml")
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    # Add walls to the maze
    for i in range(len(COMBINED_MAZE)):
        for j in range(len(COMBINED_MAZE[0])):
            if COMBINED_MAZE[i][j] == 1:  # Wall
                ET.SubElement(
                    worldbody, "geom",
                    name=f"block_{i}_{j}",
                    pos=f"{i * maze_size_scaling} {j * maze_size_scaling} {MAZE_HEIGHT / 2 * maze_size_scaling}",
                    size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {MAZE_HEIGHT / 2 * maze_size_scaling}",
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    # Serialize XML
    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, starts, goals, start_indices, goal_indices

def preprocess_maze_layout(maze_layout):
    """Convert maze layout to a numeric format compatible with JAX arrays."""
    return [[
        0 if cell == 'r' else  # Start position
        2 if cell == 'g' else  # Goal position
        cell  # Walls (1) and other values remain unchanged
        for cell in row
    ] for row in maze_layout]


class AntMaze3(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=1.0,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals, start_ind, goal_ind = make_maze(maze_size_scaling)
        raw_maze_layout = COMBINED_MAZE

        self.maze_layout = jp.array(preprocess_maze_layout(raw_maze_layout))  # Store maze layout as a 2D array
        self.maze_size_scaling = maze_size_scaling  # Store the scaling factor

        sys = mjcf.loads(xml_string)

        # Restructure possible starts and goals into a padded JAX array
        max_starts = max(len([pos for pos in possible_starts if pos[2] == i]) for i in range(len(start_ind)))
        max_goals = max(len([pos for pos in possible_goals if pos[2] == i]) for i in range(len(goal_ind)))

        self.possible_starts = jp.zeros((len(start_ind), max_starts, 2))
        self.possible_goals = jp.zeros((len(goal_ind), max_goals, 2))

        for i in range(len(start_ind)):
            starts = jp.array([pos[:2] for pos in possible_starts if pos[2] == i])
            self.possible_starts = self.possible_starts.at[i, :starts.shape[0], :].set(starts)

        for i in range(len(goal_ind)):
            goals = jp.array([pos[:2] for pos in possible_goals if pos[2] == i])
            self.possible_goals = self.possible_goals.at[i, :goals.shape[0], :].set(goals)


        # Indices of mazes with valid starts and goals
        self.start_ind = start_ind
        self.goal_ind = goal_ind

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        maze_rows, maze_cols = self.maze_layout.shape
        maze_layout_flat_size = maze_rows * maze_cols

        
        self.state_dim = 38
        self.goal_indices = jp.array([0, 1])

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # Select random start and target from the same maze
        start, target = self._random_start_and_target(rng2)

        # Set start and target in the state
        q = q.at[:2].set(start)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state


    def _is_valid_position(self, position: jp.ndarray) -> jp.bool_:
        """Checks if a position is valid (not in a wall or outside the maze)."""
        # Map scaled position to original grid coordinates
        grid_x = jp.floor(position[0] / self.maze_size_scaling + 0.5).astype(int)
        grid_y = jp.floor(position[1] / self.maze_size_scaling + 0.5).astype(int)

        # Compute bounds and check if the position is within bounds
        in_bounds_x = jp.logical_and(grid_x >= 0, grid_x < len(self.maze_layout))
        in_bounds_y = jp.logical_and(grid_y >= 0, grid_y < len(self.maze_layout[0]))
        in_bounds = jp.logical_and(in_bounds_x, in_bounds_y)

        # Check if the grid cell is not a wall
        is_not_wall = jp.logical_and(in_bounds, self.maze_layout[grid_x, grid_y] != 1)

        return jp.all(is_not_wall)


    # Todo rename seed to traj_id
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state

        # Extract the initial position and process the action
        initial_position = pipeline_state0.x.pos[0]
        clipped_action = jp.where(
            jp.abs(action[:2]) <= 0.33, 
            0, 
            jp.where(action[:2] > 0, 0.1, -0.1)
        )

        # Compute position deltas
        delta_x, delta_y = clipped_action[0], clipped_action[1]
        new_position = initial_position[:2] + jp.array([delta_x, delta_y])

        # Use jax.lax.cond to check position validity
        is_valid = self._is_valid_position(new_position)

        def valid_case(_):
            # Update the first two values of q (x and y coordinates)
            new_q = pipeline_state0.q.at[0].add(delta_x).at[1].add(delta_y)

            # Reinitialize pipeline state with updated q and zero qd
            new_qd = pipeline_state0.qd.at[:2].set(0)  # Reset velocities for x and y
            new_qd = new_qd.at[-2:].set(0)  # Reset velocities for x and y
            return self.pipeline_init(new_q, new_qd)

        def invalid_case(_):
            # Return the pipeline state unchanged
            return pipeline_state0

        pipeline_state = jax.lax.cond(is_valid, valid_case, invalid_case, None)

        # Compute position difference for debugging
        final_position = pipeline_state.x.pos[0]
        position_diff = final_position[:2] - initial_position[:2]

        # Debugging output
        # jax.debug.print(
        #    "Step: Start Pos: {start_pos}, End Pos: {end_pos}, ΔPos: {delta_pos}, Action: {act}", 
        #    start_pos=initial_position[:2], 
        #    end_pos=final_position[:2], 
        #    delta_pos=position_diff, 
        #    act=action[:2]
        #)


        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2., dtype=float)
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
    
    def get_grid_position(self, position: jp.ndarray, size_scaling: float) -> Tuple[int, int]:
        """
        Converts agent position to grid coordinates.
        Args:
            position: A 2D array representing the agent's position `[x, y]`.
            size_scaling: Scaling factor for the maze.
        
        Returns:
            grid_x, grid_y: Grid coordinates corresponding to the agent's position.
        """
        grid_x = jp.floor(position[0] / size_scaling + 0.5).astype(int)
        grid_y = jp.floor(position[1] / size_scaling + 0.5).astype(int)
        return grid_x, grid_y

    def get_local_grid(self, maze: jax.Array, agent_grid_pos: Tuple[int, int], size: int) -> jax.Array:
        """
        Extracts a local grid around the agent's position.
        
        Args:
            maze: The full maze array.
            agent_grid_pos: Tuple (grid_x, grid_y) representing the agent's position in grid coordinates.
            size: The size of the local grid (must be odd).

        Returns:
            A JAX array of the local grid centered around the agent.
        """
        half_size = size // 2
        grid_x, grid_y = agent_grid_pos

        # Pad the maze with walls (1)
        padded_maze = jax.numpy.pad(maze, pad_width=half_size, constant_values=1)

        # Adjust grid_x and grid_y to account for padding
        padded_x = grid_x + half_size
        padded_y = grid_y + half_size

        # Extract the local grid using dynamic slicing
        local_grid = lax.dynamic_slice(
            padded_maze,
            start_indices=(padded_x - half_size, padded_y - half_size),
            slice_sizes=(size, size)
        )
        return local_grid


    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position, velocities, and local maze grid."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        agent_pos = pipeline_state.x.pos[0][:2]
        target_pos = pipeline_state.x.pos[-1][:2]

        grid_x, grid_y = self.get_grid_position(agent_pos, self.maze_size_scaling)
        local_grid = self.get_local_grid(self.maze_layout, (grid_x, grid_y), size=3)

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jp.concatenate([qpos, qvel, local_grid.flatten(), target_pos])


    def _random_start_and_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Returns a random start and target location from the same maze.

        Args:
            rng: JAX random key.

        Returns:
            start: A randomly chosen start position `[x, y]` from the selected maze.
            target: A randomly chosen target position `[x, y]` from the same maze.
        """
        rng, rng_idx, rng_start, rng_target = jax.random.split(rng, 4)

        # Sample a random maze index
        maze_idx = jax.random.choice(rng_idx, self.start_ind)

        # Dynamically index the valid starts and goals for the selected maze
        valid_starts = jax.lax.dynamic_index_in_dim(self.possible_starts, maze_idx, keepdims=False)
        valid_goals = jax.lax.dynamic_index_in_dim(self.possible_goals, maze_idx, keepdims=False)

        # Determine the number of valid starts and goals (non-zero rows)
        num_starts = jp.count_nonzero(jp.any(valid_starts != 0, axis=1))
        num_goals = jp.count_nonzero(jp.any(valid_goals != 0, axis=1))

        # Sample a random start and target index
        start_idx = jax.random.randint(rng_start, (), 0, num_starts)
        goal_idx = jax.random.randint(rng_target, (), 0, num_goals)

        # Extract start and target positions
        start = valid_starts[start_idx]
        target = valid_goals[goal_idx]

        return start, target

