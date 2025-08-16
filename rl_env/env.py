"""
Created on 2025/8/1 
Author: Hao Chen (chen960216@gmail.com)
"""
import copy
import random
from typing import Optional, Dict, Tuple

import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rl_env.constant import *
from rl_env.env_proto import Env
from rl_env.utlis import generate_object_classes, generate_grid
from rl_env.stability import calculate_stability, update_maps, Item, draw_cuboid


# TODO: stop criteria
# TODO: reward function


class EnvState:
    _cache = {}
    _grid = None

    def __init__(self,
                 scan_index: int,
                 height_map: np.ndarray,
                 feasibility_map: np.ndarray,
                 pallet_space: np.ndarray,
                 item_xyz_num_dict: Optional[Dict] = None, ):
        assert isinstance(height_map, np.ndarray), "Height map must be a numpy array."
        assert isinstance(feasibility_map, np.ndarray), "Feasibility map must be a numpy array."
        assert height_map.ndim == 2, "Height map must be a 2D array."
        assert feasibility_map.ndim == 3, "Feasibility map must be a 3D array."
        assert pallet_space.ndim == 3, "Pallet space must be a 3D array."
        self.scan_index = scan_index  # Position of the scanner, can be set later
        self.height_map = height_map.copy()
        self.feasibility_map = feasibility_map.copy()
        self.pallet_space = pallet_space.copy()  # 3D grid representing the pallet space
        self.item_xyz_num_dict = copy.deepcopy(item_xyz_num_dict) if item_xyz_num_dict is not None else {}

    @classmethod
    def init_env(cls, pallet_dims: Tuple[int, int, int], ):
        """
        Initializes the environment state with empty maps and pallet space.
        :param pallet_dims:
        :return:
        """
        cls._grid = generate_grid(pallet_dims)

    @property
    def dp_coord(self):
        """
        Returns the coordinates of the scan cell.
        :return: Tuple[int, int, int]
        """
        if self._grid is None:
            self._grid = generate_grid(dimensions=self.pallet_space.shape[:3])
            # raise ValueError("Grid has not been initialized. Call init_env first.")
        return self._grid[self.scan_index]

    @property
    def manifest(self) -> np.ndarray:
        """
        Returns a dictionary of items in the manifest.
        :return: Dict[str, Item]
        """
        return np.array([(*k, v) for k, v in self.item_xyz_num_dict.items()])

    @property
    def remaining_items_total(self) -> int:
        return int(sum(self.item_xyz_num_dict.values()))

    @property
    def remaining_item_height(self) -> np.ndarray:
        """
        Returns the total height of all remaining items.
        :return: int
        """
        return np.array([dim[2] for dim, num in self.item_xyz_num_dict.items() if num > 0])

    def _max_height(self) -> int:
        # global max z; height_map is shape (X,Y)
        return int(self.height_map.max())

    def _occupied_volume(self) -> int:
        # number of filled voxels
        return int(self.pallet_space.sum())

    def _footprint_bbox(self):
        """
        Returns xmin, xmax, ymin, ymax for the tight axis-aligned bounding
        rectangle of the current footprint (any z>0). If empty, returns None.
        """
        occ_xy = self.pallet_space.any(axis=2)  # (X,Y) footprint
        xs, ys = np.where(occ_xy)
        if xs.size == 0:
            return None
        xmin, xmax = xs.min(), xs.max() + 1  # make xmax/ymax exclusive
        ymin, ymax = ys.min(), ys.max() + 1
        return xmin, xmax, ymin, ymax

    def _bbox_base_area(self):
        bbox = self._footprint_bbox()
        if bbox is None:
            return 0, None
        xmin, xmax, ymin, ymax = bbox
        area = int((xmax - xmin) * (ymax - ymin))
        return area, bbox

    def _bbox_max_height(self) -> int:
        """
        Max column height within the footprint bbox only.
        """
        area, bbox = self._bbox_base_area()
        if area == 0:
            return 0
        xmin, xmax, ymin, ymax = bbox
        # height_map is (X,Y)
        return int(self.height_map[xmin:xmax, ymin:ymax].max())

    def _bbox_projection_volume(self) -> int:
        """
        Sum of column heights within the footprint bbox only (i.e., the volume
        of the solid formed by projecting current columns down to z=0 but *only*
        over the boundary area).
        """
        area, bbox = self._bbox_base_area()
        if area == 0:
            return 0
        xmin, xmax, ymin, ymax = bbox
        return int(self.height_map[xmin:xmax, ymin:ymax].sum())

    def _fits_any_item_collision_free(self, coord) -> bool:
        """
        Fast check: does ANY remaining item (either orientation) fit at coord
        using only bounds + collision (no stability)?
        """
        x0, y0, z0 = coord
        X, Y, Z = self.pallet_space.shape
        # quick reject if no inventory
        if not self.item_xyz_num_dict:
            return False

        # iterate over remaining SKUs
        for (w0, d0, h0), cnt in self.item_xyz_num_dict.items():
            if cnt <= 0:
                continue
            # try both orientations
            for (w, d) in ((w0, d0), (d0, w0)):
                # bounds
                if x0 + w > X or y0 + d > Y or z0 + h0 > Z:
                    continue
                # collision
                sub = self.pallet_space[x0:x0 + w, y0:y0 + d, z0:z0 + h0]
                if not np.any(sub):  # empty == collision-free
                    return True
        return False

    def _next_unoccupied_after(self, start_idx: int, pallet_space: np.ndarray) -> int:
        """
        Returns the first scan index >= start_idx whose voxel is empty (0).
        If none remain, clamps to the last index.
        """
        if self._grid is None:
            # fall back to current pallet shape if needed
            self._grid = generate_grid(dimensions=pallet_space.shape[:3])

        N = len(self._grid)
        idx = start_idx
        while idx < N:
            # x, y, z = self._grid[idx]
            # if pallet_space[x, y, z] == 0:
            #     break
            if self._fits_any_item_collision_free(self._grid[idx]):
                break
            idx += 1
        # clamp so dp_coord stays valid even if everything is full
        return min(idx, N - 1)

    @property
    def compactness(self) -> float:
        occ = self._occupied_volume()
        area, bbox = self._bbox_base_area()
        if area == 0:
            return 0.0
        Hmax = self._bbox_max_height()
        if Hmax <= 0:
            return 0.0
        denom = area * Hmax
        return float(occ / max(denom, 1e-9))

    @property
    def pyramid(self) -> float:
        """
        P = occupied_volume / (bbox_projection_volume)
        (0 if empty). Penalizes voids/caves under overhangs inside the footprint.
        """
        occ = self._occupied_volume()
        proj = self._bbox_projection_volume()
        if proj <= 0:
            return 0.0
        return float(occ / max(proj, 1e-9))

    @property
    def compactness_wd_pallet_space(self) -> float:
        """
        Calculates the compactness of the entire pallet space.
        Compactness is defined as the ratio of occupied volume to the maximum possible volume.
        :return: float
        """
        occ = self._occupied_volume()
        Hmax = self._max_height()
        if Hmax <= 0:
            return 0.0
        base_area = np.prod(self.pallet_space.shape[:2])
        denom = base_area * Hmax
        return float(occ / max(denom, 1e-9))

    def add(self, obj_dims, position, stability_result, inv_key_dims):
        """
        Updates the environment state with a new item placement.

        Args:
            item (Item): The item to be placed in the environment.
        """
        w, d, h = obj_dims
        x, y, z = position
        pallet_space = self.pallet_space.copy()
        pallet_space[x: x + w, y: y + d, z: z + h] = 1
        height_map, feasibility_map = update_maps(Item(obj_dims, position, ),
                                                  stability_result['support_height'],
                                                  stability_result['support_polygon'],
                                                  self.height_map,
                                                  self.feasibility_map, )
        item_xyz_num_dict = self.item_xyz_num_dict.copy()
        key = tuple(inv_key_dims)
        item_xyz_num_dict[key] -= 1
        next_scan_idx = self._next_unoccupied_after(self.scan_index + 1, pallet_space)
        return EnvState(next_scan_idx,
                        height_map,
                        feasibility_map,
                        pallet_space,
                        item_xyz_num_dict)

    def do_no_action(self):
        """
        Returns a new environment state without any action taken.
        """
        next_scan_idx = self._next_unoccupied_after(self.scan_index + 1, self.pallet_space)
        return EnvState(next_scan_idx,
                        self.height_map.copy(),
                        self.feasibility_map.copy(),
                        self.pallet_space.copy(),
                        copy.deepcopy(self.item_xyz_num_dict))

    def action_feasible_lb_pos(self, start_pos: Tuple[int, int, int]) -> np.ndarray:
        """
        Returns a boolean array indicating feasible positions for placing an object
        starting from the given position.

        Args:
            start_pos (Tuple[int, int, int]): The starting position (x, y, h).

        Returns:
            np.ndarray: A boolean array indicating feasibility.
        """
        state_key = (self.pallet_space.shape, self.pallet_space.sum(), self.pallet_space.tobytes().__hash__())
        if state_key not in self._cache:
            self._cache[state_key] = {}
        if start_pos not in self._cache[state_key]:
            self._cache[state_key][start_pos] = self._action_feasible_lb_pos(start_pos)
        return self._cache[state_key][start_pos]

    def _action_feasible_lb_pos(self, start_pos):
        """
        Checks if the position is feasible for placing an object.

        Args:
            rt_pos (Tuple[int, int]): The rgt top point of a rectangle.

        Returns:
            np.ndarray: A boolean array indicating feasibility.
        """
        x, y, h = start_pos
        dim_x, dim_y, dim_z = self.pallet_space.shape
        if not (0 <= x < dim_x and 0 <= y < dim_y and 0 <= h < dim_z):
            raise ValueError(f"Position {start_pos} is out of pallet bounds {self.pallet_space.shape}")
        action_feasible_map = np.full((dim_x, dim_y), False, dtype=bool)
        space_at_h = self.pallet_space[:, :, h]
        for i in range(x + 1, dim_x):
            for j in range(y + 1, dim_y):
                if np.sum(space_at_h[x:i, y:j]) == 0:
                    action_feasible_map[i, j] = True
        return action_feasible_map

    def copy(self):
        """
        Creates a deep copy of the EnvState instance.

        Returns:
            EnvState: A new instance with copied attributes.
        """
        return EnvState(self.scan_index, self.height_map.copy(), self.feasibility_map.copy(),
                        self.pallet_space.copy(), self.item_xyz_num_dict.copy() if self.item_xyz_num_dict else {})

    def __getitem__(self, key):
        return self.pallet_space[key]

    def _repr__(self):
        return f"EnvState(scan_pos={self.scan_index},height_map={self.height_map.shape}, feasibility_map={self.feasibility_map.shape}, " \
               f"items={self.item_xyz_num_dict}, pallet_space={self.pallet_space.shape})"


class PalletPackingEnv(Env):
    """
    A reinforcement learning environment for the 3D Pallet Packing problem.

    This environment provides the basic simulation and visualization for stacking
    objects of different shapes onto a pallet. It's designed as a foundation
    for an RL agent to learn optimal packing strategies.

    The state and action spaces are left for the user to define and integrate.
    """

    def __init__(self, pallet_dims,
                 item_sku=5,
                 seed=888,
                 cog_uncertainty_ratio=0.1,
                 toggle_visualization=False):
        """
        Initializes the packing environment.

        Args:
            pallet_dims (tuple): A tuple of (width, length, height) for the pallet.
        """
        super().__init__()
        self.pallet_dims = pallet_dims
        self.item_sku = item_sku
        if EnvState._grid is None:
            EnvState.init_env(pallet_dims)
        # for stability calculation
        # --- New data structures for advanced stability ---
        self.cog_uncertainty_ratio = cog_uncertainty_ratio
        # height_map stores the max height at each (x, z) location on the floor.
        # Shape is (depth, width) to match (y, x) indexing.

        # --- For RL ---
        # self.remaining_objects = []
        self.initial_objects = [{'id': 1, 'dims': (3, 3, 2), 'count': 3}, {'id': 2, 'dims': (6, 6, 3), 'count': 8},
                                {'id': 3, 'dims': (6, 2, 2), 'count': 7}, {'id': 4, 'dims': (4, 5, 2), 'count': 8},
                                {'id': 5, 'dims': (3, 6, 2), 'count': 6}]
        # self.initial_objects = None
        self.num_placed_items = 0  # Counter for placed objects
        self.placed_objects = []  # Stores info about objects successfully placed
        self.item_xy_h_map = {}  # Maps item xy dimensions to their heights
        self.item_xyz_num_dict = {}  # Maps item dimensions to their count
        self.xy_to_xyz = {}  # (w,d) or (d,w) -> canonical (w0,d0,h)
        self.action_space = spaces.Discrete(pallet_dims[0] * pallet_dims[1] + 1)
        self.action_nums = pallet_dims[0] * pallet_dims[1] + 1
        self.total_scan_num = pallet_dims[0] * pallet_dims[1] * pallet_dims[
            2]  # Total number of scans in the environment
        self.observation_space = None  # TODO: Define your state/observation space
        self.state: Optional[EnvState] = None
        self.state_history = []  # To keep track of state changes
        self.total_iter_steps = len(EnvState._grid)

        self.DO_NOTHING_ACTION_IDX = pallet_dims[0] * pallet_dims[1]  # Index for "do nothing" action
        # --- For Visualization ---
        if toggle_visualization:
            self._prepare_visualization()
        # --- Set seed for reproducibility ---
        self.seed(seed)

    def _prepare_visualization(self):
        self._fig = plt.figure()
        self._axes = [self._fig.add_subplot(121, projection='3d'),
                      self._fig.add_subplot(122, )]
        self._colors = {}  # To store a unique color for each object ID

    def _get_object_color(self, obj_id):
        """Assigns a random, consistent color to each object ID."""
        if obj_id not in self._colors:
            self._colors[obj_id] = (random.random(), random.random(), random.random(), 0.8)  # R,G,B,Alpha
        return self._colors[obj_id]

    def _action_map_filter_item_feasible_xy_sz(self, state: EnvState, start_pos: iter, feasible_action_map: np.ndarray):
        """
        Returns a mask of endpoints (x_end, y_end) that are feasible from start_pos
        for ANY remaining item, allowing 0° and 90° rotations implicitly.
        """
        if state is None:
            raise ValueError("Environment state is not initialized. ")
        start_pos = np.asarray(start_pos)
        X, Y, Z = self.pallet_dims
        action_mask = np.zeros((X, Y), dtype=bool)
        x0, y0, z0 = start_pos
        for (w0, d0, h0), cnt in state.item_xyz_num_dict.items():
            if cnt <= 0:
                continue
            for (fw, fd) in ((w0, d0), (d0, w0)):
                # bounds check vs pallet
                if not (x0 + fw <= X and y0 + fd <= Y and z0 + h0 <= Z):
                    continue
                # collision-free check at current z slice
                sub = state.pallet_space[x0:x0 + fw, y0:y0 + fd, z0:z0 + h0]
                if np.any(sub):
                    continue
                # mark the endpoint cell (right/back at same z)
                x_end = x0 + fw
                y_end = y0 + fd
                # we use endpoint-1 to store in a cell grid
                action_mask[x_end - 1, y_end - 1] = True
        return feasible_action_map & action_mask

    def sample(self, rt_p: Optional[iter] = None) -> int:
        if rt_p is None:
            rt_p = self.state.dp_coord
        assert isinstance(rt_p, (list, tuple, np.ndarray)), "rt_p must be a list, tuple, or numpy array."
        lb_pos = self._np_random.choice(self.cal_feasible_action(rt_p))
        return int(lb_pos)

    def cal_feasible_action(self,
                            rt_p: Optional[iter] = None,
                            state: EnvState = None,
                            toggle_return_mask=False) -> np.ndarray:
        if rt_p is None:
            rt_p = self.state.dp_coord
        if state is None:
            state = self.state
        assert isinstance(rt_p, (list, tuple, np.ndarray)), "rt_p must be a list, tuple, or numpy array."
        # feasible_action_mask = self.state.action_feasible_lb_pos(rt_p)
        feasible_action_mask = np.ones((self.pallet_dims[0], self.pallet_dims[1]), dtype=bool)
        feasible_action_mask = self._action_map_filter_item_feasible_xy_sz(state, rt_p, feasible_action_mask)
        xs, ys = np.where(feasible_action_mask)
        if xs.size == 0:
            # No feasible actions available, return "do nothing" action
            return np.array([self.DO_NOTHING_ACTION_IDX], dtype=int)
        Y = self.pallet_dims[1]
        feasible_action_idx = xs * Y + ys
        return np.append(feasible_action_idx, self.DO_NOTHING_ACTION_IDX)

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        height_map = np.zeros((self.pallet_dims[0], self.pallet_dims[1]), dtype=int)
        # feasibility_map stores if a surface at (x, z, y) is stable to build on.
        # Shape is (depth, width, height).
        feasibility_map = np.full((self.pallet_dims[0],
                                   self.pallet_dims[1],
                                   self.pallet_dims[2]),
                                  False,
                                  dtype=bool)
        feasibility_map[:, :, 0] = True  # The ground level is always feasible
        pallet_space = np.zeros(self.pallet_dims, dtype=int)
        self.num_placed_items = 0  # Reset the counter for placed objects
        self.placed_objects = []
        # Create a flat list of all individual objects to be placed
        # self.remaining_objects = []
        self.item_xy_h_map = {}
        self.item_xyz_num_dict = {}
        self.xy_to_xyz = {}  # (w,d) or (d,w) -> canonical (w0,d0,h)
        # self.initial_objects = generate_object_classes((
        #     self.pallet_dims[0], self.pallet_dims[1], self.pallet_dims[2],),
        #     num_classes=self.item_sku,
        #     random_hdl=self._np_random, )
        # print(self.initial_objects)
        # print("Initial object classes generated:", self.initial_objects)
        for obj_class in self.initial_objects:
            w0, d0, h0 = obj_class['dims']
            cnt = obj_class['count']
            # inventory is tracked only by the canonical tuple
            self.item_xyz_num_dict[(w0, d0, h0)] = cnt
            # allow both footprints for action feasibility and height lookup
            self.item_xy_h_map[(w0, d0)] = h0
            self.item_xy_h_map[(d0, w0)] = h0
            # whichever orientation is used, decrement the canonical key
            self.xy_to_xyz[(w0, d0)] = (w0, d0, h0)
            self.xy_to_xyz[(d0, w0)] = (w0, d0, h0)
            # for _ in range(obj_class['count']):
            #     self.remaining_objects.append({
            #         'id': obj_class['id'],
            #         'dims': obj_class['dims']
            #     })
        # print("Environment Reset.")
        self.state_history = []
        self.state = EnvState(scan_index=0,
                              height_map=height_map,
                              feasibility_map=feasibility_map,
                              pallet_space=pallet_space,
                              item_xyz_num_dict=self.item_xyz_num_dict)
        return self.state

    def _any_feasible_move_exists(self) -> bool:
        """
        Checks if there are any feasible moves left in the environment.

        Returns:
            bool: True if there are feasible moves, False otherwise.
        """
        if len(self.item_xyz_num_dict) == 0:
            return False
        for dims, count in self.item_xyz_num_dict.items():
            if count > 0:
                return True
        return False

    def _is_valid_placement(self, obj_dims, position) -> Tuple[bool, dict]:
        """
        Checks if placing an object at a given position is valid.

        Args:
            obj_dims (tuple): The (width, d, h) of the object.
            position (tuple): The (x, y, z) coordinate of the bottom-front-left
                              corner of the object.

        Returns:
            bool: True if the placement is valid, False otherwise.
        """
        w, d, h = obj_dims
        x, y, z = position
        assert self.state, "Environment state is not initialized."
        # 1. Check boundary constraints
        if not (x >= 0 and y >= 0 and z >= 0 and
                x + w <= self.pallet_dims[0] and
                y + d <= self.pallet_dims[1] and
                z + h <= self.pallet_dims[2]):
            # print("Validation Fail: Out of bounds.")
            return False, {"is_stable": False, "reason": "Out of bounds"}
        # 2. Check for collisions with other objects
        if np.any(self.state[x:x + w, y:y + d, z:z + h] != 0):
            # print("Validation Fail: Collision detected.")
            return False, {"is_stable": False, "reason": "Collision with other objects"}
        # 3. Check for stability (must be placed on the floor or on another object)
        item_for_stability_check = Item(dimensions=(w, d, h), position=(x, y, z))
        stability_result = calculate_stability(item_for_stability_check,
                                               self.state.height_map,
                                               self.state.feasibility_map,
                                               cog_uncertainty_ratio=self.cog_uncertainty_ratio)
        return stability_result['is_stable'], stability_result

    def step(self, action: int) -> (EnvState, float, bool, dict):
        """
        Executes one time step within the environment.

        Args:
            action: The action to be taken by the agent. The structure of this
                    action needs to be defined by the user.
                    A possible structure: (object_index_in_remaining, position, rotation)
                    e.g., (3, (5, 0, 5), 0)

        Returns:
            observation (object): The agent's observation of the current environment.
            reward (float): The amount of reward returned after previous action.
            done (bool): Whether the episode has ended.
            info (dict): Contains auxiliary diagnostic information.
        """
        # --- TODO: This is where you will define your action logic ---
        # For this example, we'll assume a simple, pre-defined action format.
        # Action = (object_index_to_place, (x, y, z) position)
        scan_pos = self.state.dp_coord
        if action == self.DO_NOTHING_ACTION_IDX:
            # print("No action taken, environment remains unchanged.")
            self.state = next_state = self.state.do_no_action()
            self.state_history.append(self.state)
            done = False
            reward = REWARD_NO_ACTION
            if self.state.remaining_items_total == 0 or np.all(
                    (self.pallet_dims[2] - scan_pos[2]) < self.state.remaining_item_height):
                done = True
                reward = self.state.compactness_wd_pallet_space
            return self.state.copy(), reward, done, {}
        elif not (0 <= action <= np.prod(self.pallet_dims[:2])):
            raise ValueError(f"Action {action} is out of bounds for pallet area {self.pallet_dims[:2]}.")
        action_xy = np.unravel_index(action, self.pallet_dims[:2])
        x_start, y_start = scan_pos[:2]
        x_end, y_end = action_xy
        # position = (x_start, y_start,
        #             np.max(self.state.height_map[y_start:y_end, x_start:x_end]))
        position = scan_pos
        # height lookup allowing both rotations (we populated both in reset)
        # action_xy is (x_end-1, y_end-1). Convert back to inclusive-end footprint:
        fw = (x_end - x_start) + 1
        fd = (y_end - y_start) + 1
        h = self.item_xy_h_map.get((fw, fd))
        if not np.all(position == scan_pos):
            raise ValueError(f"Warning: Action position {action} does not match scan position {scan_pos}.")
        obj_dims = (fw, fd, h)
        # validate and place
        is_stable, stability_result = self._is_valid_placement(obj_dims, position)
        self.state_history.append(self.state)
        done = False
        if is_stable:
            inv_key = self.xy_to_xyz[(fw, fd)]  # canonical (w0,d0,h)
            # If valid, place the object
            self.state = next_state = self.state.add(obj_dims, position, stability_result, inv_key)
            self.item_xyz_num_dict[inv_key] -= 1
            # Store placement info for visualization and tracking
            self.num_placed_items += 1
            self.placed_objects.append({
                'id': self.num_placed_items,
                'dims': obj_dims,
                'pos': position
            })
            # Remove the object from the list of remaining ones

            # --- TODO: Define your reward function ---
            # A simple reward could be the volume of the placed object.
            # reward = REWARD_PLACEMENT_SUCCESS
            reward = .1
            # print(f"Action successful: "
            #       f"Placed object ({obj_dims}) at {position}. Reward: {reward}")
        else:
            # --- TODO: Define your penalty for invalid moves ---
            # reward = REWARD_PLACEMENT_FAILURE  # Penalty for attempting an invalid placement
            # print(f"Action failed: Invalid placement of object ({obj_dims}) at {position}. Penalty: {reward}")
            # self.state = next_state = self.state.do_no_action()
            return self.state.do_no_action(), REWARD_PLACEMENT_FAILURE, True, {}
        # --- TODO: Define your terminal condition (done) ---
        # if not self.remaining_objects:
        #     done = True
        #     print("Episode finished: All objects placed.")
        # You might also end the episode if no valid moves are possible.
        if self.state.remaining_items_total == 0 or np.all(
                (self.pallet_dims[2] - scan_pos[2]) < self.state.remaining_item_height):
            done = True
            reward = (self.state.compactness_wd_pallet_space + self.state.pyramid - 2) * 5
            # print("Episode finished: All objects placed.")
        # The next state, reward, done flag, and an info dict
        return self.state.copy(), reward, done, {}

    def render(self, mode='human', close=False):
        """
        Renders the current state of the pallet and stacked objects.
        """
        # --- 1. 3D Bin View ---
        ax1, ax2 = self._axes
        ax1.clear()
        # Set the axes properties
        ax1.set_xlim([0, self.pallet_dims[0]])
        ax1.set_ylim([0, self.pallet_dims[1]])
        ax1.set_zlim([0, self.pallet_dims[2]])
        ax1.set_xlabel('W (X)')
        ax1.set_ylabel('D (Y)')
        ax1.set_zlabel('H (Z)')
        ax1.set_title('Pallet Packing State')

        # Draw the pallet base (optional, for visual context)
        for item in self.placed_objects:
            item = Item(dimensions=item['dims'],
                        position=item['pos'])
            draw_cuboid(ax1, item)

        # --- 2. Feasibility Map ---
        ax2.clear()
        ax2.set_title("Feasibility Height Map")
        z_indices = np.arange(self.state.feasibility_map.shape[2]) + 1
        broadcasted_z = z_indices.reshape(1, 1, -1)
        height_values = self.state.feasibility_map * broadcasted_z
        projected_height_map = np.max(height_values, axis=2)
        # Create a masked array where all 0 values are masked. This is how we'll make them white.
        masked_projected_map = np.ma.masked_where(projected_height_map == 0, projected_height_map)
        cmap = plt.cm.get_cmap('viridis', self.pallet_dims[2])
        # Set the color for the masked values (where our map was 0) to white.
        cmap.set_bad(color='white')
        # Display the image using the masked array and custom colormap.
        # We subtract 1 in the vmin/vmax to map back to original z-values [0, h-1] for the color scale.
        im = ax2.imshow((masked_projected_map - 1).T, cmap=cmap, origin='lower',
                        interpolation='nearest', vmin=-1, vmax=self.pallet_dims[2] - 1)
        # Add a colorbar to show what height each color represents.
        # cbar = self._fig.colorbar(im, ax=ax2, ticks=np.arange(0, self.pallet_dims[2], 5))
        # cbar.set_label('Feasible Surface Height (z)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xticks(np.arange(0, self.pallet_dims[0], 10))
        ax2.set_yticks(np.arange(0, self.pallet_dims[1], 10))
        ax2.grid(True, which='both', color='white', linewidth=0.5)

        plt.draw()
        plt.pause(0.1)  # Pause to allow the plot to update

    def copy(self):
        """
        Returns a deep copy of the current environment state.
        """
        env = copy.deepcopy(self)
        return env


if __name__ == '__main__':
    # --- Example Usage ---
    import time

    # 1. Define the pallet and the objects to be stacked
    PALLET_DIMENSIONS = (10, 10, 10)  # (width, height, depth)
    ITEM_SKU = 5

    # 2. Create the environment
    env = PalletPackingEnv(PALLET_DIMENSIONS,
                           ITEM_SKU,
                           seed=np.random.randint(0, 2 ** 30),
                           toggle_visualization=True)
    obs = env.reset()
    print("SKU =", len(env.initial_objects))
    st = time.time()
    env.render()
    acc_rews = 0
    total_iter_steps = len(EnvState._grid)
    a = time.time()
    for i in range(total_iter_steps):
        # Sample a random action (object dimensions and position)
        action = env.sample()
        next_obs, reward, done, info = env.step(action)
        acc_rews += reward
        if done or action != env.DO_NOTHING_ACTION_IDX:
            print(f"\n--- Iteration {i + 1} ---")
            # print(f"Taking action: {action} | reward: {reward} | State: {next_obs.scan_index}"
            #       f" compactness: {next_obs.compactness} | pyramid: {next_obs.pyramid} | ")
            env.render()
        # input("D")
        obs = next_obs
        if done:
            print("Done.")
            break
    print("Time consumed for all steps:", time.time() - a, "seconds")
    # for i in range(1000):
    #     action = env.sample()
    #     print(f"\n--- Step {i + 1}: Taking action {action} ---")
    #     observation, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         print("Episode has finished.")
    #         break
    ed = time.time()
    print("\n--- Summary ---")
    print("Accumulated reward:", acc_rews)
    print(f"Total time taken: {ed - st:.2f} seconds")
    print("Compactness of the final state:", env.state.compactness)
    print("Compactness using w,d of pallet of the final state:", env.state.compactness_wd_pallet_space)
    print("Pyramid of the final state:", env.state.pyramid)
    print("Number of placed items:", env.num_placed_items, "out of", sum(
        [v['count'] for v in env.initial_objects]
    ))
    print("\nSimulation finished. Close the plot window to exit.")
    plt.show()
