"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING


from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
from isaaclab.envs.mdp import modify_env_param

if TYPE_CHECKING:
    from isaaclab.envs import RLTaskEnv, ManagerBasedRLEnv
    

def replace_value(env, env_id, data, value, num_steps):
    if env.common_step_counter > num_steps and data != value:
        return value
    # use the sentinel to indicate “no change”
    return modify_env_param.NO_CHANGE

def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        # no-op during start, since the difficulty fraction near 0 is wasting of resource.
        return modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def _recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume initial value element and final value element have the same structure as data
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    # Otherwise it's a leaf scalar: do the interpolation
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val.item())
    else:
        # cast floats or any numeric
        return new_val.item()



class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler for curriculum learning.

    Tracks per-environment difficulty levels and adjusts them based on task performance. Difficulty increases when
    position/orientation errors fall below given tolerances, and decreases otherwise (unless `promotion_only` is set).
    The normalized average difficulty across environments is exposed as `difficulty_frac` for use in curriculum
    interpolation.

    Args:
        cfg: Configuration object specifying scheduler parameters.
        env: The manager-based RL environment.

    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        vel_tol: float = 0.1,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command("base_velocity")
        curobs = env.observation_manager.compute()['curobs']
        avg_base_lin = torch.mean(curobs['base_lin_vel'],dim=1)
        vel_error = avg_base_lin -command
        vel_error  = torch.norm(vel_error, dim=1)
        move_up = (vel_error < vel_tol) 
      
        demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up[env_ids],
            self.current_adr_difficulties[env_ids] + 1,
            demot,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac



def vel_cmd_level(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    a = torch.zeros(1)
    asset: Articulation = env.scene[asset_cfg.name]
   
    command = env.command_manager.get_command("base_velocity")
    return a 
    
    # terrain: TerrainImporter = env.scene.terrain
    # command = env.command_manager.get_command("base_velocity")
    # # compute the distance the robot walked
    # distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # # robots that walked far enough progress to harder terrains
    # move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # # robots that walked less than half of their required distance go to simpler terrains
    # move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    # move_down *= ~move_up
    # # update terrain levels
    # terrain.update_env_origins(env_ids, move_up, move_down)
    # # return the mean terrain level
    # return torch.mean(terrain.terrain_levels.float())
