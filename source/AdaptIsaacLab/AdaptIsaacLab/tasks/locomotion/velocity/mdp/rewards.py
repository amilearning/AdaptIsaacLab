from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()



def lin_vel_w_x_lz2(env: ManagerBasedRLEnv,  robot_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]    
    return  torch.square(robot_asset.data.root_lin_vel_w[:,0]) 



def track_boundary_l2(env: ManagerBasedRLEnv, 
                      robot_asset_cfg: SceneEntityCfg, 
                      origin_asset_cfg: SceneEntityCfg,
                      mode: str = "quadratic" ) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    origin_viz_asset: RigidObject = env.scene[origin_asset_cfg.name]

    robot_y_pose =  robot_asset.data.root_pos_w[:,1] 
    treadmill_y_pose =  origin_viz_asset.data.root_pos_w[:,1] 

    track_width = origin_viz_asset.cfg.spawn.size[1] 
    track_half_w = 0.5 * track_width

    overflow = torch.relu(torch.abs(robot_y_pose - treadmill_y_pose) - track_half_w)  # [N

    if mode == "linear":
        pen = overflow
    else:  # "quadratic"
        pen = overflow ** 2
    return -pen


def track_yaw_alignment_l2(env: ManagerBasedRLEnv, 
                      robot_asset_cfg: SceneEntityCfg, 
                      origin_asset_cfg: SceneEntityCfg,
                      mode: str = "quadratic",
                       deadband_rad: float = 0.1 ) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    origin_viz_asset: RigidObject = env.scene[origin_asset_cfg.name]
    t_roll, t_pich, t_yaw  = euler_xyz_from_quat(origin_viz_asset.data.root_pose_w[:,3:])
    r_roll, r_pich, r_yaw  = euler_xyz_from_quat(robot_asset.data.root_pose_w[:,3:])
    d = wrap_to_pi(t_yaw - r_yaw).abs()    
    d = torch.clamp_min(d - deadband_rad, 0.0)
    if mode == "linear":
        pen = d
    else:  # quadratic
        pen = d * d    
    return -pen


