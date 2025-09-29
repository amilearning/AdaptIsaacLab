from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.assets import RigidObject, RigidObjectCfg
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length



def track_out_limit(env: ManagerBasedRLEnv, 
                      robot_asset_cfg: SceneEntityCfg, 
                      origin_asset_cfg: SceneEntityCfg,
                      deadband_meter: float = 0.1,
                      mode: str = "quadratic" ) -> torch.Tensor:   
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    origin_viz_asset: RigidObject = env.scene[origin_asset_cfg.name]
    robot_y_pose =  robot_asset.data.root_pos_w[:,1] 
    treadmill_y_pose =  origin_viz_asset.data.root_pos_w[:,1] 
    track_width = origin_viz_asset.cfg.spawn.size[1] 
    track_half_w = 0.5 * track_width
    overflow = torch.abs(robot_y_pose - treadmill_y_pose) - (track_half_w+deadband_meter)  # [N
    out_of_limits = torch.any(overflow.unsqueeze(0) > 0.0,dim=0)    
    return out_of_limits


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
