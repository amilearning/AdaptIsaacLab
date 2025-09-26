from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import torch
import isaaclab.sim as sim_utils
import AdaptIsaacLab.tasks.locomotion.velocity.mdp as mdp
from isaaclab.assets import RigidObject, RigidObjectCfg


# @configclass
# # class CommandsCfg:
# #     """Command specifications for the MDP."""

# #     base_velocity = mdp.UniformVelocityCommandCfg(
# #         asset_name="",
# #         resampling_time_range=(10.0, 10.0),
# #         rel_standing_envs=0.02,
# #         rel_heading_envs=1.0,
# #         heading_command=False,        
# #         debug_vis=True,
# #         ranges=mdp.UniformVelocityCommandCfg.Ranges(
# #             lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
# #         ),
# #     )

@configclass
class ActionsCfg:    
    leg_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[r"(FR|FL|RR|RL)_(hip|thigh|calf)_joint"], scale=1.0, use_default_offset=False)    
    leg_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[r"(FR|FL|RR|RL)_(hip|thigh|calf)_joint"], scale=1.0, use_default_offset=False)    
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[r".*_foot_joint"], scale=1.0, use_default_offset=False)        
 
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pose = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()




@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    body_pos = RewTerm(
        func=mdp.lin_vel_w_x_lz2,
        weight=1.0,
        params={"robot_asset_cfg": SceneEntityCfg("robot")},
    )
   
    track_keep = RewTerm(
        func=mdp.track_boundary_l2,
        weight=1.0,
        params={"robot_asset_cfg": SceneEntityCfg("robot"), "origin_asset_cfg": SceneEntityCfg("origin_viz")},
    )

    yaw_keep = RewTerm(
        func=mdp.track_yaw_alignment_l2,
        weight=1.0,
        params={"robot_asset_cfg": SceneEntityCfg("robot"), "origin_asset_cfg": SceneEntityCfg("origin_viz")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)    
    cart_out_of_bounds = DoneTerm(
        func= mdp.track_out_limit,
        params={"robot_asset_cfg": SceneEntityCfg("robot"), "origin_asset_cfg": SceneEntityCfg("origin_viz")},        
    )




# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class TreadmillGround(TerrainImporterCfg):

    height = 0.0
    prim_path = "/World/ground"
    terrain_type="plane"
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg( # Material for carpet
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis=False



treadmill_length = 20.0
treadmill_width = 0.5
size = (treadmill_length, treadmill_width, 0.1)
l_init_pose = (0.0, -treadmill_width/2.0-0.0001, 0.01)
r_init_pose = (0.0, treadmill_width/2.0+0.0001, 0.01)
LEFT_TREADMILL_CFG = RigidObjectCfg(        
        spawn=sim_utils.CuboidCfg(
            size=size, #(1.0, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.2, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),            
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=l_init_pose), # (0.0, 0.0, 0.01)),
    )
RIGHT_TREADMILL_CFG = RigidObjectCfg(       
        spawn=sim_utils.CuboidCfg(
            size=size, #(1.0, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.2, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),            
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=r_init_pose), # (0.0, 0.0, 0.01)),
    )

ORIGIN_VIZ_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.1, 0.9, 0.2),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.9)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=False  # or: enable_collision=False OR disable_collision=True (see note below)
    ),
    ), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.01)),
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""
    terrain = TreadmillGround()
    treadmill_left = LEFT_TREADMILL_CFG.replace(prim_path="{ENV_REGEX_NS}/treadmill_left")
    treadmill_right = RIGHT_TREADMILL_CFG.replace(prim_path="{ENV_REGEX_NS}/treadmill_right")
    origin_viz = ORIGIN_VIZ_CFG.replace(prim_path="{ENV_REGEX_NS}/origin_viz")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# Environment configuration
##
@configclass
class TreadmillEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
 