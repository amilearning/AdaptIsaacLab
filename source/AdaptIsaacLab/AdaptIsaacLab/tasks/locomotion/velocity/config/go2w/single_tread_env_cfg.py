from isaaclab.utils import configclass

from AdaptIsaacLab.tasks.locomotion.velocity.treadmill_env_cfg import TreadmillEnvCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

UNITREE_GO2W_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/frl/robots_model/unitree_model/Go2W/usd/go2w.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.7,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness={
                ".*_hip_.*": 25.0,
                ".*_thigh_.*": 25.0,
                ".*_calf_.*": 25.0,
                ".*_foot_.*": 0,
            },
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
    ],
    # fmt: on
)

@configclass
class Go2wTreadmillEnvCfg(TreadmillEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/go2w")

@configclass
class Go2wTreadmillEnvCfg_PLAY(Go2wTreadmillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False

        # # disable randomization for play
        # self.observations.policy.enable_corruption = False
        # # remove random pushing
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
