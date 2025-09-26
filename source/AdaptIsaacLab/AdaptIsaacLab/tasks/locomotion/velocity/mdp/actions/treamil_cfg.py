from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
import torch
from isaaclab.utils import configclass


def constant_commands(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)



class TreadmillActionTerm(ActionTerm):

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg, env: ManagerBasedRLEnvCfg):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    """
    Operations
    """
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):        
        # set velocity targets
        self._vel_command[:, :3] = self._processed_actions
        self._asset.write_root_velocity_to_sim(self._vel_command)



@configclass
class TreadmillActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = TreadmillActionTerm
    """The class corresponding to the action term."""

