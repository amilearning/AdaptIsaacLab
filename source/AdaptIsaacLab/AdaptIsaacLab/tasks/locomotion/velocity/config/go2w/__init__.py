import gymnasium as gym

from . import agents, single_tread_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Single-Treadmill-go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": single_tread_env_cfg.Go2wTreadmillEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WPPORunnerCfg",
    },
)
