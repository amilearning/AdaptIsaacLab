import gymnasium as gym

from . import agents, go2_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="flat-go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_env_cfg.Go2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PPORunnerCfg",
    },
)
