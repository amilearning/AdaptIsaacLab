
import torch

class TreadmillManager:
    def __init__(self, env):
        self.env = env.env.env
        self.l_treadmill = self.env.scene["treadmill_left"]
        self.r_treadmill = self.env.scene["treadmill_right"]
        self.speed_range = (-2.0, 2.0)
        self.left_vel_command = torch.zeros(self.env.num_envs, 6, device=self.env.device)
        self.right_vel_command = torch.zeros(self.env.num_envs, 6, device=self.env.device)
        self.device = self.env.device

    def reset(self):
        flags = torch.randint(0, 2, (self.env.num_envs,), device=self.device)
        l_rand_speed = torch.rand(self.env.num_envs, device=self.device) * (self.speed_range[1] - self.speed_range[0]) + self.speed_range[0]
        r_rand_speed = torch.rand(self.env.num_envs, device=self.device) * (self.speed_range[1] - self.speed_range[0]) + self.speed_range[0]
        l_rand_speed = torch.where(flags.bool(), r_rand_speed, l_rand_speed)
        self.left_vel_command[:, 0] = l_rand_speed
        self.right_vel_command[:, 0] = r_rand_speed


    def single_treadmill(self):
        self.l_treadmill.write_root_velocity_to_sim(self.left_vel_command)
        self.r_treadmill.write_root_velocity_to_sim(self.right_vel_command)