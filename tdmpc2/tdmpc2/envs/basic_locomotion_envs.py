# Description: Custom environment for Humanoid tasks, to create custom reward functions. This code is a modified version of the code in the HumanoidBench repository.
import os
import sys

import numpy as np
import gymnasium as gym

from envs.wrappers.time_limit import TimeLimit


class HumanoidLocomotionWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        return obs, reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()


def register_envs():
    """
    This function registers all the environments made available by this repository.
    """
    from gymnasium.envs import register
    from envs.basic_env_elements import ROBOTS, TASKS

    for robot in ROBOTS:
        control = "pos" # interface for controlling the robot
        for task, task_info in TASKS.items():
            task_info = task_info()
            kwargs = task_info.kwargs.copy()
            kwargs["robot"] = robot
            kwargs["control"] = control
            kwargs["task"] = task
            register(
                id=f"{robot}-{task}-v0",
                entry_point="envs.basic_env_elements:HumanoidEnv",
                max_episode_steps=task_info.max_episode_steps,
                kwargs=kwargs,
            )


def make_env(cfg):
    """
    Make Humanoid environment for locomotion task.
    """

    #print("[DEBUG: basic_locomotion_env] make_env(cfg)")

    if not cfg.task.startswith("humanoid_"):
        raise ValueError("Unknown task:", cfg.task)

    register_envs()

    policy_path = cfg.get("policy_path", None)
    mean_path = cfg.get("mean_path", None)
    var_path = cfg.get("var_path", None)
    policy_type = cfg.get("policy_type", None)
    small_obs = cfg.get("small_obs", None)
    if small_obs is not None:
        small_obs = str(small_obs)

    print("small obs start:", small_obs)

    env = gym.make( # Create the custom environment with the characteristics specified in the register function.
        cfg.task.removeprefix("humanoid_"), #TODO(my-rice): modify this line to removeprefix("humanoid_"). It is not necessary, I think.
        policy_path=policy_path,
        mean_path=mean_path,
        var_path=var_path,
        policy_type=policy_type,
        small_obs=small_obs,
    )
    env = HumanoidLocomotionWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps") #TODO(my-rice): I want to try to use way less episodes. I want to focus on the potential of the reward function. A good reward function should be able to solve the task in a few episodes.
    return env
