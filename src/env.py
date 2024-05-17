import gymnasium as gym
from enum import Enum
import numpy as np
from dm_control.utils import rewards
# class syntax
class AvailableEnvironments(Enum):
    Walker2d = "Walker2d-v4"
    Walker2dtest = "Walker2d-v5"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class Environment:
    def __init__(self, env_name, mode):
        self.env = gym.make(env_name, render_mode=mode)
        self.env._max_episode_steps = 2000
        self.observation = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.time_step = self.env.unwrapped.dt
        self.iter = 0
        self.total_action_cost = 0
        self.done = False
        self.reward = 0
        self.index=0
        self.total_fitness=0

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index
    
    def step(self, action):
        self.last_observation = self.observation

        self.total_action_cost += self.compute_action_cost(action)

        self.observation, self.reward, terminated, truncated, info  = self.env.step(action)
        self.done = terminated or truncated
        if not self.done:
            self.iter += 1 
        return self.observation, self.reward, self.done, info

    def reset(self):
        self.observation = self.env.reset()[0] # this is needed because the first observation return by the reset method is different (in shape) compared to the one returned by the step method
        self.iter = 0
        return self.observation

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_observation_space(self):
        return self.observation_space.shape[0]

    def get_action_space(self):
        return self.action_space.shape[0]
    
    def get_current_observation(self):
        return self.observation
    
    def compute_action_cost(self, action):
        return np.linalg.norm(action)

    def get_total_fitness(self):
        return self.total_fitness
    
    def fitness(self):
        standing = rewards.tolerance(self.observation[0],
                                    bounds=(1.2, float('inf')),
                                    margin=1.2/4)
        # upright = (1 + physics.torso_upright()) / 2
        stand_reward = standing
        # if self._move_speed == 0:
        move_reward = rewards.tolerance(self.observation[8],
                                bounds=(1.0, 1.0),
                                margin=1.0/2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6
        # alternate legs reward

        # left_leg = physics.named.data.xpos['left_leg', 'x']
        # right_leg = physics.named.data.xpos['right_leg', 'x']

        # #print("left_leg:",left_leg,"right_leg:",right_leg)
        # alternate_legs = 0
        # if self.left_dominant:
        #     if left_leg > right_leg + 0.1:
        #         alternate_legs = 1
        #         # Set the dominance for the next step
        #         self.left_dominant = False
        # else:
        #     if right_leg > left_leg+ 0.1:
        #         alternate_legs = 1
        #         # Set the dominance for the next step
        #         self.left_dominant = True
        fitness=stand_reward
        self.total_fitness+=fitness
        return self.reward
