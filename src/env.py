import gymnasium as gym
from enum import Enum

# class syntax
class AvailableEnvironments(Enum):
    Walker2d = "Walker2d-v4"
    Walker2dtest = "Walker2d-v5"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="human")
        self.observation = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.done = False

    def step(self, action):
        self.last_observation = self.observation
        self.observation, reward, terminated, truncated, info  = self.env.step(action)
        self.done = terminated or truncated
        return self.observation, reward, self.done, info

    def reset(self):
        self.observation = self.env.reset()
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
    
    def fitness(self):
        return 0.0 # Placeholder for the fitness function
