import gymnasium as gym
from enum import Enum
import numpy as np
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
        self.time_step = self.env.unwrapped.dt
        print("Time step: ", self.time_step)
        self.iter = 0
        self.total_action_cost = 0
        self.done = False

    def step(self, action):
        self.last_observation = self.observation

        self.total_action_cost += self.compute_action_cost(action)

        self.observation, reward, terminated, truncated, info  = self.env.step(action)
        self.done = terminated or truncated
        if not self.done:
            self.iter += 1 
        return self.observation, reward, self.done, info

    def reset(self):
        self.observation = self.env.reset()
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

    def fitness(self):
        """Fitness function for the environment"""
        alive_bonus = self.iter*self.time_step

        # compute the distance from the last observation
        #distance = np.linalg.norm(self.observation[8] - self.last_observation[8])
        #speed = distance/self.time_step
        speed = np.absolute(self.observation[8]) # Nella documentazione di Walker2d è indicato come la velocità, se questo è vero non serve memorizzare l'ultima osservazione
        control_cost = self.total_action_cost 
        fitness = alive_bonus+speed+control_cost
        print("alive_bonus: ", alive_bonus, "speed: ", speed, "control_cost: ", control_cost, "fitness: ", fitness)
        return fitness
