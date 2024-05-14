import gymnasium as gym
from enum import Enum
import numpy as np


from dm_control.utils import rewards


# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1

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

        self._move_speed = _WALK_SPEED


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


        print("RIGHT self.observation[2]: ", self.observation[2], "in degrees: ", np.degrees(self.observation[2]))
        print("LEFT self.observation[5]: ", self.observation[5], "in degrees: ", np.degrees(self.observation[5]))
        print("HEIGHT self.observation[0]: ", self.observation[0])
        print("UPRIGHT = angle of the torso. self.observation[1]: ", self.observation[1])
        # compute the degree in radiant bewteen the two legs
        stride_rad = np.absolute(self.observation[2] - self.observation[5])
        print("Stride in radiant: ", stride_rad, "in degrees: ", np.degrees(stride_rad))


        standing = rewards.tolerance(self.observation[0],
                                bounds=(_STAND_HEIGHT, float('inf')),
                                margin=_STAND_HEIGHT/2)
        upright = (1 + self.observation[1]) / 2
        stand_reward = (3*standing + upright) / 4

        move_reward = rewards.tolerance(self.observation[8],
                                        bounds=(self._move_speed, float('inf')),
                                        margin=self._move_speed/2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')
        reward = stand_reward * (5*move_reward + 1) / 6

        # stride = rewards.tolerance( # Falcata per la funzione di fitness
        #     stride_rad,
        #     bounds=(0.5, 0.6), # TODO: CAPIRE COME FUNZIONANO I BOUNDS. 0.5 radiant = 28.65 degrees, 0.6 radiant = 34.38 degrees
        #     sigmoid="gaussian",
        #     margin=0.2, # 0.2 radiant = 11.46 degrees 
        #     #value_at_margin=0,
        # )
        

        #speed = np.absolute(self.observation[8]) # Nella documentazione di Walker2d è indicato come la velocità, se questo è vero non serve memorizzare l'ultima osservazione
        control_cost = self.total_action_cost 

        fitness = move_reward + stand_reward 

        #fitness = alive_bonus+speed+control_cost
        #print("alive_bonus: ", alive_bonus, "speed: ", speed, "control_cost: ", control_cost, "fitness: ", fitness)
        print("move_reward: ", move_reward, "stand_reward: ", stand_reward, "fitness: ", fitness)
        return fitness
