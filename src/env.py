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
        self.env = gym.make(env_name, render_mode="rgb_array")
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
        self._last_action = action
        self._last_observation = self.observation

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
        # Take the wrapped environment
        # env = self.env.unwrapped.data.xmat['torso', 'zz']
        # # Get the data from the environment env

        #id = self.env.unwrapped.model.body_name2id('torso')
        # unwrapped = self.env.unwrapped


        # torso = unwrapped.get_body_com("torso")
        # #print("Torso: ", torso)
        # model = self.env.unwrapped.model
        # data = self.env.unwrapped.data
        # xmat = data.xmat
        # print("unwrapped: ", dir(unwrapped))
        # print("model: ", dir(model))
        # print("data: ", dir(data))
        #print("Xmat: ", xmat)


        # print("dir(data): ", dir(data))
        # print("dir(model): ", dir(model))
        # print("names", model.names)
        #xmat = self.env.unwrapped.data.xmat[id]
        #print("Xmat: ", xmat)

        torso_height = self.observation[0]
        torso_velocity = self.observation[8]

        angle = self.observation[1]
        torso_upright = np.cos(angle)

        #print("Torso upright: ", torso_upright,"xmat", xmat[1][8],"equal?", xmat[1][8] == torso_height)

        standing = rewards.tolerance(torso_height,
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT/4)
        upright = (1 + torso_upright) / 2
        stand_reward = (3*standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        move_reward = rewards.tolerance(torso_velocity,
                                bounds=(self._move_speed, self._move_speed),
                                margin=self._move_speed/2,
                                value_at_margin=0.0,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6
        # alternate legs reward

        # Get all the control values and calculate the action cost
        action_cost = 0
        for i in range(6):
            action_cost += self._last_action[i]**2
        action_cost = 1 - action_cost/6

        for i in range(6):
                action_cost += self._last_action[i]**2
        action_cost = action_cost/6
        action_cost = rewards.tolerance(action_cost, bounds=(0.0, 0.2),value_at_margin=0.0, margin=0.30, sigmoid='linear')


        fitness = walk_std*action_cost 
        return fitness