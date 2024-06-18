import gymnasium as gym
from enum import Enum
import numpy as np
import random

from dm_control.utils import rewards


# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.225

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1

class AvailableEnvironments(Enum):
    Walker2d = "Walker2d-v4"
    Walker2dtest = "Walker2d-v5"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class Environment:
    def __init__(self, env_name, mode, seed=None):
        if(seed is not None):
            np.random.seed(seed)
            random.seed(seed)
        self.env = gym.make(env_name, render_mode=mode)
        self.observation = self.reset(seed=seed)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.iter = 0
        self.total_action_cost = 0
        self.done = False
        self.left_dominant = True
        self.last_left_thigh_angle=0
        self.last_right_thigh_angle=0
        self.time = 0
        self.left_time = 0
        self._move_speed = _WALK_SPEED
        self.fitness = 0

    def step(self, action):
        self._last_action = action
        self._last_observation = self.observation

        self.total_action_cost += self.compute_action_cost(action)

        self.observation, reward, terminated, truncated, info  = self.env.step(action)
        self.done = terminated or truncated
        if not self.done:
            self.iter += 1 
        return self.observation, reward, self.done, info

    def reset(self, seed=None):
        if(seed!=None):
            self.observation = self.env.reset(seed=seed)[0]
        else:
            self.observation = self.env.reset()[0]
        self.iter = 0
        self.total_action_cost = 0
        self.done = False
        self.left_dominant = True
        self.last_left_thigh_angle=0
        self.last_right_thigh_angle=0
        self.time = 0
        self.left_time = 0
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

    def get_fitness(self):
        left_usage = self.left_time/self.time
        right_usage = 1 - left_usage
        alternate_legs = left_usage + right_usage - np.abs(left_usage - right_usage)
        print("Fitness totale",self.get_fitness)
        print("Peso legs", alternate_legs)
        return self.fitness*alternate_legs

    def compute_fitness(self):
        self.time += 1
        torso_height = self.observation[0]
        torso_velocity = self.observation[8]
        standing = rewards.tolerance(torso_height,
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=0.3,
                                    value_at_margin=0.0,
                                    sigmoid='linear'
                                    )
        
        angle_torso = self.observation[1]
        torso_upright = np.cos(angle_torso)
        upright = (1 + torso_upright) / 2
        stand_reward = (3*standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        move_reward = rewards.tolerance(torso_velocity,
                                bounds=(_WALK_SPEED-0.1, _WALK_SPEED+0.1),
                                margin=_WALK_SPEED/2,
                                value_at_margin=0.0,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6
        # alternate legs reward

        action_cost = 0

        for i in range(6):
            action_cost += self._last_action[i]**2
        action_cost = action_cost/6
        action_cost = 1 - action_cost


        reward_torso = rewards.tolerance(angle_torso,
                                bounds=(-0.21, 0.21),
                                margin=0.05,
                                value_at_margin=0.0,
                                sigmoid='linear')
        reward_torso = (1 + 3*reward_torso) / 4

        angle_right_thigh = self.observation[2]
        angle_left_thigh = self.observation[5]

        if self.left_dominant:
            self.left_time += 1
            if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_left_thigh > angle_right_thigh:
                self.left_dominant = False
            
            # else:
            #     if angle_left_thigh >= self.last_left_thigh_angle:
            #         left_reward = 1
            #     else:
            #         left_reward = 0
            #     if angle_right_thigh <= self.last_right_thigh_angle:
            #         right_reward = 1
            #     else:
            #         right_reward = 0
        else:
            if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_right_thigh > angle_left_thigh:
                self.left_dominant = True   
                
            # else:
            #     if angle_left_thigh <= self.last_left_thigh_angle:
            #         left_reward = 1
            #     else:
            #         left_reward = 0
            #     if angle_right_thigh >= self.last_right_thigh_angle:
            #         right_reward = 1
            #     else:
            #         right_reward = 0

        # self.last_left_thigh_angle = angle_left_thigh
        # self.last_right_thigh_angle = angle_right_thigh
        
        # alternate_legs=((left_reward+right_reward) + left_reward*right_reward)/3

        # #time_tolerance = rewards.tolerance(self.left_time/self.time,bounds=(0.45,0.55),margin=0.1,value_at_margin=0.0,sigmoid='linear')
        

        # alternate_legs = (1 + 3*alternate_legs) / 4
        #print("self.left_dominant:",self.left_dominant,"angle_left_thigh:",angle_left_thigh,"angle_right_thigh:",angle_right_thigh)
        #print("walk_std:",walk_std,"alternate_legs:",alternate_legs,"move_reward:",move_reward,"standing",standing,"upright:",upright,"reward_torso:",reward_torso)
        self.fitness += walk_std*action_cost*reward_torso


    # def fitness(self):
    #     """Fitness function for the environment"""
    #     # Take the wrapped environment
    #     # env = self.env.unwrapped.data.xmat['torso', 'zz']
    #     # # Get the data from the environment env

    #     #id = self.env.unwrapped.model.body_name2id('torso')
    #     # unwrapped = self.env.unwrapped


    #     # torso = unwrapped.get_body_com("torso")
    #     # #print("Torso: ", torso)
    #     # model = self.env.unwrapped.model
    #     # data = self.env.unwrapped.data
    #     # xmat = data.xmat
    #     # print("unwrapped: ", dir(unwrapped))
    #     # print("model: ", dir(model))
    #     # print("data: ", dir(data))
    #     #print("Xmat: ", xmat)


    #     # print("dir(data): ", dir(data))
    #     # print("dir(model): ", dir(model))
    #     # print("names", model.names)
    #     #xmat = self.env.unwrapped.data.xmat[id]
    #     #print("Xmat: ", xmat)

    #     torso_height = self.observation[0]
    #     torso_velocity = self.observation[8]

    #     angle = self.observation[1]
    #     torso_upright = np.cos(angle)

    #     #print("Torso upright: ", torso_upright,"xmat", xmat[1][8],"equal?", xmat[1][8] == torso_height)

    #     standing = rewards.tolerance(torso_height,
    #                                 bounds=(_STAND_HEIGHT, float('inf')),
    #                                 margin=_STAND_HEIGHT/4)
    #     upright = (1 + torso_upright) / 2
    #     stand_reward = (3*standing + upright) / 4
    #     if self._move_speed == 0:
    #         return stand_reward
    #     move_reward = rewards.tolerance(torso_velocity,
    #                             bounds=(self._move_speed-0.1, self._move_speed+0.1),
    #                             margin=self._move_speed/2,
    #                             value_at_margin=0.5,
    #                             sigmoid='linear')
    #     walk_std = stand_reward * (5*move_reward + 1) / 6
    #     # alternate legs reward

    #     # Get all the control values and calculate the action cost
    #     action_cost = 0
    #     # dev_std = 0.175
    #     # norm = 1/(2*np.pi*dev_std**2)**0.5
    #     # for i in range (6):
    #     #     # gaussian action cost with mean 0 and std 0.3
    #     #     action_cost += (1/(2*np.pi*dev_std**2)**0.5*np.exp(-0.5*(self._last_action[i]/dev_std)**2))/norm
    #     # action_cost = action_cost/6

    #     # for i in range(6):
    #     #     action_cost += self._last_action[i]**2
    #     # action_cost = 1 - action_cost/6
    #     # action_cost = rewards.tolerance(action_cost, bounds=(0.75, 1), margin=0.75, value_at_margin=0.5, sigmoid='linear')

    #     # for i in range(6):
    #     #         action_cost += self._last_action[i]**2
    #     # action_cost = action_cost/6
    #     # action_cost = rewards.tolerance(action_cost, bounds=(0.0, 0.2),value_at_margin=0.2, margin=0.50, sigmoid='gaussian')

    #     for i in range(6):
    #         action_cost += np.abs(self._last_action[i])
    #     action_cost = action_cost/6
    #     action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')

    #     angle_right_leg = self.observation[2]
    #     angle_left_leg = self.observation[5]

    #     if self.left_dominant:
    #         self.left_dominant = False
    #         if angle_left_leg >= angle_right_leg + (15/180*np.pi):
    #             self.alternate_legs = 1
    #         else:
    #             self.alternate_legs = 0
    #     else:
    #         self.left_dominant = True
    #         if angle_left_leg <= angle_right_leg - (15/180*np.pi):
    #             self.alternate_legs = 1
    #         else:
    #             self.alternate_legs = 0


    #     reward_alternate_legs = (1 + 3*self.alternate_legs) / 4
    #     print("walk_std: ", walk_std, "action_cost: ", action_cost, "reward_alternate_legs: ", reward_alternate_legs,"angle_right_leg: ", angle_right_leg, "angle_left_leg: ", angle_left_leg)
    #     fitness = walk_std*action_cost*reward_alternate_legs
    #     return fitness