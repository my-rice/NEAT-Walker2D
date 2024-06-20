import gymnasium as gym
from enum import Enum
import numpy as np
import random
import math
from dm_control.utils import rewards
import time

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
        # change step number
        self.env._max_episode_steps = 1000
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
        self.right_time = 0
        self._move_speed = _WALK_SPEED
        self.current_fitness = 0
        self.total_fitness = 0

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
        self.right_time = 0
        self.total_fitness = 0
        self.current_fitness = 0
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
        #return self.computed_walk_alternate()
        return self.just_alternate_legs()

    def just_alternate_legs(self):
        alternate_legs = (self.left_time+self.right_time)-np.abs(self.left_time-self.right_time)+0.25
        #print("alternate_legs",alternate_legs)
        return alternate_legs
        # if(self.left_time+self.right_time==0):
        #     return 0
        # alternate_legs = (self.left_time+self.right_time-np.abs(self.left_time-self.right_time))
        # return alternate_legs
    
    def computed_walk_alternate(self): 
        '''
            Weights the reward based on the time spent on each leg going forward
        '''
        if(self.left_time+self.right_time==0):
            return 0
        alternate_legs = (self.left_time+self.right_time-np.abs(self.left_time-self.right_time))/(self.left_time+self.right_time)
        return alternate_legs*self.current_fitness

    def fitness(self):
        #self.computed_walk_alternate_fitness_with_legs()
        self.computed_walk_alternate_fitness()

    def computed_walk_alternate_fitness_with_legs(self):


        
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
                                value_at_margin=0.1,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6

        action_cost = 0

        for i in range(6):
            action_cost += np.abs(self._last_action[i])
        action_cost = action_cost/6

        action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.225, sigmoid='hyperbolic')

        reward_torso = rewards.tolerance(angle_torso,
                                bounds=(-0.21, 0.21),
                                margin=0.05,
                                value_at_margin=0.0,
                                sigmoid='linear')
        reward_torso = (1 + 3*reward_torso) / 4

        angle_right_thigh = self.observation[2]
        angle_left_thigh = self.observation[5]
        angle_right_leg = self.observation[3]
        angle_left_leg = self.observation[6]
        # print("angle_right_thigh",angle_right_thigh, "angle_left_thigh",angle_left_thigh, "angle_right_leg",angle_right_leg, "angle_left_leg",angle_left_leg)
        difference = np.abs(angle_left_thigh-angle_right_thigh)
        weight = rewards.tolerance(difference,bounds=(0.0,0.8),margin=0.5,value_at_margin=0.1,sigmoid='linear')
        difference = weight*difference

        zero_neighbor_up = 0.12
        zero_neighbor_down = -0.12
        if self.left_dominant:
            if(torso_velocity*0.002>0):
                self.left_time += torso_velocity*0.002*difference
        else:
            if(torso_velocity*0.002>0):
                self.right_time += torso_velocity*0.002*difference

        if angle_left_thigh > angle_right_thigh+math.radians(45) and angle_left_leg > zero_neighbor_down and angle_left_leg < zero_neighbor_up:
            self.left_dominant = True
            
           
        if angle_right_thigh > angle_left_thigh+math.radians(45) and angle_right_leg > zero_neighbor_down and angle_right_leg < zero_neighbor_up:
            self.left_dominant = False   
        
          
     
        
        self.current_fitness += walk_std*action_cost*reward_torso

    def computed_walk_alternate_fitness(self):
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
                                value_at_margin=0.1,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6

        action_cost = 0

        for i in range(6):
            action_cost += np.abs(self._last_action[i])
        action_cost = action_cost/6

        action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.225, sigmoid='hyperbolic')

        reward_torso = rewards.tolerance(angle_torso,
                                bounds=(-0.21, 0.21),
                                margin=0.05,
                                value_at_margin=0.0,
                                sigmoid='linear')
        reward_torso = (1 + 3*reward_torso) / 4

        angle_right_thigh = self.observation[2]
        angle_left_thigh = self.observation[5]
        difference = np.abs(angle_left_thigh-angle_right_thigh)
        difference = rewards.tolerance(difference,bounds=(0.0,0.8),margin=0.5,value_at_margin=0.1,sigmoid='linear')
        if self.left_dominant:
            if(torso_velocity*0.002>0):
                self.left_time += torso_velocity*0.002#*difference
        else:
            if(torso_velocity*0.002>0):
                self.right_time += torso_velocity*0.002#*difference
        if angle_left_thigh > angle_right_thigh+math.radians(45):
            self.left_dominant = True
            
           
        if angle_right_thigh > angle_left_thigh+math.radians(45):
                self.left_dominant = False   
        
          
     
        self.current_fitness = walk_std*action_cost*reward_torso
        self.total_fitness += self.current_fitness

    def computed_time_alternate(self):
        left_usage = self.left_time/self.time
        right_usage = 1 - left_usage
        alternate_legs = left_usage + right_usage - np.abs(left_usage - right_usage)
        alternate_legs = rewards.tolerance(alternate_legs,bounds=(0.85,1.0),margin=0.15,value_at_margin=0.1, sigmoid="gaussian")
        return self.current_fitness*alternate_legs
    
    def computed_time_alternate_fitness(self):

        
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
                                value_at_margin=0.1,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6
        # alternate legs reward

        action_cost = 0

        for i in range(6):
            action_cost += np.abs(self._last_action[i])
        action_cost = action_cost/6

        action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.225, sigmoid='hyperbolic')

        reward_torso = rewards.tolerance(angle_torso,
                                bounds=(-0.21, 0.21),
                                margin=0.05,
                                value_at_margin=0.0,
                                sigmoid='linear')
        reward_torso = (1 + 3*reward_torso) / 4

        angle_right_thigh = self.observation[2]
        angle_left_thigh = self.observation[5]
      
            # else:
        if self.left_dominant:
            self.left_time += 1

        if angle_left_thigh > angle_right_thigh+math.radians(30):
            self.left_dominant = True
            
        
        if angle_right_thigh > angle_left_thigh+math.radians(30):
                self.left_dominant = False   
        
          
     
        self.total_fitness += walk_std*action_cost*reward_torso
    
    def compute_time_instant_fitness(self):
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
            if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_left_thigh > angle_right_thigh:
                self.left_dominant = False
                left_reward = 1
                right_reward = 1
            else:
                if angle_left_thigh >= self.last_left_thigh_angle:
                    left_reward = 1
                else:
                    left_reward = 0
                if angle_right_thigh <= self.last_right_thigh_angle:
                    right_reward = 1
                else:
                    right_reward = 0
        else:
            if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_right_thigh > angle_left_thigh:
                self.left_dominant = True   
                left_reward = 1
                right_reward = 1
            else:
                if angle_left_thigh <= self.last_left_thigh_angle:
                    left_reward = 1
                else:
                    left_reward = 0
                if angle_right_thigh >= self.last_right_thigh_angle:
                    right_reward = 1
                else:
                    right_reward = 0

        self.last_left_thigh_angle = angle_left_thigh
        self.last_right_thigh_angle = angle_right_thigh

        alternate_legs=((left_reward+right_reward) + left_reward*right_reward)/3


        
        alternate_legs = (1 + 3*alternate_legs) / 4

        self.current_fitness = alternate_legs*walk_std*action_cost*reward_torso
        self.total_fitness += self.current_fitness

    def compute_standard_fitness(self):
        

        torso_height = self.observation[0]
        torso_velocity = self.observation[8]

        angle = self.observation[1]
        torso_upright = np.cos(angle)


        standing = rewards.tolerance(torso_height,
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT/4)
        upright = (1 + torso_upright) / 2
        stand_reward = (3*standing + upright) / 4
        move_reward = rewards.tolerance(torso_velocity,
                                bounds=(self._move_speed-0.1, self._move_speed+0.1),
                                margin=self._move_speed/2,
                                value_at_margin=0.5,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6
        # alternate legs reward

        action_cost = 0


        # for i in range(6):
        #     action_cost += self._last_action[i]**2
        # action_cost = 1 - action_cost/6
        # action_cost = rewards.tolerance(action_cost, bounds=(0.75, 1), margin=0.75, value_at_margin=0.5, sigmoid='linear')

        # for i in range(6):
        #         action_cost += self._last_action[i]**2
        # action_cost = action_cost/6
        # action_cost = rewards.tolerance(action_cost, bounds=(0.0, 0.2),value_at_margin=0.2, margin=0.50, sigmoid='gaussian')

        for i in range(6):
            action_cost += np.abs(self._last_action[i])
        action_cost = action_cost/6
        action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')

        self.current_fitness = walk_std*action_cost 
        self.total_fitness += self.current_fitness

