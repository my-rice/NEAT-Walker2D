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
    """ Class that wraps the gym environment and implements the methods to interact with it."""
    def __init__(self, env_name, mode, seed=None, exponent_legs=1.0):
        if(seed is not None):
            np.random.seed(seed)
            random.seed(seed)
        self.env = gym.make(env_name, render_mode=mode, frame_skip=1)
        # change step number
        self.env._max_episode_steps = 5000
        self.observation = self.reset(seed=seed)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.iter = 0
        self.total_action_cost = 0
        self.done = False
        self.left_dominant = True
        self.last_left_thigh_angle=0
        self.last_right_thigh_angle=0
        self.time = 0.0
        self.left_time = 0
        self.right_time = 0
        self._move_speed = _WALK_SPEED
        self.current_fitness = 0
        self.total_fitness = 0
        self.exponent_legs = exponent_legs
        self.last_x_left=0
        self.last_x_right=0
        self.last_x_thigh_left=0
        self.last_x_thigh_right=0
        self.last_x_torso = 0

        self.last_x_left_foot = 0
        self.last_x_right_foot = 0

    def step(self, action):
        """ Do a step in the environment with the given action."""
        self._last_action = action
        self._last_observation = self.observation

        self.total_action_cost += self.compute_action_cost(action)

        self.observation, reward, terminated, truncated, info  = self.env.step(action)
        self.done = terminated or truncated
        if not self.done:
            self.iter += 1 
        return self.observation, reward, self.done, info

    def reset(self, seed=None):
        """ Reset the environment."""
        if(seed!=None):
            self.observation = self.env.reset(seed=seed)[0]
        else:
            self.observation = self.env.reset()[0]
        self.iter = 0
        self.total_action_cost = 0.0
        self.done = False
        self.left_dominant = True
        self.last_left_thigh_angle=0
        self.last_right_thigh_angle=0
        self.time = 0
        self.left_time = 0
        self.right_time = 0
        self.total_fitness = 0
        self.current_fitness = 0
        self.last_x_left_foot = 0
        self.last_x_right_foot = 0
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
        """ Compute the fitness of the agent."""
        #self.compute_standard_fitness_with_x_foot_constraint()
        self.compute_standard_fitness()

    def alternate_legs_exponential(self):
        """ Returns the total fitness multiplied by the exponential of the time spent on each leg going forward."""
        alternate_legs = (self.left_time+self.right_time)-np.abs(self.left_time-self.right_time)
        alternate_legs = rewards.tolerance(alternate_legs,bounds=(1.5,2),margin=0.5,value_at_margin=0.01, sigmoid="hyperbolic")
        alternate_legs+=1
        alternate_legs = alternate_legs**self.exponent_legs
        return alternate_legs

    
    def computed_walk_alternate_total_time(self): 
        '''
            Weights the reward based on the time spent on each leg going forward
        '''
        left_usage = self.left_time/1000.0 # It depends on the number of steps
        right_usage = self.right_time/1000.0
        alternate_legs = left_usage + right_usage - np.abs(left_usage - right_usage)
        alternate_legs = rewards.tolerance(alternate_legs,bounds=(2,2),margin=2,value_at_margin=0.01, sigmoid="linear")
        alternate_legs+=1
        alternate_legs = alternate_legs**self.exponent_legs
        if(alternate_legs>2):
            print("alternate_legs",alternate_legs)
        return alternate_legs
    
 
    def get_x_left_foot(self):
        """ Returns the x position of the left foot."""
        return self.env.unwrapped.data.xpos[7][0]
    
    def get_x_right_foot(self):
        """ Returns the x position of the right foot."""
        return self.env.unwrapped.data.xpos[4][0]


    def angle_constraint_alternate_legs_fitness(self):
        """ Reward function that rewards the correct alternance of legs based on the angle of the thighs, legs and feet."""
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
        if(torso_velocity<0):
            walk_std = 0
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
        angle_right_foot = self.observation[4]
        angle_left_foot = self.observation[7]
        # print("angle_right_thigh",angle_right_thigh, "angle_left_thigh",angle_left_thigh, "angle_right_leg",angle_right_leg, "angle_left_leg",angle_left_leg)
        difference = np.abs(angle_left_thigh-angle_right_thigh)
        weight = rewards.tolerance(difference,bounds=(0.0,0.8),margin=0.5,value_at_margin=0.1,sigmoid='linear')
        difference = weight*difference

        zero_neighbor_up = 0.12
        zero_neighbor_down = -0.12
        if self.left_dominant:
            if(torso_velocity>0):
                self.left_time += torso_velocity*0.008
        else:
            if(torso_velocity>0):
                self.right_time += torso_velocity*0.008

        if angle_left_thigh > angle_right_thigh+math.radians(60) and angle_left_leg > zero_neighbor_down and angle_left_leg < zero_neighbor_up and angle_left_thigh>0 and angle_right_thigh<0 and angle_left_foot>zero_neighbor_down and angle_left_foot<zero_neighbor_up:
            self.left_dominant = True
            
           
        if angle_right_thigh > angle_left_thigh+math.radians(60) and angle_right_leg > zero_neighbor_down and angle_right_leg < zero_neighbor_up and angle_right_thigh>0 and angle_left_thigh<0 and angle_right_foot>zero_neighbor_down and angle_right_foot<zero_neighbor_up:
            self.left_dominant = False   

        self.current_fitness += walk_std*action_cost*reward_torso
        return float(walk_std)

    def computed_walk_alternate_fitness_constant_velocity(self):
        """ Reward function that rewards the correct alternance of legs based on the distance covered by each leg. The distance is computed by multiplying a weight based on the velocity of the torso."""
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
        if(torso_velocity<0):
            walk_std = 0
        
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
        good_moves = rewards.tolerance(torso_velocity,bounds=(0.9,1.1),margin=0.2,value_at_margin=0.0,sigmoid='linear')
        if self.left_dominant:
            if(torso_velocity*0.002>0):
                self.left_time += good_moves*0.008
        else:
            if(torso_velocity*0.002>0):
                self.right_time += good_moves*0.008
        if angle_left_thigh > angle_right_thigh+math.radians(60) and angle_left_thigh>0 and angle_right_thigh<0:
            self.left_dominant = True
           
        if angle_right_thigh > angle_left_thigh+math.radians(60) and angle_right_thigh>0 and angle_left_thigh<0:
            self.left_dominant = False   

        self.current_fitness = walk_std*action_cost*reward_torso
        self.total_fitness += self.current_fitness
        return float(walk_std)

    def computed_time_alternate(self):
        """ Returns the final fitness of the agent based on the time spent on each leg going forward."""
        left_usage = self.left_time/self.time
        right_usage = 1 - left_usage
        alternate_legs = left_usage + right_usage - np.abs(left_usage - right_usage)
        alternate_legs = rewards.tolerance(alternate_legs,bounds=(0.85,1.0),margin=0.15,value_at_margin=0.1, sigmoid="gaussian")
        return self.current_fitness*alternate_legs
    
    def computed_time_alternate_fitness(self):
        """ Reward function that consider the time spent on each leg going forward."""
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
      
        if self.left_dominant:
            self.left_time += 1

        if angle_left_thigh > angle_right_thigh+math.radians(30):
            self.left_dominant = True
            
        
        if angle_right_thigh > angle_left_thigh+math.radians(30):
                self.left_dominant = False   
        
          
     
        self.total_fitness += walk_std*action_cost*reward_torso
    
    def compute_time_instant_fitness(self):
        """ Reward function that rewards the correct alternance of legs."""

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
                                value_at_margin=0.3,
                                sigmoid='linear')
        walk_std = stand_reward * (9*move_reward + 1) / 10

        action_cost = 0

        for i in range(6):
            action_cost += np.abs(self._last_action[i])
        action_cost = action_cost/6
        action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')


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
        """This reward function is made up of many components:
        - stand_reward: reward for standing upright. It is based on the torso height and the torso uprightness
        - move_reward: reward for moving at the desired speed which is 1 m/s
        - walk_std: a combination of stand_reward and move_reward
        - action_cost: cost for using the control
        """

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
        for i in range(6):
            action_cost += self._last_action[i]**2
        action_cost = 1 - action_cost/6
        # action_cost = rewards.tolerance(action_cost, bounds=(0.75, 1), margin=0.75, value_at_margin=0.5, sigmoid='linear')

        # for i in range(6):
        #         action_cost += self._last_action[i]**2
        # action_cost = action_cost/6
        # action_cost = rewards.tolerance(action_cost, bounds=(0.0, 0.2),value_at_margin=0.2, margin=0.50, sigmoid='gaussian')

        # for i in range(6):
        #     action_cost += np.abs(self._last_action[i])
        # action_cost = action_cost/6
        # action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')

        self.current_fitness = walk_std*action_cost 
        self.total_fitness += self.current_fitness


    def compute_standard_fitness_with_x_foot_constraint(self):
        """ Reward function used to make explicit the alternace leg term. This reward function is made up of many components:
        - stand_reward: reward for standing upright. It is based on the torso height and the torso uprightness
        - move_reward: reward for moving at the desired speed which is 1 m/s
        - walk_std: a combination of stand_reward and move_reward
        - alternate_legs: reward for alternating legs. It is based on the angle of the left and right thighs. The reward is 1 if they follow a sinusoidal pattern
        - action_cost: cost for using the control
        - reward_torso: reward for keeping the torso upright. This is redundant with stand_reward but it is kept for now
        """

        torso_height = self.observation[0]
        torso_velocity = self.observation[8]

        angle = self.observation[1]
        torso_upright = np.cos(angle)

        left_foot = self.get_x_left_foot()
        right_foot = self.get_x_right_foot()

        if torso_height > 1.5:
            self.current_fitness = 0
            self.total_fitness += self.current_fitness
            return

        # To avoid the robot to stay still
        if left_foot < (self.last_x_left_foot+0.001) and right_foot < (self.last_x_right_foot+0.001):
            self.current_fitness = 0
            self.total_fitness += self.current_fitness
            return
        
        if left_foot > self.last_x_left_foot:
            self.last_x_left_foot = left_foot
        if right_foot > self.last_x_right_foot:
            self.last_x_right_foot = right_foot   

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
        angle_torso = self.observation[1]
        
        #print("self._last_action", self._last_action)
        for i in range(6):
            action_cost += self._last_action[i]**2
        action_cost = 1 - action_cost/6
        #action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.30, sigmoid='hyperbolic')

        angle_right_thigh = self.observation[2]
        angle_left_thigh = self.observation[5]

        if self.left_dominant:
            if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_left_thigh > angle_right_thigh:
                self.left_dominant = False
                left_reward = 1
                right_reward = 1
            else:
                if angle_left_thigh > self.last_left_thigh_angle:
                    left_reward = 1
                    self.last_left_thigh_angle = angle_left_thigh
                else:
                    left_reward = 0
                if angle_right_thigh < self.last_right_thigh_angle:
                    self.last_right_thigh_angle = angle_right_thigh
                    right_reward = 1
                else:
                    right_reward = 0
        else:
            if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_right_thigh > angle_left_thigh:
                self.left_dominant = True   
                left_reward = 1
                right_reward = 1
            else:
                if angle_left_thigh < self.last_left_thigh_angle:
                    self.last_left_thigh_angle = angle_left_thigh
                    left_reward = 1
                else:
                    left_reward = 0
                if angle_right_thigh > self.last_right_thigh_angle:
                    self.last_right_thigh_angle = angle_right_thigh
                    right_reward = 1
                else:
                    right_reward = 0

        
        
        alternate_legs=((left_reward+right_reward) + left_reward*right_reward)/3
        #alternate_legs = (1 + 3*alternate_legs) / 4

        reward_torso = rewards.tolerance(angle_torso,
                                bounds=(-0.21, 0.21),
                                margin=0.05,
                                value_at_margin=0.0,
                                sigmoid='linear')
        reward_torso = (1 + 3*reward_torso) / 4
        
        #print("walk_std", round(walk_std, 5), "action_cost", round(action_cost, 5), "reward_torso", round(reward_torso, 5), "alternate_legs", round(alternate_legs, 5),"angle_left_thigh", round(angle_left_thigh, 5), "angle_right_thigh", round(angle_right_thigh, 5))
        self.current_fitness = (4*alternate_legs + 2*reward_torso*walk_std)/6 #+ action_cost)/7
        self.total_fitness += self.current_fitness





    def computed_walk_alternate_fitness_with_legs(self):
        """ Reward function that rewards the correct alternance of legs based on the angle of the thighs and legs and the gait"""

        
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
