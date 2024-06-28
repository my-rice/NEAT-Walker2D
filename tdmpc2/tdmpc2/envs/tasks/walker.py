import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import walker
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np
from pyquaternion import Quaternion
import csv
import math
_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_YOGA_STAND_HEIGHT = 1.0
_YOGA_LIE_DOWN_HEIGHT = 0.08
_YOGA_LEGS_UP_HEIGHT = 1.1


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'walker.xml')), common.ASSETS


@walker.SUITE.add('custom')
def walk_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def run_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def arabesque(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Arabesque task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='arabesque', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def lie_down(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Lie Down task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='lie_down', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def legs_up(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Legs Up task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='legs_up', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def headstand(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Headstand task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def flip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def backflip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Backflip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=-walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)

@walker.SUITE.add('custom')
def walk_custom(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = CustomPlanarWalker(move_speed=walker._WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


class CustomPlanarWalker(walker.PlanarWalker):
    """Custom PlanarWalker task."""
    def __init__(self, move_speed, random=None):
        super().__init__(move_speed, random)
        print("CustomPlanarWalker initialized with move_speed:",self._move_speed)
        self.left_dominant = True
        self.last_left_thigh_angle = 0
        self.last_right_thigh_angle = 0
        self.left_time = 0
        self.right_time = 0

        self.time = 0
        self.posbefore = 0


    def get_reward(self,physics):
        """ new_fitness_gym_based"""

        if physics.time() < 0.03:
            #print("resetting")
            self.right_time = 0.0
            self.left_time = 0.0
            self.last_left_thigh_angle = 0.0
            self.last_right_thigh_angle = 0.0
            self.left_dominant = True

        
        
        torso_height = physics.torso_height()

        # ONLY FOR TD-MPC2
        if torso_height < 1.0:
            return 0.0


        torso_velocity = physics.horizontal_velocity()
        self.time += 1.0
        standing = rewards.tolerance(torso_height,
                                    bounds=(walker._STAND_HEIGHT, float('inf')),
                                    margin=0.3,
                                    value_at_margin=0.0,
                                    sigmoid='linear'
                                    )
        
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4
        
        if self._move_speed == 0:
            return stand_reward
        move_reward = rewards.tolerance(torso_velocity,
                                bounds=(self._move_speed-0.1,self._move_speed+0.1),
                                margin=self._move_speed/2,
                                value_at_margin=0.1,
                                sigmoid='linear')
        walk_std = stand_reward * (5*move_reward + 1) / 6
        if(torso_velocity<=0):
            walk_std = 0

        quat_left_thigh = physics.named.data.xquat['left_thigh']
        quat_right_thigh = physics.named.data.xquat['right_thigh']
        quat_left_thigh = Quaternion(quat_left_thigh)
        quat_right_thigh = Quaternion(quat_right_thigh)
        angle_left_thigh = quat_left_thigh.angle
        angle_right_thigh = quat_right_thigh.angle

        quat_torso = physics.named.data.xquat['torso']
        quat_torso = Quaternion(quat_torso)
        
        if(torso_velocity > 0.8):
            if self.left_dominant:
                if np.abs(angle_left_thigh-angle_right_thigh) > (45*np.pi/180) and angle_left_thigh > angle_right_thigh:
                    self.left_dominant = False
                    self.left_time += 1
                    self.right_time += 1
                else:
                    if angle_left_thigh > self.last_left_thigh_angle:
                        self.left_time += 1
                    else:
                        self.left_time += 0
                    if angle_right_thigh < self.last_right_thigh_angle:
                        self.right_time += 1
                    else:
                        self.right_time += 0
            else:
                if np.abs(angle_left_thigh-angle_right_thigh) > (45*np.pi/180) and angle_right_thigh > angle_left_thigh:
                    self.left_dominant = True  
                    self.left_time += 1
                    self.right_time += 1
                else:
                    if angle_left_thigh < self.last_left_thigh_angle:
                        self.left_time += 1
                    else:
                        self.left_time += 0
                    if angle_right_thigh > self.last_right_thigh_angle:
                        self.right_time += 1
                    else:
                        self.right_time += 0
        
        #posafter, height, ang = self.sim.data.qpos[0:3]

        #reward = (posafter - self.posbefore) / 0.025
        action_cost = 0
        for i in range(6):
            action_cost += physics.control()[i]**2
        action_cost = 1-action_cost/6

        #reward -= 1e-3 * action_cost

        #self.posbefore = self.sim.data.qpos[0]
    
        alternate_legs = (self.left_time+self.right_time)-np.abs(self.left_time-self.right_time)
        alternate_legs = rewards.tolerance(alternate_legs,bounds=(1.5,2),margin=0.5,value_at_margin=0.01, sigmoid="hyperbolic")
        #alternate_legs+=1
        #alternate_legs = alternate_legs**6
    
        return (3*walk_std + action_cost + 3*alternate_legs) / 7


    
    

    # def get_reward(self, physics):
    #     """ WORKING. Reward function used fot "alternance_left_dominant" experiment. This reward function is made up of many components:
    #     - stand_reward: reward for standing upright. It is based on the torso height and the torso uprightness
    #     - move_reward: reward for moving at the desired speed which is 1 m/s
    #     - walk_std: a combination of stand_reward and move_reward
    #     - alternate_legs: reward for alternating legs. It is based on the angle of the left and right thighs. The reward is 1 if they follow a sinusoidal pattern
    #     - action_cost: cost for using the control
    #     - reward_torso: reward for keeping the torso upright. This is redundant with stand_reward but it is kept for now
        
    #     """
    #     #print("Using custom reward function")

    #     if physics.time() < 0.03:
    #         #print("physics.time()",physics.time())
    #         self.left_dominant = True
    #         self.last_left_thigh_angle = 0
    #         self.last_right_thigh_angle = 0
    #         self.left_time = 0
    #         self.right_time = 0
    #         self.last_x_left_foot = 0
    #         self.last_x_right_foot = 0
        
    #     torso_height = physics.torso_height()
    #     if torso_height > 1.5 or torso_height < 0.8:
    #         return 0

    #     standing = rewards.tolerance(physics.torso_height(),
    #                                 bounds=(walker._STAND_HEIGHT, float('inf')),
    #                                 margin=walker._STAND_HEIGHT/4)
    #     upright = (1 + physics.torso_upright()) / 2
    #     stand_reward = (3*standing + upright) / 4

    #     move_reward = rewards.tolerance(physics.horizontal_velocity(),
    #                             bounds=(self._move_speed-0.1, self._move_speed+0.1),
    #                             margin=self._move_speed/2,
    #                             value_at_margin=0.5,
    #                             sigmoid='linear')
    #     walk_std = stand_reward * (5*move_reward + 1) / 6
    #     # alternate legs reward

    #     action_cost = 0

    #     for i in range(6):
    #         action_cost += physics.control()[i]**2
    #     action_cost = action_cost/6
    #     action_cost = 1 - action_cost

    #     quat_left_thigh = physics.named.data.xquat['left_thigh']
    #     quat_right_thigh = physics.named.data.xquat['right_thigh']
    #     quat_left_thigh = Quaternion(quat_left_thigh)
    #     quat_right_thigh = Quaternion(quat_right_thigh)
    #     angle_left_thigh = quat_left_thigh.angle
    #     angle_right_thigh = quat_right_thigh.angle

    #     quat_torso = physics.named.data.xquat['torso']
    #     quat_torso = Quaternion(quat_torso)
    #     angle_torso = quat_torso.angle

    #     if self.left_dominant:
    #         if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_left_thigh > angle_right_thigh:
    #             self.left_dominant = False
    #             left_reward = 1
    #             right_reward = 1
    #         else:
    #             if angle_left_thigh > self.last_left_thigh_angle:
    #                 left_reward = 1
    #                 self.last_left_thigh_angle = angle_left_thigh
    #             else:
    #                 left_reward = 0
    #             if angle_right_thigh < self.last_right_thigh_angle:
    #                 self.last_right_thigh_angle = angle_right_thigh
    #                 right_reward = 1
    #             else:
    #                 right_reward = 0
    #     else:
    #         if np.abs(angle_left_thigh-angle_right_thigh) > (30*np.pi/180) and angle_right_thigh > angle_left_thigh:
    #             self.left_dominant = True   
    #             left_reward = 1
    #             right_reward = 1
    #         else:
    #             if angle_left_thigh < self.last_left_thigh_angle:
    #                 self.last_left_thigh_angle = angle_left_thigh
    #                 left_reward = 1
    #             else:
    #                 left_reward = 0
    #             if angle_right_thigh > self.last_right_thigh_angle:
    #                 self.last_right_thigh_angle = angle_right_thigh
    #                 right_reward = 1
    #             else:
    #                 right_reward = 0

    #     alternate_legs=((left_reward+right_reward) + left_reward*right_reward)/3

    #     alternate_legs = (1 + 3*alternate_legs) / 4


    #     reward_torso = rewards.tolerance(angle_torso,
    #                             bounds=(-0.21, 0.21),
    #                             margin=0.05,
    #                             value_at_margin=0.0,
    #                             sigmoid='linear')
    #     reward_torso = (1 + 3*reward_torso) / 4
    #     #print("walk_std:", round(walk_std, 5), "alternate_legs:", round(alternate_legs, 5), "angle_left_thigh:", round(angle_left_thigh, 5), "angle_right_thigh:", round(angle_right_thigh, 5))
    #     return (3*alternate_legs + 3*reward_torso*walk_std + action_cost)/7
        
    # def get_reward(self, physics):
    #     """ This function defines the reward function for the task. It is made up of many components:
    #     - stand_reward: reward for standing upright. It is based on the torso height and the torso uprightness
    #     - move_reward: reward for moving at the desired speed which is 1 m/s
    #     - walk_std: a combination of stand_reward and move_reward
    #     - alternate_legs: reward for alternating legs. It is based on the angle of the left and right thighs. The reward is 1 if they follow a sinusoidal pattern 
    #     - action_cost: cost for using the control
    #     - reward_torso: reward for keeping the torso upright. This is redundant with stand_reward but it is kept for now

        
    #     """
    #     #print("Using custom reward function")
    #     #print("physics.time()",physics.time())
    #     if physics.time() < 0.03:
    #         #print("resetting")
    #         self.last_x_left_foot = 0
    #         self.last_x_right_foot = 0
        
    #     torso_height = physics.torso_height()
    #     # if torso_height > 1.5:
    #     #     return 0

    #     # left_foot = physics.named.data.xpos['left_foot','x']
    #     # right_foot = physics.named.data.xpos['right_foot','x']

    #     # if left_foot < (self.last_x_left_foot+0.001) and right_foot < (self.last_x_right_foot+0.001):
    #     #     return 0        
    #     #ONLY FOR TD-MPC2
    #     if torso_height < 1.0:
    #         return 0
        
    #     # if left_foot > self.last_x_left_foot+0.001:
    #     #     self.last_x_left_foot = left_foot
    #     # if right_foot > self.last_x_right_foot+0.001:
    #     #     self.last_x_right_foot = right_foot 


    #     standing = rewards.tolerance(torso_height,
    #                                 bounds=(walker._STAND_HEIGHT, float('inf')),
    #                                 margin=walker._STAND_HEIGHT/4)
    #     upright = (1 + physics.torso_upright()) / 2
    #     stand_reward = (3*standing + upright) / 4

    #     move_reward = rewards.tolerance(physics.horizontal_velocity(),
    #                             bounds=(self._move_speed-0.1, self._move_speed+0.1),
    #                             margin=self._move_speed/2,
    #                             value_at_margin=0.5,
    #                             sigmoid='linear')
    #     walk_std = stand_reward * (5*move_reward + 1) / 6
    #     # alternate legs reward

    #     action_cost = 0

    #     for i in range(6):
    #         action_cost += physics.control()[i]**2
    #     action_cost = action_cost/6
    #     action_cost = 1 - action_cost


    #     #print("physics.named.data.xmat",physics.named.data.xpos)
        
    #     #print("physics.named.data.qvel",physics.named.data.qvel)
    #     #print("physics.named.data.cvel",physics.named.data.cvel)
    #     #print("dir physics.named.data.xmat",dir(physics.named.data))
        

    #     #print("physics.named.data.qvel[3:8]: right_ankle left_ankle:",physics.named.data.qvel[3:9])


    #     quat_left_thigh = physics.named.data.xquat['left_thigh']
    #     quat_right_thigh = physics.named.data.xquat['right_thigh']
    #     quat_left_thigh = Quaternion(quat_left_thigh)
    #     quat_right_thigh = Quaternion(quat_right_thigh)
    #     angle_left_thigh = quat_left_thigh.angle
    #     angle_right_thigh = quat_right_thigh.angle

    #     quat_torso = physics.named.data.xquat['torso']
    #     quat_torso = Quaternion(quat_torso)
    #     angle_torso = quat_torso.angle

    #     #print("angle_left_thigh:",angle_left_thigh,"angle_right_thigh:",angle_right_thigh,"angle_torso:",angle_torso)
        
        
    #     # the distance between the feet must follow a sinusoidal pattern

    #     step_frequency = self._move_speed / 0.5 # 0.5 is the step length
    #     angular_frequency = 2*np.pi*step_frequency
    #     amplitude = 0.34907 # 20 degrees

    #     ideal_left_foot = amplitude*np.sin(angular_frequency*physics.time()) + 0.174533
    #     ideal_right_foot = amplitude*np.sin(angular_frequency*physics.time() + np.pi) + 0.174533
    #     #print("ideal_left_foot:",ideal_left_foot,"ideal_right_foot:",ideal_right_foot)
    #     left_reward = rewards.tolerance(angle_left_thigh,
    #                                     bounds=(ideal_left_foot-0.05, ideal_left_foot+0.05),
    #                                     margin=0.05,
    #                                     value_at_margin=0.1,
    #                                     sigmoid='gaussian')
    #     right_reward = rewards.tolerance(angle_right_thigh,
    #                                     bounds=(ideal_right_foot-0.05, ideal_right_foot+0.05),
    #                                     margin=0.05,
    #                                     value_at_margin=0.1,
    #                                     sigmoid='gaussian')
        
    #     alternate_legs = (left_reward + right_reward) / 2
                                
    #     #print("angle_left_leg:",angle_left_leg,"angle_right_leg:",angle_right_leg, "l-r:",angle_left_leg-angle_right_leg)
    #     #print("left_reward:",left_reward,"right_reward:",right_reward,"current_left:",current_left,"current_right:",current_right,"vel_left_foot:",self.vel_left_foot,"vel_right_foot:",self.vel_right_foot,"angle_left_leg:",angle_left_leg,"angle_right_leg:",angle_right_leg)
        
        
    #     #alternate_legs=((left_reward+right_reward) + left_reward*right_reward)/3

    #     alternate_legs = (1 + 3*alternate_legs) / 4


    #     reward_torso = rewards.tolerance(angle_torso,
    #                             bounds=(-0.21, 0.21),
    #                             margin=0.05,
    #                             value_at_margin=0.0,
    #                             sigmoid='linear')
    #     reward_torso = (1 + 3*reward_torso) / 4
    #     #print("walk:", round(walk_std, 5), "alternate_legs:", round(alternate_legs, 5), "left_r:", round(left_reward, 5), "right_r:", round(right_reward, 5), "ideal_left:", round(ideal_left_foot, 5), "ideal_right:", round(ideal_right_foot, 5))
    #     return (3*walk_std + action_cost + reward_torso + 2*alternate_legs)/7

    # def get_reward(self, physics):
    #     #print("Using custom reward function")

    #     if physics.time() < 0.03:
    #         #print("physics.time()",physics.time())
    #         self.left_dominant = True
    #         self.last_left_thigh_angle = 0
    #         self.last_right_thigh_angle = 0
    #         self.left_time = 0
    #         self.right_time = 0
    #         self.last_x_left_foot = 0
    #         self.last_x_right_foot = 0


        
    #     torso_height = physics.torso_height()
    #     if torso_height > 1.5:
    #         return 0

    #     left_foot = physics.named.data.xpos['left_foot','x']
    #     right_foot = physics.named.data.xpos['right_foot','x']

    #     if left_foot < (self.last_x_left_foot+0.001) and right_foot < (self.last_x_right_foot+0.001):
    #         return 0        
    #     #ONLY FOR TD-MPC2
    #     if torso_height < 1.0:
    #         return 0
        
    #     if left_foot > self.last_x_left_foot+0.001:
    #         self.last_x_left_foot = left_foot
    #     if right_foot > self.last_x_right_foot+0.001:
    #         self.last_x_right_foot = right_foot 


    #     standing = rewards.tolerance(torso_height,
    #                                 bounds=(walker._STAND_HEIGHT, float('inf')),
    #                                 margin=walker._STAND_HEIGHT/4)
    #     upright = (1 + physics.torso_upright()) / 2
    #     stand_reward = (3*standing + upright) / 4

    #     move_reward = rewards.tolerance(physics.horizontal_velocity(),
    #                             bounds=(self._move_speed-0.1, self._move_speed+0.1),
    #                             margin=self._move_speed/2,
    #                             value_at_margin=0.5,
    #                             sigmoid='linear')
    #     walk_std = stand_reward * (5*move_reward + 1) / 6
    #     # alternate legs reward

    #     action_cost = 0

    #     for i in range(6):
    #         action_cost += physics.control()[i]**2
    #     action_cost = action_cost/6
    #     action_cost = 1 - action_cost


    #     #print("physics.named.data.xmat",physics.named.data.xpos)
        
    #     #print("physics.named.data.qvel",physics.named.data.qvel)
    #     #print("physics.named.data.cvel",physics.named.data.cvel)
    #     #print("dir physics.named.data.xmat",dir(physics.named.data))
        

    #     #print("physics.named.data.qvel[3:8]: right_ankle left_ankle:",physics.named.data.qvel[3:9])


    #     quat_left_thigh = physics.named.data.xquat['left_thigh']
    #     quat_right_thigh = physics.named.data.xquat['right_thigh']
    #     quat_left_thigh = Quaternion(quat_left_thigh)
    #     quat_right_thigh = Quaternion(quat_right_thigh)
    #     angle_left_thigh = quat_left_thigh.angle
    #     angle_right_thigh = quat_right_thigh.angle

    #     quat_torso = physics.named.data.xquat['torso']
    #     quat_torso = Quaternion(quat_torso)
    #     angle_torso = quat_torso.angle

    #     print("angle_left_thigh:",angle_left_thigh,"angle_right_thigh:",angle_right_thigh,"angle_torso:",angle_torso)
        
    #     if self.left_dominant:
    #         if angle_left_thigh > (25*np.pi/180) and angle_right_thigh < (-15*np.pi/180):
    #             self.left_dominant = False
    #             left_reward = 1
    #             right_reward = 1
    #         else:
    #             if angle_left_thigh > self.last_left_thigh_angle:
    #                 self.last_left_thigh_angle = angle_left_thigh
    #                 left_reward = 1
    #             else:
    #                 left_reward = 0
    #             if angle_right_thigh < self.last_right_thigh_angle:
    #                 self.last_right_thigh_angle = angle_right_thigh
    #                 right_reward = 1
    #             else:
    #                 right_reward = 0
    #     else:
    #         if angle_right_thigh > (25*np.pi/180) and angle_left_thigh < (-15*np.pi/180):
    #             self.left_dominant = True   
    #             left_reward = 1
    #             right_reward = 1
    #         else:
    #             if angle_left_thigh < self.last_left_thigh_angle:
    #                 self.last_left_thigh_angle = angle_left_thigh
    #                 left_reward = 1
    #             else:
    #                 left_reward = 0
    #             if angle_right_thigh > self.last_right_thigh_angle:
    #                 self.last_right_thigh_angle = angle_right_thigh
    #                 right_reward = 1
    #             else:
    #                 right_reward = 0

    #     #print("angle_left_leg:",angle_left_leg,"angle_right_leg:",angle_right_leg, "l-r:",angle_left_leg-angle_right_leg)
    #     #print("left_reward:",left_reward,"right_reward:",right_reward,"current_left:",current_left,"current_right:",current_right,"vel_left_foot:",self.vel_left_foot,"vel_right_foot:",self.vel_right_foot,"angle_left_leg:",angle_left_leg,"angle_right_leg:",angle_right_leg)
        
        
    #     alternate_legs=((left_reward+right_reward) + left_reward*right_reward)/3

    #     alternate_legs = (1 + 3*alternate_legs) / 4


    #     reward_torso = rewards.tolerance(angle_torso,
    #                             bounds=(-0.21, 0.21),
    #                             margin=0.05,
    #                             value_at_margin=0.0,
    #                             sigmoid='linear')
    #     reward_torso = (1 + 3*reward_torso) / 4
    #     #print("walk_std:",walk_std,"alternate_legs:",alternate_legs,"move_reward:",move_reward,"stand_reward:",stand_reward,"upright:",upright)
    #     return alternate_legs*walk_std*action_cost*reward_torso



    # def get_reward(self, physics):
    #     #print("Using custom reward function")

    #     if physics.time() < 0.03:
    #         self.left_dominant = True
    #         self.last_left_thigh_angle = 0
    #         self.last_right_thigh_angle = 0
    #         self.left_time = 0
    #         self.right_time = 0


    #     standing = rewards.tolerance(physics.torso_height(),
    #                                 bounds=(walker._STAND_HEIGHT, float('inf')),
    #                                 margin=0.3,
    #                                 value_at_margin=0.0,
    #                                 sigmoid='linear'
    #                                 )
    #     upright = (1 + physics.torso_upright()) / 2
    #     stand_reward = (3*standing + upright) / 4
    #     if self._move_speed == 0:
    #         return stand_reward
    #     move_reward = rewards.tolerance(physics.horizontal_velocity(),
    #                             bounds=(self._move_speed-0.1, self._move_speed+0.1),
    #                             margin=self._move_speed/2,
    #                             value_at_margin=0.1,
    #                             sigmoid='linear')
    #     walk_std = stand_reward * (5*move_reward + 1) / 6
    #     # alternate legs reward

    #     action_cost = 0

    #     for i in range(6):
    #         action_cost += physics.control()[i]**2
    #     action_cost = action_cost/6
    #     action_cost = rewards.tolerance(action_cost, bounds=(0.225, 0.425),value_at_margin=0.2, margin=0.225, sigmoid='hyperbolic')

    #     quat_left_thigh = physics.named.data.xquat['left_thigh']
    #     quat_right_thigh = physics.named.data.xquat['right_thigh']
    #     quat_left_thigh = Quaternion(quat_left_thigh)
    #     quat_right_thigh = Quaternion(quat_right_thigh)
    #     angle_left_thigh = quat_left_thigh.angle
    #     angle_right_thigh = quat_right_thigh.angle
        
    #     quat_left_leg = physics.named.data.xquat['left_leg']
    #     quat_right_leg = physics.named.data.xquat['right_leg']
    #     quat_left_leg = Quaternion(quat_left_leg)
    #     quat_right_leg = Quaternion(quat_right_leg)
    #     angle_left_leg = quat_left_leg.angle
    #     angle_right_leg = quat_right_leg.angle

    #     #print("angle_left_leg:",angle_left_leg,"angle_right_leg:",angle_right_leg)

    #     quat_torso = physics.named.data.xquat['torso']
    #     quat_torso = Quaternion(quat_torso)
    #     angle_torso = quat_torso.angle

    #     reward_torso = rewards.tolerance(angle_torso,
    #                             bounds=(-0.21, 0.21),
    #                             margin=0.05,
    #                             value_at_margin=0.0,
    #                             sigmoid='linear')
    #     reward_torso = (1 + 3*reward_torso) / 4
    #     difference = np.abs(angle_left_thigh-angle_right_thigh)
    #     weight = rewards.tolerance(difference,bounds=(0.0,0.8),margin=0.5,value_at_margin=0.1,sigmoid='linear')
    #     difference = weight*difference
    #     zero_neighbor_up = 0.12
    #     zero_neighbor_down = -0.12
    #     torso_velocity = physics.horizontal_velocity()
    #     if self.left_dominant:
    #         if(torso_velocity*0.002>0):
    #             self.left_time += torso_velocity*0.002*difference
    #     else:
    #         if(torso_velocity*0.002>0):
    #             self.right_time += torso_velocity*0.002*difference
    #     if angle_left_thigh > angle_right_thigh+math.radians(45) and angle_left_leg > zero_neighbor_down and angle_left_leg < zero_neighbor_up:
    #         self.left_dominant = True
            
           
    #     if angle_right_thigh > angle_left_thigh+math.radians(45) and angle_right_leg > zero_neighbor_down and angle_right_leg < zero_neighbor_up:
    #             self.left_dominant = False 
        

        
    #     if(self.left_time+self.right_time==0):
    #         return 0
    #     alternate_legs = (self.left_time+self.right_time-np.abs(self.left_time-self.right_time))/(self.left_time+self.right_time)

        
    #     return alternate_legs*walk_std*action_cost*reward_torso
        





    # def get_reward(self, physics):
        # #print("Using custom reward function")
        # standing = rewards.tolerance(physics.torso_height(),
        #                             bounds=(walker._STAND_HEIGHT, float('inf')),
        #                             margin=walker._STAND_HEIGHT/4)
        # upright = (1 + physics.torso_upright()) / 2
        # stand_reward = (3*standing + upright) / 4
        # if self._move_speed == 0:
        #     return stand_reward
        # move_reward = rewards.tolerance(physics.horizontal_velocity(),
        #                         bounds=(self._move_speed, self._move_speed),
        #                         margin=self._move_speed/2,
        #                         value_at_margin=0.5,
        #                         sigmoid='linear')
        # walk_std = stand_reward * (5*move_reward + 1) / 6
        # # alternate legs reward

        # action_cost = 0

        # for i in range(6):
        #     action_cost += physics.control()[i]**2
        # action_cost = action_cost/6
        # action_cost = rewards.tolerance(action_cost, bounds=(0.0, 0.2),value_at_margin=0.0, margin=0.30, sigmoid='linear')

        # #print("physics.named.data.xmat",physics.named.data.xquat)
        # #print("dir physics.named.data.xmat",dir(physics.named.data))

        # quat_left_thigh = physics.named.data.xquat['left_leg']
        # quat_right_thigh = physics.named.data.xquat['right_leg']
        # quat_left_thigh = Quaternion(quat_left_thigh)
        # quat_right_thigh = Quaternion(quat_right_thigh)
        # angle_left_leg = quat_left_thigh.angle
        # angle_right_leg = quat_right_thigh.angle

        # #print("angle_left_leg:",angle_left_leg,"angle_right_leg:",angle_right_leg)

        # if self.left_dominant:
        #     self.left_dominant = False
        #     if angle_left_leg >= angle_right_leg + (15/180*np.pi):
        #         alternate_legs = 1
        #     else:
        #         alternate_legs = 0
        # else:
        #     self.left_dominant = True
        #     if angle_left_leg <= angle_right_leg - (15/180*np.pi):
        #         alternate_legs = 1
        #     else:
        #         alternate_legs = 0
        # # left_leg = physics.named.data.xpos['left_leg', 'x']
        # # right_leg = physics.named.data.xpos['right_leg', 'x']

        # alternate_legs = (1 + 3*alternate_legs) / 4

        # #print("walk_std:",walk_std,"alternate_legs:",alternate_legs,"move_reward:",move_reward,"stand_reward:",stand_reward,"upright:",upright)
        # return alternate_legs*walk_std*action_cost


#     def get_reward(self,physics):
#         """ This function defines the reward function for the task. This is the easiest reward that accomplishes the task. 
#         It is made up of many components:
#         - stand_reward: reward for standing upright. It is based on the torso height and the torso uprightness
#         - move_reward: reward for moving at the desired speed which is 1 m/s
#         - walk_std: a combination of stand_reward and move_reward
#         - action_cost: cost for using the control

#         Note that the action cost is responsible for the agent to alternate legs, as it is the only way to minimize the cost.
#         """
#         #print("dir(physics):",dir(physics))
#         standing = rewards.tolerance(physics.torso_height(),
#                                      bounds=(walker._STAND_HEIGHT, float('inf')),
#                                      margin=walker._STAND_HEIGHT/4)
#         upright = (1 + physics.torso_upright()) / 2
#         stand_reward = (3*standing + upright) / 4
#         if self._move_speed == 0:
#             return stand_reward
#         else:
#             move_reward = rewards.tolerance(physics.horizontal_velocity(),
#                                 bounds=(self._move_speed-0.1, self._move_speed+0.1),
#                                 margin=self._move_speed/2,
#                                 value_at_margin=0.5,
#                                 sigmoid='linear')
#             walk_std = stand_reward * (5*move_reward + 1) / 6

#             action_cost = 0
#             # print("physics.control:",physics.control)
#             # print("dir(physics.comtrol):",dir(physics.control))
#             # print("physics.control():",physics.control())
#             for i in range(6):
#                 action_cost += physics.control()[i]**2
#             action_cost = action_cost/6
#             action_cost = rewards.tolerance(action_cost, bounds=(0.0, 0.2),value_at_margin=0.0, margin=0.30, sigmoid='linear')

#         #print("action_cost:",action_cost,"mean:",sum(np.abs(physics.control()))/6)
#         return walk_std*action_cost
        
# class BackwardsPlanarWalker(walker.PlanarWalker):
#     """Backwards PlanarWalker task."""
#     def __init__(self, move_speed, random=None):
#         super().__init__(move_speed, random)
    
#     def get_reward(self, physics):
#         standing = rewards.tolerance(physics.torso_height(),
#                                  bounds=(walker._STAND_HEIGHT, float('inf')),
#                                  margin=walker._STAND_HEIGHT/2)
#         upright = (1 + physics.torso_upright()) / 2
#         stand_reward = (3*standing + upright) / 4
#         if self._move_speed == 0:
#             return stand_reward
#         else:
#             move_reward = rewards.tolerance(physics.horizontal_velocity(),
#                                             bounds=(-float('inf'), -self._move_speed),
#                                             margin=self._move_speed/2,
#                                             value_at_margin=0.5,
#                                             sigmoid='linear')
#             return stand_reward * (5*move_reward + 1) / 6


class YogaPlanarWalker(walker.PlanarWalker):
    """Yoga PlanarWalker tasks."""
    
    def __init__(self, goal='arabesque', move_speed=0, random=None):
        super().__init__(0, random)
        self._goal = goal
        self._move_speed = move_speed
    
    def _arabesque_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        left_foot_down = rewards.tolerance(left_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_STAND_HEIGHT/2)
        right_foot_up = rewards.tolerance(right_foot_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        arabesque_reward = (3*standing + left_foot_down + right_foot_up + upright) / 6
        return arabesque_reward
    
    def _lie_down_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_down = rewards.tolerance(feet_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        lie_down_reward = (3*torso_down + thigh_down + upright) / 5
        return lie_down_reward
    
    def _legs_up_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        legs_up_reward = (3*torso_down + 2*legs_up + thigh_down + upright) / 7
        return legs_up_reward
    
    def _flip_reward(self, physics):
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_up = rewards.tolerance(thigh_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upside_down_reward = (3*legs_up + 2*thigh_up) / 5
        if self._move_speed == 0:
            return upside_down_reward
        move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                    bounds=(self._move_speed, float('inf')) if self._move_speed > 0 else (-float('inf'), self._move_speed),
                                    margin=abs(self._move_speed)/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
        return upside_down_reward * (5*move_reward + 1) / 6

    def get_reward(self, physics):
        if self._goal == 'arabesque':
            return self._arabesque_reward(physics)
        elif self._goal == 'lie_down':
            return self._lie_down_reward(physics)
        elif self._goal == 'legs_up':
            return self._legs_up_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')


if __name__ == '__main__':
    env = legs_up()
    obs = env.reset()
    import numpy as np
    next_obs, reward, done, info = env.step(np.zeros(6))
