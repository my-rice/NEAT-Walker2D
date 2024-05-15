import random, sys, os, pygame
import numpy as np
from src.legged_robot import LeggedRobot
from pygame.locals import *
import gymnasium as gym
import time
import multiprocessing
from src.env import Environment, AvailableEnvironments

class LeggedRobotApp(object):

    def __init__(self, genomes, config, env_name="Walker2d-v4", agent_name="LeggedRobot"):
        print(env_name)
        self.envs = []
        self.start=0
        self.score = 0
        self.crash_info = []
        self.robots = []
        for index, genome in enumerate(genomes):
            env=None
            if(index==len(genomes)-1):
                env=Environment(env_name, "human")
            else:
                env=Environment(env_name, "rgb_array")
            env.set_index(index)
            observation = env.reset()
            self.envs.append([env, observation])
            legged_robot = LeggedRobot(env.observation_space.shape[0],env.action_space.shape[0],genome,config)
            self.robots.append(legged_robot)


    def on_loop(self):
        
        for index, env in enumerate(self.envs):
            if(self.start==0):
                env[1]=env[1][0]
            action = self.robots[index].compute_action(env[1])
            env[1], _, done , _=env[0].step(action)

            if done:
                self.crash_info.append((self.robots[index]))
                del self.envs[index]
                print("The environment", env[0].get_index(), "has been removed")
                if(len(self.envs)==0):
                    print("All the environments are dead")
                    return True
            
        self.start=self.start+1
                        

    
    def play(self):    
        self.start=0
        while True:
            if self.on_loop():
                return
            
    



