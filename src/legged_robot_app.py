import random, sys, os, pygame
import numpy as np
from src.legged_robot import LeggedRobot
from pygame.locals import *
import gymnasium as gym
import time
import multiprocessing
from src.env import Environment, AvailableEnvironments

class LeggedRobotApp(object):

    def __init__(self, genomes, config, env_name="Walker2d-v4", agent_name="LeggedRobot", render=False):
        self.experiments = []
        self.start=0
        self.score = 0
        self.crash_info = []
        self.robots = []
        for index, genome in enumerate(genomes):
            env=None
            if(index==len(genomes)-1):
                if(render):
                    env=Environment(env_name, "human")
                else:
                    env=Environment(env_name, "rgb_array")
            else:
                env=Environment(env_name, "rgb_array")
            env.reset()
            legged_robot = LeggedRobot(env.observation_space.shape[0],env.action_space.shape[0],genome,config)
            self.experiments.append([env,legged_robot,0.0])

    def set_fitness(self, fitness, index):
        self.experiments[index][2] = fitness

    def on_loop(self):
        
        for count, experiment in enumerate(self.experiments):
            env = experiment[0]
            agent = experiment[1]
            fitness = experiment[2]
            # if(self.start==0):
            #     env[1]=env[1][0]
            action = agent.compute_action(env.get_current_observation())
            _, _, done , _=env.step(action)
            fitness = fitness + env.fitness()
            self.set_fitness(fitness, count)
            if done:
                self.crash_info.append((agent.get_genome(), fitness))
                del self.experiments[count]
                #print("The environment", env.get_index(), "has been removed")
                env.close()
                if(len(self.experiments)==0):
                    return True
            
        self.start=self.start+1
                        

    
    def play(self):    
        self.start=0
        while True:
            if self.on_loop():
                return
            
    



