import numpy as np
from src.legged_robot import LeggedRobot
import gymnasium as gym
from src.env import Environment, AvailableEnvironments

class LeggedRobotApp(object):
    """ Main class that instantiates the environment and the agent and runs the simulation."""
    def __init__(self, genomes, config, env_name="Walker2d-v4", agent_name="LeggedRobot", render=False, seed=None, feed_forward=True, exponent_legs=1.0):
        self.experiments = [] # list of experiments, each experiment is a list [env, agent, fitness]
        self.crash_info = [] # list of tuples (genome, fitness)
        for index, genome in enumerate(genomes):
            env=None
            if(index==len(genomes)-1):
                if(render):
                    env=Environment(env_name, "human", seed=seed, exponent_legs=exponent_legs)
                else:
                    env=Environment(env_name, "rgb_array", seed=seed, exponent_legs=exponent_legs)
            else:
                env=Environment(env_name, "rgb_array", seed=seed, exponent_legs=exponent_legs)
            env.reset()
            legged_robot = LeggedRobot(env.observation_space.shape[0],env.action_space.shape[0],genome,config, feed_forward=feed_forward)
            self.experiments.append([env,legged_robot,0.0])

    def set_fitness(self, fitness, index):
        self.experiments[index][2] = fitness

    def on_loop(self):
        """ Main loop that runs the simulation."""
        for count, experiment in enumerate(self.experiments):
            env = experiment[0]
            agent = experiment[1]
            action = agent.compute_action(env.get_current_observation())
            _, reward, done , _=env.step(action)
            env.fitness()
            if done:
                total_fitness = env.get_total_fitness()
                #print("total fitness", total_fitness)
                self.crash_info.append((agent.get_genome(), total_fitness))
                del self.experiments[count]
                env.close()
                if(len(self.experiments)==0):
                    return True
            
                        

    
    def play(self):    
        while True:
            if self.on_loop():
                return
            
    



