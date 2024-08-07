import numpy as np
from src.agent import Agent
from enum import Enum
import neat
# class syntax
class AvailableAgents(Enum):
    legged_agent = "LeggedRobot"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class LeggedRobot(Agent):
    """ Class that inherits from Agent and implements the methods to compute the action and update the agent."""
    def __init__(self, observation_space_dim, action_space_dim, genome, config, feed_forward) -> None: # teoricamente qua dovrebbe cambiare in base all'agente che passiamo
        # We need to initialize the neural network with the given observation space and action space
        if(feed_forward == True):
            self.neural_network = neat.nn.FeedForwardNetwork.create(genome,config)
        else:
            self.neural_network = neat.nn.RecurrentNetwork.create(genome,config)
        self.observation_space_dim = observation_space_dim
        self.genome = genome
        self.config = config
        self.action_space_dim = action_space_dim     

    def get_genome(self):
        return self.genome

    def compute_action(self, observation):
        # Compute the action based on the network 
        input = observation
        output = self.neural_network.activate(input)
        # compute a random action for now
        return output

    def __str__(self):
        return "Observation space dim: " + str(self.observation_space_dim) + "\nAction space dim: " + str(self.action_space_dim) + "\nAgent: LeggedRobot"