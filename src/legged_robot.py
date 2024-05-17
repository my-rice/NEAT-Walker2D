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

    def __init__(self, observation_space_dim, action_space_dim, genome, config) -> None: # teoricamente qua dovrebbe cambiare in base all'agente che passiamo
        # We need to initialize the neural network with the given observation space and action space
        self.genome = genome
        self.neural_network = neat.nn.FeedForwardNetwork.create(genome,config)
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim     

    def get_genome(self):
        return self.genome

    def compute_action(self, observation):
        # Compute the action based on the network 
        input = observation
        output = self.neural_network.activate(input)
        # compute a random action for now
        #output = np.random.rand(self.action_space_dim)
        return output

    def update_agent(self, fitness: float) -> None:
        # Update the agent based on the fitness
        raise NotImplementedError

    def __str__(self):
        return "Observation space dim: " + str(self.observation_space_dim) + "\nAction space dim: " + str(self.action_space_dim) + "\nAgent: LeggedRobot"