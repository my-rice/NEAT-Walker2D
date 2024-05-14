import numpy as np
from src.agent import Agent
from enum import Enum

# class syntax
class AvailableAgents(Enum):
    legged_agent = "LeggedRobot"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class LeggedRobot(Agent):

    def __init__(self, observation_space_dim, action_space_dim, genome, config) -> None: # teoricamente qua dovrebbe cambiare in base all'agente che passiamo
        # We need to initialize the neural network with the given observation space and action space
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim     
        print("Observation space dim: ", self.observation_space_dim) 
        print("Action space dim: ", self.action_space_dim)   
    
    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        # Compute the action based on the network 
        
        # compute a random action for now
        self.action = np.random.uniform(-1, 1, self.action_space_dim)
        return self.action

    def update_agent(self, fitness: float) -> None:
        # Update the agent based on the fitness
        raise NotImplementedError

    def __str__(self):
        return "Observation space dim: " + str(self.observation_space_dim) + "\nAction space dim: " + str(self.action_space_dim) + "\nAgent: LeggedRobot"