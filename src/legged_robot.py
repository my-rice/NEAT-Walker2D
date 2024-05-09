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

    def __init__(self, observation_space_dim, action_space_dim) -> None:
        # We need to initialize the neural network with the given observation space and action space
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim        
    
    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        # Compute the action based on the network 
        self.action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self.action

    def update_agent(self, fitness: float) -> None:
        # Update the agent based on the fitness
        raise NotImplementedError

    def __str__(self):
        return "Observation space dim: " + str(self.observation_space_dim) + "\nAction space dim: " + str(self.action_space_dim) + "\nAgent: LeggedRobot"