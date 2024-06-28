import numpy as np

from abc import ABC, abstractmethod

class Agent(ABC):
	""" Abstract class that defines the interface for an agent. It has a method to compute the action based on 
	the observation. This class should be inherited by the specific agent class which can add new methods. 
	For example, to update the agent after each step in the environment based on the reward."""
	def __init__(self, observation_space_dim: int, action_space_dim: int) -> None:
		self.observation_space_dim = observation_space_dim
		self.action_space_dim = action_space_dim

	@abstractmethod
	def compute_action(self, observation: np.ndarray) -> np.ndarray:
		raise NotImplementedError
	