import numpy as np

class Agent:

	def __init__(self):
		raise NotImplementedError
	
	def compute_action(self, observation: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def update_agent(self, fitness: float) -> None:
		# Update the agent based on the fitness
		raise NotImplementedError