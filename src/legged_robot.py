import numpy as np
from src.agent import Agent
from enum import Enum
import neat
from src.neural import NN
# class syntax
class AvailableAgents(Enum):
    legged_agent = "LeggedRobot"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class LeggedRobot(Agent):

    def __init__(self, observation_space_dim, action_space_dim, genomes: neat.DefaultGenome, config: neat.Config) -> None:
        # We need to initialize the neural network with the given observation space and action space
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim     
        
        self.genomes = genomes

        self.nets = []
        
        self.best_fitness = 0
        self.nns = []
        self.best_nn = None

        # We create a neural network for every given genome
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config) # Create a neural network for every genome in the population 
            self.nets.append(net)
            genome.fitness = 0
            self.nns.append(NN(config, genome, (60, 130)))

        self.best_nn = None
        self.best_input = None   
    
    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        # Compute the action based on the network 
        
        # compute a random action for now
        """Compute the next move of every car and update their fitness

        Args:
            genomes (neat.DefaultGenome): The neat genomes
            track (pygame.Surface): The track on which the car is being drawn
            width (int): The width of the window
        """
        i = 0
        for net in self.nets:
            
            legged_data = observation
            output = net.activate(legged_data)
            
            # Output gets treated and the car is updated in the next lines
            choice = output.index(max(output))
            
            # Refreshing nodes of all neural networks
            for node in self.nns[i].nodes:
                node.inputs = legged_data
                node.output = choice

            # Mettere le azioni possibile
                
            i += 1

        # Refresh cars sprites, number of cars which are still alive and update their fitness


        # self.genomes[i][1].fitness += car.get_reward()
        # if self.genomes[i][1].fitness > self.best_fitness:
        #     self.best_fitness = self.genomes[i][1].fitness
        #     self.best_nn = self.nns[i]


        self.action = np.random.uniform(-1, 1, self.action_space_dim)
        return self.action

    def update_agent(self, fitness: float) -> None:
        # Update the agent based on the fitness
        raise NotImplementedError

    def __str__(self):
        return "Observation space dim: " + str(self.observation_space_dim) + "\nAction space dim: " + str(self.action_space_dim) + "\nAgent: LeggedRobot"
    
