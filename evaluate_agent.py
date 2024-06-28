import neat
import pickle
import sys
import os
from  src.legged_robot_app import LeggedRobotApp
from genome_wrapper import DefaultGenomeWrapper

def run_winner(path):
    # Load configuration.
    print("Running winner from path: ", path)
    config_path = os.path.join(os.path.dirname(path), 'config-neat.ini')
    config = neat.Config(DefaultGenomeWrapper, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
                         
    #load Genome
    genomes = pickle.load(open(path, 'rb'))
    
    walker_app = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5", feed_forward=False)
    walker_app.play()


if __name__ == "__main__":

    # Take the path of the winner from the command line
    if len(sys.argv)<2:
        print("Usage: python evaluate_agent.py <path_to_winner>")
        sys.exit(1)
    
    path = sys.argv[1]

    # if the path is a directory, run all the winners in the directory
    if not os.path.exists(path):
        print("The path does not exist")
        sys.exit(1)
    
    # chdir to the last directory

    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".pkl"):
                run_winner(os.path.join(path, file))
    else:
        if path.endswith(".pkl"):
            run_winner(path)
