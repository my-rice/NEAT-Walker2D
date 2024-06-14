import neat
import pickle
import sys
from  src.legged_robot_app import LeggedRobotApp
from genome_wrapper import DefaultGenomeWrapper

def run_winner(n=1):
    # Load configuration.
    config = neat.Config(DefaultGenomeWrapper, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat.ini')
    
    #load Genome
    genomes = pickle.load(open('results/fortewinner4.pkl', 'rb'))
    
    for i in range(0,n):
        # Play game and get results
        flappy_Bio = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5")
        flappy_Bio.play()

def main():
    if len(sys.argv)>1:
        run_winner(int(sys.argv[1]))
    else:
        run_winner()

if __name__ == "__main__":
	main()