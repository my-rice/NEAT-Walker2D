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
    genomes = pickle.load(open('/home/giovanni/Desktop/progettodiDavidino/legged_locomotion/results/fitnessconalternanza_2024-06-16_12-44-02/best_net_4.pkl', 'rb'))
    
    for i in range(0,n):
        # Play game and get results
        flappy_Bio = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5")
        flappy_Bio.play()

def main():
    if len(sys.argv)>1:
        run_winner(int(sys.argv[1]))
    else:
        run_all()
        #run_winner()

def run_all():
    # Load configuration.
    config = neat.Config(DefaultGenomeWrapper, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat.ini')
    
    #load Genome
    for i in range(0,10):
        print("Running winner ", i)
        genomes = pickle.load(open('results/nuovafitnessancora_2024-06-16_00-09-10/best_net_'+str(i)+'.pkl', 'rb'))
        
        # Play game and get results
        flappy_Bio = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5")
        flappy_Bio.play()

if __name__ == "__main__":
	main()