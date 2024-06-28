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
    genomes = pickle.load(open('results/new4_2024-06-26_20-16-46/best_net_3.pkl', 'rb'))
    
    for i in range(0,n):
        # Play game and get results
        flappy_Bio = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5", feed_forward=False)
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
    for i in range(7,8):
        print("Running winnersss ", i)
        try:
            genomes = pickle.load(open('results/new9_2024-06-27_20-58-15/best_net_'+str(i)+".pkl", 'rb'))
        
            # Play game and get results
            flappy_Bio = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5", feed_forward=False,exponent_legs=6)
            flappy_Bio.play()
        except:
            pass

if __name__ == "__main__":
	main()  