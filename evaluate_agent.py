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
    genomes = pickle.load(open('/home/giovanni/Desktop/progettodiDavidino/legged_locomotion/results/fitnessconalternanza30gradi_2024-06-16_23-51-10/best_net_7.pkl', 'rb'))
    
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
    for i in range(0,9):
        print("Running winners ", i)
        try:
            genomes = pickle.load(open('best_for_now'+str(i)+".pkl", 'rb'))
        
            # Play game and get results
            flappy_Bio = LeggedRobotApp([genomes], config, render=True, env_name="Walker2d-v5", feed_forward=False)
            flappy_Bio.play()
        except:
            pass

if __name__ == "__main__":
	main()