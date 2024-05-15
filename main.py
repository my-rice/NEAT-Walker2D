from src.env import Environment, AvailableEnvironments
from src.legged_robot import LeggedRobot, AvailableAgents
import hydra
from omegaconf import MISSING, OmegaConf
import neat
import pickle
from src.legged_robot_app import LeggedRobotApp

ENV_NAME = None
AGENT_NAME = None

def evolutionary_driver(n, neat_config):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))

    # Run until we achive n.
    winner = p.run(eval_genomes, n=n) 

    # dump
    pickle.dump(winner, open('winner.pkl', 'wb'))

    plot_winner(file_winner_net='winner.pkl', config=config)

def plot_winner(file_winner_net, config):
    with open(file_winner_net, 'rb') as f:
        winner = pickle.load(f)
    print('\nBest genome:\n{!s}'.format(winner))

def eval_genomes(genomes, config):
    
    # Play game and get results
    _,genomes = zip(*genomes)
    
    legged_Bio = LeggedRobotApp(genomes, config, ENV_NAME, AGENT_NAME)
    legged_Bio.play()
    results = legged_Bio.crash_info
    
    # Calculate fitness and top score
    top_score = 0
    for result, genomes in results:

        score = 3
        distance = 2
        energy = 1

        fitness = score*3000 + 0.2*distance - energy*1.5
        genomes.fitness = -1 if fitness == 0 else fitness
        

    # print score
    print('The top score was:', top_score)

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg):
    global ENV_NAME, AGENT_NAME
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")
    assert cfg.max_episodes > 0
    assert cfg.generations > 0
    assert cfg.render is True or cfg.render is False
    assert AvailableEnvironments.has_value(cfg.env_name)
    assert AvailableAgents.has_value(cfg.agent_name)
    assert cfg.neat_config is not None
    ENV_NAME = cfg.env_name
    AGENT_NAME = cfg.agent_name
    evolutionary_driver( n=cfg.generations, neat_config=cfg.neat_config) # check if is a real config

if __name__ == "__main__":
    main()