from src.env import Environment, AvailableEnvironments
from src.legged_robot import LeggedRobot, AvailableAgents
import hydra
from omegaconf import MISSING, OmegaConf
import neat
import pickle
from src.legged_robot_app import LeggedRobotApp
import time
ENV_NAME = None
AGENT_NAME = None
START_TIME = None
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
    print(time.time()*1000-START_TIME)
    plot_winner(file_winner_net='winner.pkl', config=config)

def plot_winner(file_winner_net, config):
    with open(file_winner_net, 'rb') as f:
        winner = pickle.load(f)
    print('\nBest genome:\n{!s}'.format(winner))

def eval_genomes(genomes, config):
    genomes_dict = {}
    for tuple in genomes:
        key = tuple[0]
        genome = tuple[1]
        genomes_dict[key] = genome
    # Play game and get results
    _, genomes = zip(*genomes)

    start_time = time.time()
    legged_Bio = LeggedRobotApp(genomes, config, ENV_NAME, AGENT_NAME)
    
    results = legged_Bio.crash_info
    top_score = 0
    for key, fitness in results.items():
        genomes_dict[key].fitness = fitness
        if fitness > top_score:
            top_score = fitness


    

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg):
    global ENV_NAME, AGENT_NAME, START_TIME
    START_TIME = time.time()*1000
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