import neat.genome
from src.env import Environment, AvailableEnvironments
from src.legged_robot import LeggedRobot, AvailableAgents
import hydra
from omegaconf import MISSING, OmegaConf
import neat
from src.population_wrapper import PopulationWrapper
import pickle
from src.legged_robot_app import LeggedRobotApp
from mpi4py import MPI
import time
import numpy as np
import math
from logger import Logger
import random
import json
import yaml
from genome_wrapper import DefaultGenomeWrapper
ENV_NAME = None
AGENT_NAME = None


def plot_winner(file_winner_net, config):
    with open(file_winner_net, 'rb') as f:
        winner = pickle.load(f)
    print('\nBest genome:\n{!s}'.format(winner))

def eval_genomes(genomes, config, seed):
    
    # Play game and get results
    _,genomes = zip(*genomes)
    legged_Bio = LeggedRobotApp(genomes, config, ENV_NAME, AGENT_NAME, seed=seed)
    legged_Bio.play()
    results = legged_Bio.crash_info
    
    # Calculate fitness and top score
    top_score = 0
    for result in results:
        genome = result[0]
        fitness = result[1]
        genome.fitness = fitness
        if fitness > top_score:
            top_score = fitness
    for idx,genome in enumerate(genomes):
        if genome.fitness == None:
            print(idx)  
        

    # print score
    #print('The top score was:', top_score)

# def master_loop(migration_steps, bests_to_migrate, comm, size):


#     comm.bcast(bests_to_migrate, root=0)# send to slaves number of bests to migrate

#     # wait for slaves to finish and receive bests
#     for i in range(1, size):
#         data = comm.recv(source=i, tag=1)
#         if(i==1):
#             print(f"Master received data from slave {i}: {data}")
#     # migration convention
#     # send to slaves migrated bests or stop them
#     # at then end of the entire loop, save the best and kill the remaining slaves
#     pass

def slave_loop(migration_steps,comm,n, neat_config,seed,logger):
    # wait for master to start with the number of information (one of them is the number of bests to migrate)
    rank = comm.Get_rank()
    size = comm.Get_size()
    seed=seed+rank
    np.random.seed(seed)
    random.seed(seed)
    dims = [math.sqrt(size), math.sqrt(size)]
    periods = [True, True]  
    reorder = True 
    cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)
    coords = cart_comm.Get_coords(rank)
    


    config = neat.Config(DefaultGenomeWrapper, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config)

    # Create the population, which is the top-level object for a NEAT run.
    p = PopulationWrapper(config)
    # Add a stdout reporter to show progress in the terminal.
    # p.add_reporter(neat.StdOutReporter(True))
    # Run until we achive n.
    count_migrations = 0
    while count_migrations < migration_steps:
        north, south = cart_comm.Shift(0, 1)
        west, east = cart_comm.Shift(1, 1)
        best = p.run_mpi(eval_genomes, n=n, rank=rank,logger=logger, migration_step=count_migrations, seed=seed)
        pickle.dump(best, open("actual"+str(rank)+".pkl", "wb"))
        print("I am rank ", rank, " and I am in generation ", p.get_generation(), " and best fitness is ", best.fitness)
        
        if(count_migrations==migration_steps-1):
            try:
                pickle.dump(best, open("winner"+str(rank)+".pkl", "wb"))
                print("I am rank ", rank, " and I AM DONE, best fitness is ", best.fitness)
            except:
                print("I am rank ", rank, " and I AM BLOCKED size is ", size, " and count_migrations is ", count_migrations)
            break
        if(count_migrations==migration_steps-2):
            comm.bcast(rank, root=rank)
            recv_data = []
            for i in range (size):
                if i != rank:
                    rank_received=comm.bcast(None,root=i)
                    recv_data.append(rank_received)
            recv_genomes = []
            for neighbor in recv_data:
                genomes = pickle.load(open('actual'+str(neighbor)+'.pkl', 'rb')) # is rb so there isn't problem with mutual exclusion of files
                recv_genomes.append(genomes)
            p.replace_n_noobs(recv_genomes)
           
        else:
            neighbors = [north, south, west, east]
            for neighbor in neighbors:
                if neighbor != MPI.PROC_NULL:
                    comm.isend(rank, dest=neighbor, tag=1)
            recv_data = []
            for neighbor in neighbors:
                if neighbor != MPI.PROC_NULL:
                    rank_received = comm.recv(source=neighbor, tag=1)
                    recv_data.append(rank_received)
            recv_genomes = []
            for neighbor in recv_data:
                genomes = pickle.load(open('actual'+str(neighbor)+'.pkl', 'rb'))
                recv_genomes.append(genomes)

            p.replace_n_noobs(recv_genomes)



        count_migrations+=1
        # sync receive from master what to do next
        # if stop, break
        # if continue, change population and continue loop


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if cfg.multiple_experiments:      
        # run the experiment
        num_experiments = 0
        with open("multiple_exp.json") as f:
            multiple_exp = json.load(f)
        exp_names = multiple_exp["experiments"]["exp_names"]
        num_experiments = len(exp_names)

        for i in range(0, num_experiments):
            comm.Barrier()
            if rank == 0:
                configure_experiments_file(cfg, i)
            comm.Barrier()

            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            
            for key, value in config.items():
                cfg[key] = value

            run_experiment(cfg,rank,comm,size)
    else:
        run_experiment(cfg,rank,comm,size)


def configure_experiments_file(cfg,iteration=0):
    
    # load multiple_exp.json
    with open("multiple_exp.json") as f:
        multiple_exp = json.load(f)
    
    #print("multiple_exp", multiple_exp)
    exp_names = multiple_exp["experiments"]["exp_names"]
    yaml_configs = multiple_exp["experiments"]["config_yaml"]
    neat_configs = multiple_exp["experiments"]["config_neat"]

    print("exp_names", exp_names)
    # print("yaml_configs", yaml_configs)
    # print("neat_configs", neat_configs)
    
    exp_name = exp_names[iteration] 
    i = iteration
    print("Running experiment", i+1,"/",len(exp_names), "with name", exp_name)

    # change the experiment name in the cfg object
    cfg.experiment_name = exp_name

    # load the config.yaml file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # update parameters in the config.yaml file
    
    for key, value in yaml_configs.items():
        print("config[key]", config[key], "value[i]", value[i], "key", key, "value", value)
        config[key] = value[i]
        cfg[key] = value[i]
        print("cfg[key]", cfg[key], "value[i]", value[i], "key", key, "value", value)
    
    config["experiment_name"] = exp_name

    # sort the config.yaml file
    config = dict(sorted(config.items()))
    #print("config UPDATED", config)

    # store the updated config.yaml file
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    # load the config_neat.ini file
    with open("config-neat.ini") as f:
        config_neat = f.read()
    
    for key, value in neat_configs.items():
        # find a line that starts with key
        lines = config_neat.split("\n")
        for j,line in enumerate(lines):
            if line.startswith(key):
                lines[j] = key + " = " + str(value[i])
                break
        config_neat = "\n".join(lines)

    #print("config_neat UPDATED", config_neat)

    # store the updated config_neat.ini file
    with open("config-neat.ini", "w") as f:
        f.write(config_neat)

def run_experiment(cfg,rank=0,comm=None,size=1):
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
    assert cfg.migration_steps > 0
    assert cfg.seed is not None
    ENV_NAME = cfg.env_name
    AGENT_NAME = cfg.agent_name


    logger = Logger(cfg,rank=rank,comm=comm)
    slave_loop(migration_steps=cfg.migration_steps,comm=comm,n=cfg.generations, neat_config=cfg.neat_config, seed=cfg.seed,logger=logger) # check if is a real config
    
    logger.save_results(rank=rank)
    
if __name__ == "__main__":
    main()