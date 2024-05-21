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
ENV_NAME = None
AGENT_NAME = None


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

def slave_loop(migration_steps,comm,n, neat_config,seed):
    # wait for master to start with the number of information (one of them is the number of bests to migrate)
    rank = comm.Get_rank()
    size = comm.Get_size()
    dims = [math.sqrt(size), math.sqrt(size)]
    periods = [True, True]  
    reorder = True 
    cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)
    coords = cart_comm.Get_coords(rank)
    


    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
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
        best = p.run_mpi(eval_genomes, n=n, rank=rank)
        if(rank==0):
            print("I am rank ", rank, " and I am in generation ", p.get_generation(), " and best fitness is ", best.fitness)
        
        if(count_migrations==migration_steps-1):
            try:
                pickle.dump(best, open("winner"+str(rank)+".pkl", "wb"))
                print("I am rank ", rank, " and I AM DONE, best fitness is ", best.fitness)
            except:
                print("I am rank ", rank, " and I AM BLOCKED size is ", size, " and count_migrations is ", count_migrations)
            break
        if(count_migrations==migration_steps-2):
            comm.bcast(best, root=rank)
            recv_data = []
            for i in range(1, size):
                if i != rank:
                    best_received=comm.bcast(None,root=i)
                    recv_data.append(best_received)
            p.replace_n_noobs(recv_data)
           
        else:
            neighbors = [north, south, west, east]
            for neighbor in neighbors:
                if neighbor != MPI.PROC_NULL:
                    comm.isend(best, dest=neighbor, tag=1)
            recv_data = []
            for neighbor in neighbors:
                if neighbor != MPI.PROC_NULL:
                    best_received = comm.recv(source=neighbor, tag=1)
                    recv_data.append(best_received)
            p.replace_n_noobs(recv_data)



        count_migrations+=1
        # sync receive from master what to do next
        # if stop, break
        # if continue, change population and continue loop


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
    assert cfg.migration_steps > 0
    assert cfg.seed is not None
    ENV_NAME = cfg.env_name
    AGENT_NAME = cfg.agent_name

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

  
    slave_loop(migration_steps=cfg.migration_steps,comm=comm,n=cfg.generations, neat_config=cfg.neat_config, seed=cfg.seed) # check if is a real config

if __name__ == "__main__":
    main()