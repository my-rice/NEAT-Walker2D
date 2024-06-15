import os
import datetime
import pandas as pd
from omegaconf import OmegaConf
import pickle


class Logger:
    
    def __init__(self, cfg,rank=0,comm=None):
        """Initialize the Logger class"""
        self.cfg = cfg
        self.columns = ["generation", "fitness", "migration_step"]
        self.exp_data = pd.DataFrame(columns=self.columns)
        self.results_dir = None

        if rank == 0:
            self._make_dir()
            with open(os.path.join(self.results_dir, "config.yaml"), "w") as f:
                OmegaConf.save(self.cfg, f)
            self.genome_path = os.path.join(os.getcwd(), self.cfg.neat_config)
            try:
                with open(self.genome_path, "r") as f:
                    genome = f.read()
            except FileNotFoundError:
                print("Genome file not found")
                os._exit(1)

            with open(os.path.join(self.results_dir, "config-neat.ini"), "w") as f:
                f.write(genome)
            
            # Broadcast the results_dir to all processes
            self.results_dir = comm.bcast(self.results_dir, root=0)
        else:
            self.results_dir = comm.bcast(self.results_dir, root=0)

    def _make_dir(self):
        """Create a directory to store the results of the experiment"""
 
        curr_dir = os.getcwd()
        now = datetime.datetime.now()
        # Format it as a string
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = os.path.join(curr_dir, "results/"+self.cfg.experiment_name+"_"+now_str)
        try:
            os.makedirs(self.results_dir)
        except FileExistsError:
            print("Directory already exists")
            os._exit(1)
        print("Results will be saved in ", self.results_dir)


    def log(self, rank=None, generation=None, fitness=None, migration_step=None):
        """Log the data for the experiment"""
        #print("Logging data for rank ", rank, " generation ", generation, " fitness ", fitness, " migration step ", migration_step)
        
        if rank is None:
            rank = 0 # The experiment is not parallelized. A default rank is enough to log data
        
        new_row = [generation, fitness, migration_step]
        
        self.exp_data.loc[len(self.exp_data)] = new_row
            
    def save_results(self,rank=0):
        """Save the results for the experiment for each rank. If the experiment is parallelized, please, provide the rank. Otherwise, the default rank is 0."""
        self.exp_data.to_csv(os.path.join(self.results_dir, f"results_{rank}.csv"), index=False)

    def save_net(self, net, rank):
        """Save the neural network"""
        print("Sto salvando le reti")
        path = os.path.join(self.results_dir, f"best_net_{rank}.pkl")
        #pickle.dump(net, open("winner"+str(rank)+".pkl", "wb"))
        pickle.dump(net, open(path, "wb"))

        
        
        
        