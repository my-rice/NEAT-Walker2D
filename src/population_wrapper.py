from neat.population import Population

from neat.six_util import iteritems, itervalues
import heapq
import time
import random
import pickle
class CompleteExtinctionException(Exception):
    pass

class PopulationWrapper(Population):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)
        self.population_ranking = []  
        # for g in itervalues(self.population):
        #     fitness = random.randint(0, 100)
        #     self.population_ranking.append((fitness, g.key))
        # heapq.heapify(self.population_ranking)
        self.rank = None
        self.last_printed = 0
    def get_population(self):
        return self.population # DA LEVARE
    
    def get_generation(self):
        return self.generation  
    
    def replace_n_noobs(self, genome_to_replace): # NOTE: we assume that population dict key is the genome id
     

        for g in itervalues(self.population):
            self.population_ranking.append((g.fitness, g.key))

        heapq.heapify(self.population_ranking)
        worst_genomes = []
        for i in range(0, len(genome_to_replace)):
            worst_genomes.append(heapq.heappop(self.population_ranking))

       
            
            
        for i in range(0, len(genome_to_replace)):
            #print("type", type(genome_to_replace[i]))
            genome_to_replace[i].key = worst_genomes[i][1]
            self.population[genome_to_replace[i].key] = genome_to_replace[i]
            #print("I am rank ", self.rank, "self.population[genome_to_replace[i].key]", dir(self.population[genome_to_replace[i].key].fitness))
            # print("type", type(self.population))
            # print("type 2: ", type(self.population[genome_to_replace[i].key]))
            # print("self.population[genome_to_replace[i].key].node_indexer",self.population[genome_to_replace[i].key].node_indexer)
            #print("self.config.genome_type.node_indexer",dir(self.config))
            # print("self.config.genome_config",self.config.genome_config.node_indexer)
            # os.exit(1)
            
            #self.config.genome_config = self.config.genome_type.parse_config(genome_dict)

            
            #self.population[genome_to_replace[i].key].node_indexer = None
        
        #print("self.config.genome_config.node_indexer",self.config.genome_config.node_indexer)
        #self.config.genome_config.node_indexer = None
        
        self.config.genome_config.reset_node_indexer()
        #self.config.genome_config.connection_indexer = None
        self.population_ranking = []
        
        #print(" self.species.species.items", self.species.species.members)
        for key,value in self.species.species.items():
            for f,k in worst_genomes:
                if k in value.members.keys():
                    self.species.species[key].members[k] = self.population[k]
                                
        self.species.speciate(self.config, self.population, self.generation) # Reproduction iterates on species, if we don't call this method, the new genomes will not be in any species

        # pollo = self.population
        # try:
        #     self.population = self.reproduction.reproduce(self.config, self.species,
        #                                                     self.config.pop_size, self.generation)
        # except:
        #     print("ciao, sono rank ", self.rank, " e ho fallito la riproduzione")
            
       
                
        
        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                            self.config.genome_config,
                                                            self.config.pop_size)
            else:
                raise CompleteExtinctionException()
        
        # # Divide the new population into species.
        # self.species.speciate(self.config, self.population, self.generation) 

        
        self.reporters.end_generation(self.config, self.population, self.species)
        

        self.generation += 1
        
        
    def run_mpi(self, fitness_function, n=None, rank=None, logger=None, migration_step=None, seed=None, feed_forward=True, exponent_legs=1.0):
        self.rank = rank
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")
            
        k=0
        while n is None or k < n:
            k += 1
            if(k%50==0):
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Rank:",rank)
                pickle.dump(self.best_genome, open("best_for_now"+str(rank)+".pkl", "wb"))
                # print("The best of rank:", rank, "is", self.best_genome.fitness)

            #print("I am rank ", rank, " and I am in generation ", self.generation)
            self.reporters.start_generation(self.generation)
            
            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config, seed, feed_forward=feed_forward, exponent_legs=exponent_legs)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            if logger is not None:
                logger.log(rank=rank, generation=k, fitness=best.fitness, migration_step=migration_step)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
            if(self.best_genome.fitness>self.last_printed):
                self.last_printed = self.best_genome.fitness
                print("The best of rank:", rank, "is", self.best_genome.fitness,"in generation", self.generation)  
            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break
            # Create the next generation from the current generation.
            specie_attuale = self.species.species


            if(k!=n):
                try:
                    self.population = self.reproduction.reproduce(self.config, self.species,
                                                            self.config.pop_size, self.generation)
                except Exception as e:
                    print("I am the rank crashed: ", rank, " k is ", k, " and the exception is: ")
                    for key,value in self.species.species.items():
                        print(rank,"Species key:" ,value.key)
                        for m in value.members:
                            print(rank,"Genome:", m.key, m.fitness)

                
                # Check for complete extinction.
                if not self.species.species:
                    self.reporters.complete_extinction()

                    # If requested by the user, create a completely new population,
                    # otherwise raise an exception.
                    if self.config.reset_on_extinction:
                        self.population = self.reproduction.create_new(self.config.genome_type,
                                                                    self.config.genome_config,
                                                                    self.config.pop_size)
                    else:
                        raise CompleteExtinctionException()
                
                # Divide the new population into species.
                self.species.speciate(self.config, self.population, self.generation)

                
                self.reporters.end_generation(self.config, self.population, self.species)
              

                self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)


        return self.best_genome