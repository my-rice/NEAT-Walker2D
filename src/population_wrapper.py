from neat.population import Population

from neat.six_util import iteritems, itervalues
import heapq
import time
import random
    
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
        self.worst_genome=None
        
    def get_population(self):
        return self.population # DA LEVARE
    
    def get_generation(self):
        return self.generation  
    
    def replace_n_noobs(self, genome_to_replace): # NOTE: we assume that population dict key is the genome id
        # change ID of the best genomes
        for g in itervalues(self.population):
            self.population_ranking.append((g.fitness, g.key))

        heapq.heapify(self.population_ranking)
        worst_genomes = []
        for i in range(0, len(genome_to_replace)):
            worst_genomes.append(heapq.heappop(self.population_ranking))
        for i in range(0, len(genome_to_replace)):
            genome_to_replace[i].key = worst_genomes[i][1]
            self.population[genome_to_replace[i].key] = genome_to_replace[i]
        self.population_ranking = []
        
    def run_mpi(self, fitness_function, n=None, rank=None):
        
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")
        
        k = 0
        while n is None or k < n:
            k += 1
            #print("I am rank ", rank, " and I am in generation ", self.generation)
            self.reporters.start_generation(self.generation)
            
            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best
            
            
                
            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break
            # Create the next generation from the current generation.
            if(k!=n):
                self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

          
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