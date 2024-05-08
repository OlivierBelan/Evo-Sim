from evo_simulator.ALGORITHMS.Algorithm import Algorithm
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.GENERAL.Reproduction import Reproduction_NN
from evo_simulator.GENERAL.Mutation_NN import Mutation
from evo_simulator.GENERAL.Genome import Genome_NN
import numpy as np


from typing import Dict, Any
import random
import time


class GA(Algorithm):
    def __init__(self, config_path_file:str, name:str = "GA") -> None:
        Algorithm.__init__(self, config_path_file, name)
        # Initialize configs
        self.config_ga:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["GA", "Genome_NN", "Reproduction"])
        self.verbose:bool = True if self.config_ga["GA"]["verbose"] == "True" else False

        self.pop_size:int = int(self.config_ga["GA"]["pop_size"])
        self.is_first_generation:bool = True
        self.reproduction:Reproduction_NN = Reproduction_NN(config_path_file, self.attributes_manager)
        self.mutation:Mutation = Mutation(config_path_file, self.attributes_manager)
        self.is_sbx:bool = True if self.config_ga["Reproduction"]["is_sbx"] == "True" else False
        self.auto_update_sigma:bool = True if self.config_ga["GA"]["auto_update_sigma"] == "True" else False
        self.best_fitness:float = None
        self.sigma_toggle:bool = True

    def run(self, global_population:Population) -> Population:
        self.population_manager = global_population

        if self.is_first_generation == True: 
            self.is_first_generation = False
            self.first_generation(self.population_manager)
            return self.population_manager
        self.ajust_population(self.population_manager)

        self.__sigma_update(self.population_manager)
        
        # 1 - Reproduction
        self.__reproduction(self.population_manager)

        # 2 - Mutation
        self.__mutation(self.population_manager)

        # 4 - Update population
        global_population.population = self.population_manager.population

        return global_population


    def first_generation(self, population_manager:Population) -> None:
        start_time = time.time()
        self.ajust_population(population_manager)
        print(self.name+": First generation time:", time.time() - start_time, "s")

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_ga["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=True)
            population[new_genome.id] = new_genome


    def __reproduction(self, population:Population) -> None:
        # 1- Reproduction
        population.population = self.reproduction.reproduction(population, self.pop_size)


    def __mutation(self, population:Population) -> None:
        if self.is_sbx == True: return population
        # 1 - Mutation (attributes only)
        pop_dict:Dict[int, Genome_NN] = population.population
        # 1.1 - Mutation Neuron (attributes only)
        population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in pop_dict.items() if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_neuron_params}
        self.mutation.attributes.neurons_sigma(population_to_mutate, self.attributes_manager.parameters_neuron_names)
        # 1.2 - Mutation Synapse (attributes only)
        population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in pop_dict.items() if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_synapse_params}
        self.mutation.attributes.synapses_sigma(population_to_mutate, self.attributes_manager.parameters_synapse_names)


    def __sigma_update(self, population:Population):
        if self.auto_update_sigma == False: return

        # 1 - Check current best fitness
        best_genome_fitness:float = population.best_genome.fitness.score
        if self.best_fitness is None:
            self.best_fitness = best_genome_fitness        
        if abs(best_genome_fitness) > abs(self.best_fitness): # no stagnation
            self.best_fitness = best_genome_fitness
        elif abs(best_genome_fitness) <= abs(self.best_fitness): # stagnation (update sigma)
            
            # 2 - check if sigma is at min or max in order to know if we need to increase or decrease it
            weight_sigma:np.ndarray = self.attributes_manager.sigma_parameters["weight"]
            if weight_sigma <= self.attributes_manager.sigma_min_parameters["weight"] + 0.0001: # in case min is 0
                self.sigma_toggle = False
            elif weight_sigma >= self.attributes_manager.sigma_max_parameters["weight"]:
                self.sigma_toggle = True
            
            # 3 - update sigma
            if self.sigma_toggle == True:
                weight_sigma *= self.attributes_manager.sigma_decay_parameters["weight"]
            elif self.sigma_toggle == False:
                weight_sigma += weight_sigma * (1 - self.attributes_manager.sigma_decay_parameters["weight"])
            increase_deacrease:str = "descreasing" if self.sigma_toggle == True else "increasing"
            print("GA sigma:(", increase_deacrease,")", self.attributes_manager.sigma_parameters["weight"], "decay:", self.attributes_manager.sigma_decay_parameters["weight"], "min:", self.attributes_manager.sigma_min_parameters["weight"], "max:", self.attributes_manager.sigma_max_parameters["weight"])
            return
        print("GA sigma:( stable )", self.attributes_manager.sigma_parameters["weight"], "decay:", self.attributes_manager.sigma_decay_parameters["weight"], "min:", self.attributes_manager.sigma_min_parameters["weight"], "max:", self.attributes_manager.sigma_max_parameters["weight"])
