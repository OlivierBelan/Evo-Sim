from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
import evo_simulator.TOOLS as TOOLS
from evo_simulator.ALGORITHMS.NEAT.Specie import Specie_Manager
from evo_simulator.ALGORITHMS.NEAT.Mutation import Mutation_NEAT
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Reproduction import Reproduction_NN
from typing import Dict, Any, List
import numpy as np
import time

class NEAT(Algorithm):
    def __init__(self, config_path_file:str, name:str = "NEAT", extra_info:Dict[Any, Any] = None) -> None:
        Algorithm.__init__(self, config_path_file, name, extra_info)

        # Initialize configs
        self.config_neat:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NEAT", "Genome_NN", "Specie"])
        self.reproduction:Reproduction_NN = Reproduction_NN(config_path_file, self.attributes_manager)
        self.mutation:Mutation_NEAT = Mutation_NEAT(config_path_file, self.attributes_manager)
        self.verbose:bool = True if self.config_neat["NEAT"]["verbose"] == "True" else False

        # NEAT
        self.__config_neat(self.config_neat["NEAT"])
        self.specie:Specie_Manager = Specie_Manager(config_path_file)
        self.is_first_generation:bool = True
        self.neurons_status_population:np.ndarray = np.array([], dtype=np.bool8)
        self.nb_neurons:int = 0
        self.best_fitness:float = None
        self.sigma_toggle:bool = True
    
    def __config_neat(self, config:Dict[int, Genome_NN]) -> None:
        self.pop_size:int = int(config["pop_size"])
        self.auto_update_sigma:bool = True if config["auto_update_sigma"] == "True" else False
    

    def run(self, global_population:Population) -> Population:        
        self.population_manager.population = global_population.population
        # self.first_generation(self.population_manager, evaluation_function)
        if self.is_first_generation == True: 
            self.first_generation(self.population_manager)
            return self.population_manager

        if len(self.population_manager.population) != self.pop_size:
            raise Exception("Population size is not correct: " + str(len(self.population_manager.population)) + " instead of " + str(self.pop_size))
            self.ajust_population(self.population_manager)

        # 0 - Syncronize population with species
        self.syncronize_population_with_species(self.population_manager)

        # 1 - Print stats
        self.__print_stats()

        # 2 - Update sigma
        self.__sigma_update(self.population_manager)
        
        # 3 - Speciation/Selection
        self.specie.speciation(self.population_manager)

        # 4 - Reproduction
        self.__reproduction(self.population_manager)

        # 5 - Mutation
        self.__mutation_neat(self.population_manager)

        if len(self.population_manager.population) != self.pop_size: raise Exception("Population size is not correct: " + str(len(self.population_manager.population)) + " instead of " + str(self.pop_size))

        # 6 - Update population
        global_population.population = self.population_manager.population

        # 7 - Free memory        
        self.free_memory_species()
        return global_population

    def __sigma_update(self, population:Population):
        if self.auto_update_sigma == False: return
        best_genome_fitness:float = population.best_genome.fitness.score
        if self.best_fitness is None:
            self.best_fitness = best_genome_fitness
        if abs(best_genome_fitness) > abs(self.best_fitness):
            self.best_fitness = best_genome_fitness
        elif abs(best_genome_fitness) <= abs(self.best_fitness):
            weight_sigma:np.ndarray = self.attributes_manager.sigma_parameters["weight"]
            if weight_sigma <= self.attributes_manager.sigma_min_parameters["weight"] + 0.0001: # in case min is 0
                self.sigma_toggle = False
            elif weight_sigma >= self.attributes_manager.sigma_max_parameters["weight"]:
                self.sigma_toggle = True

            if self.sigma_toggle == True:
                weight_sigma *= self.attributes_manager.sigma_decay_parameters["weight"]
            elif self.sigma_toggle == False:
                weight_sigma += weight_sigma * (1 - self.attributes_manager.sigma_decay_parameters["weight"])
            increase_deacrease:str = "descreasing" if self.sigma_toggle == True else "increasing"
            print("NEAT sigma:(", increase_deacrease,")", self.attributes_manager.sigma_parameters["weight"], "decay:", self.attributes_manager.sigma_decay_parameters["weight"], "min:", self.attributes_manager.sigma_min_parameters["weight"], "max:", self.attributes_manager.sigma_max_parameters["weight"])
            return
        print("NEAT sigma:( stable )", self.attributes_manager.sigma_parameters["weight"], "decay:", self.attributes_manager.sigma_decay_parameters["weight"], "min:", self.attributes_manager.sigma_min_parameters["weight"], "max:", self.attributes_manager.sigma_max_parameters["weight"])

    def first_generation(self, population_manager:Population) -> None:
        start_time = time.time()
        self.is_first_generation = False
        population:Dict[int, Genome_NN] = population_manager.population
        self.ajust_population(population_manager)
        first_key:int = next(iter(population))
        self.nb_neurons:int = population[first_key].nn.nb_neurons
        population_manager:Population = self.__mutation_neat(population_manager)
        print(self.name+": First generation time:", time.time() - start_time, "s")
    
    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_neat["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=True)
            population[new_genome.id] = new_genome


    def __reproduction(self, population:Population) -> None:
        # 1- Clear population
        new_population:Dict[int, Genome_NN] = {}
        population.population.clear()

        # 2- Reproduction of each specie
        for specie in self.specie.species.values():
            # 2.1 - Update new population
            new_population.update(self.reproduction.reproduction(specie, specie.reproduction_size, replace=True))

        # 3- Update population
        population.population = new_population
    
    def __mutation_neat(self, population:Population) -> Population:
        population:Population = self.mutation.mutation_neat(population)
        return population


    def __get_info_stats_species(self):
        stats:List[List[int, float, float, int]] = []
        for specie in self.specie.species.values():
            specie.update_info()
            if specie.best_genome is None: continue
            best_fitness:float = specie.fitness.score
            mean_fitness:float = specie.fitness.mean
            stagnation:float = specie.stagnation
            best_genome:Genome_NN = specie.best_genome
            nb_neurons:int = len(best_genome.nn.hiddens["neurons_indexes_active"])
            nb_synapses:int = best_genome.nn.synapses_actives_indexes[0].size
            stats.append([specie.id, len(specie.population), (best_genome.id, round(best_fitness, 3), nb_neurons, nb_synapses), round(mean_fitness, 3), stagnation])
        return stats

    def __get_info_distance(self):
        mean_distance:float = self.specie.mean_distance
        distance_threshold_ratio:float = self.specie.distance_threshold_ratio
        mean_distance_threshold_ratio:float = (mean_distance + (mean_distance * distance_threshold_ratio))
        distance_threshold_min:float = self.specie.distance_threshold_min
        distance_used:float = self.specie.distance_threshold_used

        print("Mean_distance:", round(mean_distance, 3), 
        ", Mean_distance * distance_threshold_ratio ("+ str(distance_threshold_ratio)+"):", round(mean_distance_threshold_ratio,3),
        ", Min distance:", distance_threshold_min, 
        ", Current distance used:", distance_used)
        print("--------------------------------------------------------------------------------------------------------------------->>> " +self.name)

    def __print_stats(self):
        if len(self.specie.species) == 0 or self.verbose == False: return
        self.__get_info_distance()
        titles = [["Species", "Size", "Best(id, fit, neur, syn)", "Avg", "Stagnation"]]
        titles.extend(self.__get_info_stats_species())
        col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for row in titles:
            print("".join(str(word).ljust(col_width) for word in row))
        print("\n")

    def syncronize_population_with_species(self,population:Population) -> None:
        for specie in self.specie.species.values():
            for genome_id in specie.population.keys():
                if genome_id in population.population:
                    specie.population[genome_id] = population.population[genome_id]

    def free_memory_species(self) -> None:
        for specie in self.specie.species.values():
            for genome_id in specie.population.keys():
                specie.population[genome_id] = None
