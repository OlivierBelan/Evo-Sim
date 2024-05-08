from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_Decoder
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.GENERAL.Distance import Distance
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES_algorithm import CMA_ES_algorithm
from typing import Dict, Any, List
import numpy as np
import time


class Decoder_CMA(Algorithm):
    def __init__(self, config_path_file:str, name:str = "Decoder_CMA") -> None:
        Algorithm.__init__(self, config_path_file, name)
        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Decoder", "Genome_Decoder"])
        self.verbose:bool = True if self.config_es["Decoder"]["verbose"] == "True" else False
    
        self.pop_size:int = int(self.config_es["Decoder"]["pop_size"])
        self.elite_size:int = np.floor(float(self.config_es["Decoder"]["elites_ratio"]) * self.pop_size).astype(int)
        self.mu:float = float(self.config_es["Decoder"]["mu"])
        self.sigma:float = float(self.config_es["Decoder"]["sigma"])
        self.max_param:float = float(self.config_es["Decoder"]["max_param"])
        self.min_param:float = float(self.config_es["Decoder"]["min_param"])
        self.mu_max:float = float(self.config_es["Decoder"]["mu_max"])
        self.mu_min:float = float(self.config_es["Decoder"]["mu_min"])
        self.sigma_max:float = float(self.config_es["Decoder"]["sigma_max"])
        self.sigma_min:float = float(self.config_es["Decoder"]["sigma_min"])

        self.is_first_generation:bool = True
        self.distance:Distance = Distance(config_path_file)
        
        
        # Initialize es_algorithms
        self.init_es_algorithms()

    def init_es_algorithms(self):

        self.neuron_decoder_cma_es:CMA_ES_algorithm = CMA_ES_algorithm(
                                                                population_size=self.pop_size, 
                                                                parameters_size=5, # neuron_parameters_size,
                                                                elite_size=self.elite_size,
                                                                mean=self.mu,
                                                                sigma=self.sigma,
                                                                mean_max=self.mu_max,
                                                                mean_min=self.mu_min,
                                                                sigma_max=self.sigma_max,
                                                                sigma_min=self.sigma_min,
                                                                is_clipped=True,
                                                                )

        self.synapse_decoder_cma_es:CMA_ES_algorithm = CMA_ES_algorithm(
                                                                population_size=self.pop_size, 
                                                                parameters_size=2, # synapse_parameters_size,
                                                                elite_size=self.elite_size,
                                                                mean=self.mu,
                                                                sigma=self.sigma,
                                                                mean_max=self.mu_max,
                                                                mean_min=self.mu_min,
                                                                sigma_max=self.sigma_max,
                                                                sigma_min=self.sigma_min,
                                                                is_clipped=True,
                                                                )


    def run(self, global_population:Population) -> Population:

        self.population_manager.population = global_population.population
        self.first_generation(self.population_manager)
        self.ajust_population(self.population_manager)
        if self.is_first_generation == True:
            self.is_first_generation = False
            global_population.population = self.population_manager.population
            return global_population

        # 0 - Update CMA-ES
        start_time = time.time()
        self.__update_cma_es_by_fitness(self.population_manager)
        print(self.name+": Update CMA-ES time:", time.time() - start_time, "s")

        # 1 - Update population parameters
        start_time = time.time()
        self.__update_population_parameter(self.population_manager)
        print(self.name+": Update population parameters time:", time.time() - start_time, "s")

        # 3 - Update population
        global_population.population = self.population_manager.population

        return global_population

            
    def first_generation(self, population_manager:Population) -> None:
        if  population_manager.is_first_generation == True:
            start_time = time.time()
            self.ajust_population(population_manager)
            population_manager.is_first_generation = False
            self.__update_population_parameter(population_manager)
            print(self.name+": First generation time:", time.time() - start_time, "s")

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_Decoder] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_Decoder = Genome_Decoder(get_new_genome_id(), self.config_es["Genome_Decoder"])
            new_genome.info["is_elite"] = False
            population[new_genome.id] = new_genome

    def __update_population_parameter(self, population_manager:Population) -> None:
        # 1 - Get parameters from CMA-ES algorithms
        genomes_dict:Dict[int, Genome_Decoder] = population_manager.population
        self.neuron_parameters = self.neuron_decoder_cma_es.get_parameters()
        self.synapse_parameters = self.synapse_decoder_cma_es.get_parameters()

        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            
            # 2.1 Update neuron parameters
            genome.decoder_neuron = self.neuron_parameters[index].astype(np.float32)

            # 2.2 Update synapse parameters
            genome.decoder_synapse = self.synapse_parameters[index].astype(np.float32) # negative delay is not allowed (will be set to 0 by the runner)
            if genome.decoder_synapse[1] < 0:
                genome.decoder_synapse[1] = 0.0

    def __update_cma_es_by_fitness(self, population_manager:Population) -> None:
        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_Decoder] = population_manager.population
        fitnesses:List[int] = []
        for genome in genomes_dict.values():
            fitnesses.append(genome.fitness.score)
        fitnesses:np.ndarray = np.array(fitnesses)
        elites_indexes:np.ndarray = fitnesses.argsort()[::-1]

        # print("elites_indexes", elites_indexes, "size", elites_indexes.size)
        # print("fitnesses", fitnesses, "size", fitnesses.size)
        # print("fitness_max:", fitnesses[elites_indexes[0]], "fitness_min:", fitnesses[elites_indexes[-1]])

        # Update CMA-ES
        self.neuron_decoder_cma_es.update(elites_indexes, fitnesses)
        self.synapse_decoder_cma_es.update(elites_indexes, fitnesses)
