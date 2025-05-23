from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN, Genome_Classic
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Reproduction import Reproduction_NN, Reproduction_Classic
from evo_simulator.ALGORITHMS.NEAT.Mutation import Mutation_NEAT
from GENERAL.Mutation_NN import Mutation
from evo_simulator.GENERAL.Archive import Archive
from typing import Dict, Any, List, Tuple
import random
import time
import math
import numpy as np


class MAP_ELITE(Algorithm):
    def __init__(self, config_path_file:str, name:str = "MAP_ELITE", extra_info:Dict[Any, Any] = None, is_rastrigin=False) -> None:
        Algorithm.__init__(self, config_path_file, name, extra_info)
        # Initialize configs
        self.config_map_elite:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["MAP_ELITE", "Archive", "Genome_NN", "Reproduction", "Mutation"])
        self.verbose:bool = True if self.config_map_elite["MAP_ELITE"]["verbose"] == "True" else False
        self.pop_size:int = int(self.config_map_elite["MAP_ELITE"]["pop_size"])
        self.mutation_type:str = self.config_map_elite["Mutation"]["mutation_type"]
        self.start_using_archive_ratio:float = float(self.config_map_elite["MAP_ELITE"]["start_using_archive_ratio"])

        self.is_first_generation:bool = True


        self.reproduction:Reproduction_NN = Reproduction_NN(config_path_file, self.attributes_manager)
        self.mutation_neat:Mutation_NEAT = Mutation_NEAT(config_path_file, self.attributes_manager)
        self.mutation_ga:Mutation = Mutation(config_path_file, self.attributes_manager)
        self.is_sbx:str = True if self.config_map_elite["Reproduction"]["is_sbx"] == "True" else False

        # test rastrigin
        self.is_rastrigin:bool = is_rastrigin
        if is_rastrigin == False:
            self.archive:Archive = Archive(
                                        config_section_name="Archive",
                                        config_path_file=config_path_file, 
                                        genome_builder_function=self.__build_genome_function_nn
                                        )
        else:
            self.__init_rastigin(config_path_file) ## FOR TESTING RASTRIGIN FUNCTION & DEBUGGING


    def run(self, global_population:Population) -> Population:
        if self.is_rastrigin == True:
            return self.run_rastrigin(global_population) ## FOR TESTING RASTRIGIN FUNCTION & DEBUGGING

        # 0 - Get random genome from the archive
        if self.is_first_generation == True:
            self.is_first_generation = False
            self.population_manager = self.__get_genome_from_archive()
            return self.population_manager

        # 1 - Update archive
        self.population_manager.population = global_population.population
        self.archive.update(self.population_manager)

        # 2 - Get genome from the archive
        self.population_manager = self.__get_genome_from_archive()

        # 3 - Reproduction
        self.__reproduction(self.population_manager) # A REVOIR

        # 4 - Mutation
        self.__mutation(self.population_manager) # A REVOIR

        # 6 - Get best genome from the archive (n best)
        self.population_best = self.archive.get_best_population()

        # 7 - Update population
        global_population.population = self.population_manager.population

        return global_population


    def __build_genome_function_nn(self) -> Genome_NN:
        new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_map_elite["Genome_NN"], self.attributes_manager)
        new_genome.nn.set_arbitrary_parameters(is_random=True)
        return new_genome

    def __get_genome_from_archive(self) -> Population:
        if self.archive.get_archive_filled_ratio() < self.start_using_archive_ratio or self.archive.get_archive_filled_nb() < self.pop_size:
            return self.archive.get_new_random_genome_not_from_archive(self.pop_size)
        else:
            return self.archive.get_random_genome_from_archive(self.pop_size)

    def __reproduction(self, population:Population) -> None:
        # 1- Reproduction
        population.population = self.reproduction.reproduction(population, self.pop_size)

    def __mutation(self, population:Population) -> None:
        # GA mutation
        if self.mutation_type == "classic":
            if self.is_sbx == True: raise Exception("SBX mutation is activated in the config file (is_sbx = True) in Reproduction section, but the mutation type is classic, please set is_sbx = False or change the mutation type to sbx")
            self.__mutation_GA(population)
        # NEAT mutation
        elif self.mutation_type == "topology":
            if self.is_sbx == True: raise Exception("SBX mutation is activated in the config file (is_sbx = True) in Reproduction section, but the mutation type is topology, please set is_sbx = False or change the mutation type to sbx")
            self.__mutation_neat(population)
        elif self.mutation_type == "sbx":
            if self.is_sbx == False:
                raise Exception("SBX mutation is not activated in the config file (is_sbx = False) in Reproduction section, please set is_sbx = True or change the mutation type to classic or topology")                
        else: 
            raise Exception("Mutation type not found, can be classic or topology or sbx")

    def __mutation_GA(self, population:Population) -> None:
        # 1 - Mutation (attributes only)
        pop_dict:Dict[int, Genome_NN] = population.population
        # 1.1 - Mutation Neuron (attributes only)
        population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in pop_dict.items() if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_neuron_params}
        self.mutation.attributes.neurons_sigma(population_to_mutate, self.attributes_manager.parameters_neuron_names)
        # 1.2 - Mutation Synapse (attributes only)
        population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in pop_dict.items() if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_synapse_params}
        self.mutation.attributes.synapses_sigma(population_to_mutate, self.attributes_manager.parameters_synapse_names)

    def __mutation_neat(self, population:Population) -> Population:
        population:Population = self.mutation_neat.mutation_neat(population)
        return population

    def __mutation_classic(self, population:Population) -> Population:
        population_dict:Dict[int, Genome_Classic] = population.population
        for genome in population_dict.values():
            genome.parameter = self.mutation.attributes.epsilon_sigma_jit(genome.parameter, sigma_paramater=0.3, min=0, max=1, mu_bias=0, sigma_coef=1)
        return population


    # START test rastrigin
    def __init_rastigin(self, config_path_file:str) -> None:
        # test rastrigin
        self.reproduction:Reproduction_Classic = Reproduction_Classic(config_path_file, self.attributes_manager)
        # self.archive:Archive = Archive(config_path_file, self.__build_genome_function_classic, name=name, nb_generation=nb_generation, folder_path=archive_folder_path)
        self.archive:Archive = Archive(
                                    config_section_name="Archive",
                                    config_path_file=config_path_file, 
                                    genome_builder_function=self.__build_genome_function_classic
                                    )
        self.mutation:Mutation = Mutation(config_path_file, self.attributes_manager)
        self.config_map_elite.update(TOOLS.config_function(config_path_file, ["Genome_Classic"]))
        self.parameter_size:int = int(self.config_map_elite["Genome_Classic"]["parameter_size"])


    def run_rastrigin(self, global_population:Population) -> Population:
        # 0 - Get random genome from the archive
        if self.archive.get_archive_filled_ratio() < self.start_using_archive_ratio or self.archive.get_archive_filled_nb() < self.pop_size:
            self.population_manager:Population = self.archive.get_new_random_genome_not_from_archive(self.pop_size)
        else:
            self.population_manager:Population = self.archive.get_random_genome_from_archive(self.pop_size)
        
        # 1 - Reproduction
        self.__reproduction_rastrigin(self.population_manager)

        # 2 - Mutation
        # self.__mutation_rastrigin(self.population_manager)

        # 4 - Evaluation
        self.__test_rastrigin(self.population_manager)

        # 6 - Update archive
        self.archive.update(self.population_manager)

        # 7 - Get best genome from the archive (10 best)
        self.population_manager = self.archive.get_best_population()

        # 8 - Update population
        global_population.population = self.population_manager.population

        return global_population

    def __test_rastrigin(self, population_manager:Population) -> None:
        population_dict:Dict[int, Genome_Classic] = population_manager.population
        for genome in population_dict.values():
            genome.fitness.score, genome.info["description"] = self.__rastrigin_map_elite(genome.parameter)

    def __rastrigin_map_elite(self, xx:np.ndarray) -> Tuple[float, np.ndarray]:
        x = xx * 10 - 5 # scaling to [-5, 5]
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f, np.array([xx[0], xx[1]])

    def __build_genome_function_classic(self) -> Genome_Classic:
        new_genome:Genome_Classic = Genome_Classic(get_new_genome_id(), self.config_map_elite["Genome_Classic"], self.attributes_manager)
        # new_genome.parameter = self.mutation.attributes.epsilon_mu_sigma_jit(new_genome.parameter, mu_parameter=0, sigma_paramater=1, min=-1, max=1, mu_bias=0, sigma_coef=1)
        new_genome.parameter:np.ndarray = np.random.rand(self.parameter_size)
        return new_genome

    def __reproduction_rastrigin(self, population:Population) -> None:
        population.population = self.reproduction.reproduction(population, self.pop_size)

    def __mutation_rastrigin(self, population:Population) -> None:
        self.__mutation_classic(population)
    # END test rastrigin
