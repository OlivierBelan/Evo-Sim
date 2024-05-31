from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES_algorithm import CMA_ES_algorithm
from typing import Dict, Any, List
import numpy as np
import time


class CMA_ES(Algorithm):
    def __init__(self, config_path_file:str, name:str = "CMA_ES", extra_info:Dict[Any, Any] = None) -> None:
        Algorithm.__init__(self, config_path_file, name, extra_info)
        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["CMA_ES", "Genome_NN", "Runner_Info"])

        self.pop_size:int = int(self.config_es["CMA_ES"]["pop_size"])
        self.is_first_generation:bool = True
        self.genome_core:Genome_NN = Genome_NN(-1, self.config_es["Genome_NN"], self.attributes_manager)
        self.verbose:bool = True if self.config_es["CMA_ES"]["verbose"] == "True" else False
        self.network_type:str = self.config_es["Genome_NN"]["network_type"]
        if self.network_type == "SNN":
            self.decay_method:str = self.config_es["Runner_Info"]["decay_method"]
        
        
        self.__init_es_algorithms()

    def __init_es_algorithms(self):
        self.neuron_names:List[str] = list(set(self.attributes_manager.parameters_neuron_names))
        self.synapse_names:List[str] = list(set(self.attributes_manager.parameters_synapse_names))
        self.cma_parameter_names:List[str] = self.neuron_names + self.synapse_names
        self.neuron_parameters_size = self.genome_core.nn.nb_neurons
        self.neuron_parameters_total_size:int = self.genome_core.nn.nb_neurons * len(self.neuron_names)
        self.synapse_parameters_size = self.genome_core.nn.synapses_actives_indexes[0].size
        self.synapse_parameters_total_size:int = self.genome_core.nn.synapses_actives_indexes[0].size * len(self.synapse_names)
        param_size:int = self.neuron_parameters_total_size + self.synapse_parameters_total_size
        elite_size:int = np.floor(self.pop_size * float(self.config_es["CMA_ES"]["elites_ratio"])).astype(int)        
        is_clipped:bool = True if self.config_es["CMA_ES"]["is_clipped"] == "True" else False

        # print("self.neuron_names", self.neuron_names)
        # print("self.synapse_names", self.synapse_names)
        # print("neuron_parameters_size", self.neuron_parameters_size)
        # print("neuron_parameters_total_size", self.neuron_parameters_total_size)
        # print("synapse_parameters_size", self.synapse_parameters_size)
        # print("synapse_parameters_total_size", self.synapse_parameters_total_size)
        # print("param_size", param_size)
        # print("elite_size", elite_size)
        # print("is_clipped", is_clipped)
        # exit()
        mu:float = float(self.config_es["CMA_ES"]["mu"])
        mu_max:float = float(self.config_es["CMA_ES"]["mu_max"])
        mu_min:float = float(self.config_es["CMA_ES"]["mu_min"])

        sigma:float = float(self.config_es["CMA_ES"]["sigma"])
        sigma_max:float = float(self.config_es["CMA_ES"]["sigma_max"])
        sigma_min:float = float(self.config_es["CMA_ES"]["sigma_min"])



        self.cma_algo:CMA_ES_algorithm =  CMA_ES_algorithm(
                                                self.pop_size, 
                                                param_size,
                                                elite_size,
                                                mu,
                                                sigma,
                                                # Mu max and min
                                                mu_max,
                                                mu_min,
                                                # Sigma max and min
                                                sigma_max,
                                                sigma_min,
                                                # Is clipped
                                                is_clipped=is_clipped,
            )

    def run(self, global_population:Population) -> Population:

        self.population_manager.population = global_population.population
        if self.is_first_generation == True:
            start_time = time.time()
            self.first_generation(self.population_manager)
            self.__update_population_parameter(self.population_manager)
            global_population = self.population_manager
            print(self.name+": First generation time:", time.time() - start_time, "s")
            return global_population
        self.ajust_population(self.population_manager)
        
        # 0 - Update CMA-ES
        self.__update_cma_es_by_fitness(self.population_manager)

        # 1 - Update population parameters
        self.__update_population_parameter(self.population_manager)

        # 2 - Update population
        global_population = self.population_manager
        print("CMA SIGMA", np.mean(self.cma_algo.sigma))

        return global_population

            
    def first_generation(self, population_manager:Population) -> None:
        self.is_first_generation = False
        self.ajust_population(population_manager)

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_es["Genome_NN"], self.attributes_manager)
            new_genome.info["is_elite"] = False
            new_genome.nn.set_arbitrary_parameters(is_random=False, weight_random=True)
            if self.network_type == "SNN":
                new_genome.nn.parameters["threshold"][:] = 0.1
                if self.decay_method == "lif":
                    new_genome.nn.parameters["tau"][:] = 200
                else:
                    new_genome.nn.parameters["tau"][:] = 0.1
            population[new_genome.id] = new_genome

    def __update_population_parameter(self, population_manager:Population) -> None:
        # 1 - Get parameters from CMA-ES algorithms
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        cma_parameters:np.ndarray = self.cma_algo.get_parameters()

        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            nn:NN = genome.nn
            parameters_index:int = 0
            for name in self.cma_parameter_names:
                # 2.1 Update neuron parameters
                if name in self.neuron_names:
                    nn.parameters[name] = nn.parameters[name].copy()
                    nn.parameters[name] = cma_parameters[index, parameters_index:parameters_index+self.neuron_parameters_size]
                    parameters_index += self.neuron_parameters_size
                # 2.2 Update synapse parameters
                if name in self.synapse_names:
                    nn.parameters[name] = nn.parameters[name].copy()
                    nn.parameters[name][nn.synapses_actives_indexes] = cma_parameters[index, parameters_index:parameters_index+self.synapse_parameters_size]
                    parameters_index += self.synapse_parameters_size


    def __update_cma_es_by_fitness(self, population_manager:Population) -> None:
        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        fitnesses:List[float] = []
        for genome in genomes_dict.values():
            fitnesses.append(genome.fitness.score)
        fitnesses:np.ndarray = np.array(fitnesses)
        elites_indexes:np.ndarray = fitnesses.argsort()[::-1]

        # print("elites_indexes", elites_indexes, "size", elites_indexes.size)
        # print("fitnesses", fitnesses, "size", fitnesses.size)
        # print("fitness_max:", fitnesses[elites_indexes[0]], "fitness_min:", fitnesses[elites_indexes[-1]])

        # Update CMA-ES
        self.cma_algo.update(elites_indexes, fitnesses)
