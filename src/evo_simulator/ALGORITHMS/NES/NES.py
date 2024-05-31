from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population
from ALGORITHMS.NES.NES_algo import NES_algo
from evo_simulator.GENERAL.NN import NN
from typing import Dict, Any, List
import numpy as np


class NES(Algorithm): # Natural Evolution Strategies
    def __init__(self, config_path_file:str, name:str = "NES", extra_info:Dict[Any, Any] = None) -> None:
        Algorithm.__init__(self, config_path_file, name, extra_info)
        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NES", "Genome_NN"])
        self.verbose:bool = True if self.config_es["NES"]["verbose"] == "True" else False
        self.algorithme_name:str = name

        self.pop_size:int = int(self.config_es["NES"]["pop_size"])
        self.is_first_generation:bool = True
        self.genome_core:Genome_NN = Genome_NN(-1, self.config_es["Genome_NN"], self.attributes_manager)
        self.network_type:str = self.config_es["Genome_NN"]["network_type"]
        self.is_neuron_param:bool = True if "bias" in self.attributes_manager.mu_parameters else False
        
        # Initialize es_algorithms
        self.init_NES_algorithms(config_path_file)

    def init_NES_algorithms(self, config_path_file:str):
        config_nes:Dict[str, Any] = TOOLS.config_function(config_path_file, ["NES"])["NES"]

        self.neuron_parameters_size:int = self.genome_core.nn.nb_neurons

        self.synapse_parameters_size:int = self.genome_core.nn.synapses_actives_indexes[0].size
        elite_ratio:float = float(config_nes["elite_ratio"])

        # 1. Mu params
        mu_init:float = float(config_nes["mu_init"]) 

        # 2. Sigma params
        sigma_init:float = float(config_nes["sigma_init"])
        sigma_alpha:float = float(config_nes["sigma_alpha"])
        sigma_decay:float = float(config_nes["sigma_decay"])
        sigma_limit:float = float(config_nes["sigma_limit"])
        sigma_max_change:float = float(config_nes["sigma_max_change"])

        # 3. Learning rate params
        learning_rate:float = float(config_nes["learning_rate"])
        learning_rate_decay:float = float(config_nes["learning_rate_decay"])
        learning_rate_limit:float = float(config_nes["learning_rate_limit"])

        # 4. Other params
        average_baseline:bool = True if config_nes["average_baseline"] == "True" else False
        parameters_decay:float = float(config_nes["parameters_decay"])
        rank_fitness:bool = True if config_nes["rank_fitness"] == "True" else False
        forget_best:bool = True if config_nes["forget_best"] == "True" else False
        optimizer_name:str = config_nes["optimizer_name"]


        if self.network_type == "ANN" and self.is_neuron_param == True:
            param_size:int = self.neuron_parameters_size + self.synapse_parameters_size
        elif self.network_type == "SNN" or (self.network_type == "ANN" and self.is_neuron_param == False):
            param_size:int = self.synapse_parameters_size

        self.nes_algo:NES_algo = NES_algo(   
                                    population_size=self.pop_size, 
                                    num_params=param_size,
                                    elite_ratio=elite_ratio,
                                    # 1 - Mu params
                                    mu_init=mu_init,

                                    # 2 - Sigma params
                                    sigma_init=sigma_init,

                                    sigma_alpha=sigma_alpha,
                                    sigma_decay=sigma_decay,
                                    sigma_limit=sigma_limit,
                                    sigma_max_change=sigma_max_change,

                                    # 3 - Learning rate params
                                    learning_rate=learning_rate,
                                    learning_rate_decay=learning_rate_decay,
                                    learning_rate_limit=learning_rate_limit,

                                    # 4 - Other params
                                    average_baseline=average_baseline,
                                    parameters_decay=parameters_decay,
                                    rank_fitness=rank_fitness,
                                    forget_best=forget_best,
                                    optimizer_name=optimizer_name
                                    )


    def run(self, global_population:Population) -> Population:

        self.population_manager.population = global_population.population

        if self.is_first_generation == True: 
            self.is_first_generation = False
            self.first_generation(self.population_manager)
            self.__update_population_parameter(self.population_manager)
            return self.population_manager
        
        # 1 - Update NES
        self.__update_NES_by_fitness(self.population_manager)

        # 2 - Update population parameters
        self.__update_population_parameter(self.population_manager)

        # 3 - Update population
        global_population.population = self.population_manager.population

        return global_population

            
    def first_generation(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        parameters_names:List[str] = self.attributes_manager.mu_parameters.keys()
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_es["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=False, weight_random=False)
            for param_name in parameters_names:
                if param_name in new_genome.nn.parameters: # Parameters are set from the attributes_manager which contains information from your config file otherwise it will set by arbitrary values
                    if param_name == "weight" or param_name == "delay": # synapses parameters
                        new_genome.nn.parameters[param_name][new_genome.nn.synapses_actives_indexes] = TOOLS.epsilon_mu_sigma_jit(
                                                                                        parameter=new_genome.nn.parameters[param_name][new_genome.nn.synapses_actives_indexes],  
                                                                                        mu_parameter=self.attributes_manager.mu_parameters[param_name],
                                                                                        sigma_paramater=self.attributes_manager.sigma_parameters[param_name],
                                                                                        min=self.attributes_manager.min_parameters[param_name],
                                                                                        max=self.attributes_manager.max_parameters[param_name],
                        )
                    else: # neuron parameters
                        new_genome.nn.parameters[param_name] = TOOLS.epsilon_mu_sigma_jit(
                                                                                        parameter=new_genome.nn.parameters[param_name],  
                                                                                        mu_parameter=self.attributes_manager.mu_parameters[param_name],
                                                                                        sigma_paramater=self.attributes_manager.sigma_parameters[param_name],
                                                                                        min=self.attributes_manager.min_parameters[param_name],
                                                                                        max=self.attributes_manager.max_parameters[param_name],
                    )
            population[new_genome.id] = new_genome


    def __update_population_parameter(self, population_manager:Population) -> None:
        # 1 - Get parameters from OpenES algorithms
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        self.population_parameters = self.nes_algo.get_parameters()


        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            nn:NN = genome.nn
            if nn.parameters["weight"].flags.writeable == False: # Can happen when the genome is from another thread (i.e in parallel computing)
                nn.parameters["weight"] = nn.parameters["weight"].copy()

            # 2.2 Update Weight parameters
            if self.network_type == "ANN" and self.is_neuron_param == True:
                nn.parameters["weight"][nn.synapses_actives_indexes] = self.population_parameters[index][:self.synapse_parameters_size]

                if nn.parameters["bias"].flags.writeable == False: nn.parameters["bias"] = nn.parameters["weight"].copy()
                nn.parameters["bias"] = self.population_parameters[index][self.synapse_parameters_size:]

            elif self.network_type == "SNN" or (self.network_type == "ANN" and self.is_neuron_param == False):
                nn.parameters["weight"][nn.synapses_actives_indexes] = self.population_parameters[index]
            

    def __update_NES_by_fitness(self, population_manager:Population) -> None:
        # !!! Important to keep the same order of the fitnesses and the genomes and the parameters generated by the OpenES
        # Because of the antithetic sampling, the order of the fitnesses and the genomes and the parameters generated by the OpenES are not the same

        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        fitnesses:List[int] = []
        for genome in genomes_dict.values():
            fitnesses.append(genome.fitness.score)
        fitnesses:np.ndarray = np.array(fitnesses)

        # Update NES
        self.nes_algo.update(None, fitnesses)
