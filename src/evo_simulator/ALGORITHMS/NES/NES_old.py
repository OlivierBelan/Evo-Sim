from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Distance import Distance
from ALGORITHMS.NES.NES_algo import PEPG
from ALGORITHMS.NES.OpenES_algo import OpenAI_ES
from evo_simulator.GENERAL.NN import NN
from typing import Dict, Any, List
import numpy as np


class NES(Algorithm): # Natural Evolution Strategies
    def __init__(self, config_path_file:str, name:str = "NES") -> None:
        Algorithm.__init__(self, config_path_file, name)
        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NES", "Genome_NN"])
        self.verbose:bool = True if self.config_es["NES"]["verbose"] == "True" else False

        self.pop_size:int = int(self.config_es["NES"]["pop_size"])
        self.is_first_generation:bool = True
        self.distance:Distance = Distance(config_path_file)
        self.genome_core:Genome_NN = Genome_NN(-1, self.config_es["Genome_NN"], self.attributes_manager)
        self.network_type:str = self.config_es["Genome_NN"]["network_type"]
        self.is_neuron_param:bool = True if "bias" in self.attributes_manager.mu_parameters else False
        
        # Initialize es_algorithms
        algorithme_name:str = name
        # algorithme_dict:Dict[str, Callable] = {"PEPG":self.init_PEPG_algorithms, "OPENAI_ES":self.init_OpenAI_ES_algorithms}
        # if algorithme_name not in algorithme_dict: raise Exception("The algorithme name is not correct. Please choose between", algorithme_dict.keys())
        # algorithme_dict[algorithme_name](config_path_file)
        if algorithme_name == "PEPG":
            self.init_PEPG_algorithms(config_path_file)
        elif algorithme_name == "OPENAI_ES":
            self.init_OpenAI_ES_algorithms(config_path_file)
        else:
            raise Exception("The algorithme name is not correct. Please choose between", ["PEPG", "OPENAI_ES"])

    def init_PEPG_algorithms(self, config_path_file:str):
        config_pepg:Dict[str, Any] = TOOLS.config_function(config_path_file, ["PEPG"])["PEPG"]

        self.neuron_parameters_size:int = self.genome_core.nn.nb_neurons

        self.synapse_parameters_size:int = self.genome_core.nn.synapses_actives_indexes[0].size
        elite_ratio:float = float(config_pepg["elite_ratio"])

        # 1. Mu params
        mu_init:float = float(config_pepg["mu_init"]) 

        # 2. Sigma params
        sigma_init:float = float(config_pepg["sigma_init"])
        sigma_alpha:float = float(config_pepg["sigma_alpha"])
        sigma_decay:float = float(config_pepg["sigma_decay"])
        sigma_limit:float = float(config_pepg["sigma_limit"])
        sigma_max_change:float = float(config_pepg["sigma_max_change"])

        # 3. Learning rate params
        learning_rate:float = float(config_pepg["learning_rate"])
        learning_rate_decay:float = float(config_pepg["learning_rate_decay"])
        learning_rate_limit:float = float(config_pepg["learning_rate_limit"])

        # 4. Other params
        average_baseline:bool = True if config_pepg["average_baseline"] == "True" else False
        parameters_decay:float = float(config_pepg["parameters_decay"])
        rank_fitness:bool = True if config_pepg["rank_fitness"] == "True" else False
        forget_best:bool = True if config_pepg["forget_best"] == "True" else False
        optimizer_name:str = config_pepg["optimizer_name"]


        if self.network_type == "ANN" and self.is_neuron_param == True:
            param_size:int = self.neuron_parameters_size + self.synapse_parameters_size
        elif self.network_type == "SNN" or (self.network_type == "ANN" and self.is_neuron_param == False):
            param_size:int = self.synapse_parameters_size

        self.optmizer:PEPG = PEPG(   
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

    def init_OpenAI_ES_algorithms(self, config_path_file:str):
        config_onpenAI_ES:Dict[str, Any] = TOOLS.config_function(config_path_file, ["OPENAI_ES"])["OPENAI_ES"]

        self.neuron_parameters_size:int = self.genome_core.nn.nb_neurons

        self.synapse_parameters_size:int = self.genome_core.nn.synapses_actives_indexes[0].size

        # 1. Mu params
        mu_init:float = float(config_onpenAI_ES["mu_init"])

        # 2. Sigma params
        sigma_init:float = float(config_onpenAI_ES["sigma_init"])
        sigma_decay:float = float(config_onpenAI_ES["sigma_decay"])
        sigma_limit:float = float(config_onpenAI_ES["sigma_limit"])

        # 3. Learning rate params
        learning_rate:float = float(config_onpenAI_ES["learning_rate"])
        learning_rate_decay:float = float(config_onpenAI_ES["learning_rate_decay"])
        learning_rate_limit:float = float(config_onpenAI_ES["learning_rate_limit"])

        # 4. Other params
        antithetic:bool = True if config_onpenAI_ES["antithetic"] == "True" else False
        parameters_decay:float = float(config_onpenAI_ES["parameters_decay"])
        rank_fitness:bool = True if config_onpenAI_ES["rank_fitness"] == "True" else False
        forget_best:bool = True if config_onpenAI_ES["forget_best"] == "True" else False
        optimizer_name:str = config_onpenAI_ES["optimizer_name"]

        if self.network_type == "ANN" and self.is_neuron_param == True:
            param_size:int = self.neuron_parameters_size + self.synapse_parameters_size
        elif self.network_type == "SNN" or (self.network_type == "ANN" and self.is_neuron_param == False):
            param_size:int = self.synapse_parameters_size

        self.optmizer:OpenAI_ES = OpenAI_ES(   
                                        population_size=self.pop_size, 
                                        num_params=param_size,
                                        # 1 - Mu params
                                        mu_init=mu_init,

                                        # 2 - Sigma params
                                        # sigma_init=self.attributes_manager.sigma_parameters[param][0],
                                        sigma_init=sigma_init,
                                        sigma_decay=sigma_decay,
                                        sigma_limit=sigma_limit,

                                        # 3 - Learning rate params
                                        learning_rate=learning_rate,
                                        learning_rate_decay=learning_rate_decay,
                                        learning_rate_limit=learning_rate_limit,

                                        # 4 - Other params
                                        antithetic=antithetic,
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
        self.ajust_population(self.population_manager)
        
        # 1 - Update NES
        self.__update_NES_by_fitness(self.population_manager)

        # 2 - Update population parameters
        self.__update_population_parameter(self.population_manager)

        # 3 - Update population
        global_population.population = self.population_manager.population

        return global_population

            
    def first_generation(self, population_manager:Population) -> None:
        self.ajust_population(population_manager)

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_es["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=False, weight_random=True)
            population[new_genome.id] = new_genome

    def __update_population_parameter(self, population_manager:Population) -> None:
        # 1 - Get parameters from NES algorithms
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        self.population_parameters = self.optmizer.get_parameters()


        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            nn:NN = genome.nn
            # 2.2 Update Weight parameters
            if self.network_type == "ANN":
                nn.parameters["weight"] = nn.parameters["weight"].copy()
                if self.is_neuron_param == True:
                    nn.parameters["weight"][nn.synapses_actives_indexes] = self.population_parameters[index][:self.synapse_parameters_size]
                    nn.parameters["bias"] = self.population_parameters[index][self.synapse_parameters_size:]
                else:
                    nn.parameters["weight"][nn.synapses_actives_indexes] = self.population_parameters[index]

            elif self.network_type == "SNN":
                nn.parameters["weight"] = nn.parameters["weight"].copy()
                nn.parameters["weight"][nn.synapses_actives_indexes] = self.population_parameters[index]
            

    def __update_NES_by_fitness(self, population_manager:Population) -> None:
        # !!! Important to keep the same order of the fitnesses and the genomes and the parameters generated by the NES
        # Because of the antithetic sampling, the order of the fitnesses and the genomes and the parameters generated by the NES are not the same

        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        fitnesses:List[int] = []
        for genome in genomes_dict.values():
            fitnesses.append(genome.fitness.score)
        fitnesses:np.ndarray = np.array(fitnesses)

        # Update NES
        # self.parameters_NES["weight"].update(None, fitnesses)
        self.optmizer.update(None, fitnesses)
