from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population_NN as Population
from typing import Dict, Any, List
import numpy as np
import time

import jax
from evosax import OpenES
from evosax import DE
from evosax import ARS
from evosax import SNES
from evosax import PGPE
from evosax import FitnessShaper


class EvoSax_algo(Algorithm):
    def __init__(self, config_path_file:str, algo_name:str, extra_info:Dict[Any, Any] = None) -> None:
        Algorithm.__init__(self, config_path_file, algo_name, extra_info)

        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, [algo_name,"Genome_NN","NEURO_EVOLUTION", "Runner_Info"])

        self.pop_size:int = int(self.config_es[algo_name]["pop_size"])
        self.is_first_generation:bool = True
        self.genome_core:Genome_NN = Genome_NN(-1, self.config_es["Genome_NN"], self.attributes_manager)
        self.verbose:bool = True if self.config_es[algo_name]["verbose"] == "True" else False
        self.optimization_type:str = self.config_es["NEURO_EVOLUTION"]["optimization_type"]
        self.algo_name:str = algo_name
        self.is_neuron_param:bool = True if "bias" in self.attributes_manager.mu_parameters else False
        self.network_type:str = self.config_es["Genome_NN"]["network_type"]
        if self.network_type == "SNN":
            self.decay_method:str = self.config_es["Runner_Info"]["decay_method"]
        
        
        # Initialize es_algorithms
        self.__init_es_jax(config_path_file=config_path_file)
 
    def __init_es_jax(self, config_path_file:str=None) -> None:
        self.neuron_parameters_size:int = self.genome_core.nn.nb_neurons
        self.synapse_parameters_size:int = self.genome_core.nn.synapses_actives_indexes[0].size

        if self.is_neuron_param:
            parameters_size:int = self.neuron_parameters_size + self.synapse_parameters_size
        else:
            parameters_size:int = self.synapse_parameters_size


        self.jax_seed = jax.random.PRNGKey(np.random.randint(0, 100000))
        if self.algo_name == "DE-evosax":
            self.optimizer:DE = DE(popsize=self.pop_size, num_dims=parameters_size)

        elif self.algo_name == "NES-evosax":
            self.optimizer:SNES = self.__init_snes(config_path_file, parameters_size)

        elif self.algo_name == "ARS-evosax":
            self.optimizer:ARS = self.__init_ars(config_path_file, parameters_size)

        elif self.algo_name == "OpenES-evosax":
            self.optimizer:OpenES = self.__init_openes(config_path_file, parameters_size)

        elif self.algo_name == "PEPG-evosax":
            self.optimizer:PGPE = self.__init_pepg(config_path_file, parameters_size)
        else:
            raise Exception("Algo name not found -> only available DE-evosax, NES-evosax, ARS-evosax, OpenES-evosax, PGPE-evosax")

        self.es_hyperparameters = self.optimizer.default_params
        self.params_state = self.optimizer.initialize(self.jax_seed, self.es_hyperparameters)


    def __init_snes(self, config_path_file:str, parameters_size:int):
        config_snes:Dict[str, Any] = TOOLS.config_function(config_path_file, ["NES-evosax"])["NES-evosax"]
        
        self.optimizer:SNES = SNES(popsize=
                                   self.pop_size, 
                                   num_dims=parameters_size, 
                                   sigma_init=float(config_snes["sigma_init"]), 
                                   temperature=float(config_snes["temperature"]), 
                                   mean_decay=float(config_snes["mean_decay"]))
        return self.optimizer

    def __init_ars(self, config_path_file:str, parameters_size:int):
        config_ars:Dict[str, Any] = TOOLS.config_function(config_path_file, ["ARS-evosax"])["ARS-evosax"]
        
        self.optimizer:ARS = ARS(popsize=self.pop_size, 
                                num_dims=parameters_size, 
                                elite_ratio=float(config_ars["elite_ratio"]),
                                opt_name=config_ars["optimizer"],
                                 
                                lrate_init=float(config_ars["learning_rate"]),
                                lrate_decay=float(config_ars["learning_rate_decay"]),
                                lrate_limit=float(config_ars["learning_rate_limit"]),
                                 
                                sigma_init=float(config_ars["sigma_init"]),
                                sigma_decay=float(config_ars["sigma_decay"]),
                                sigma_limit=float(config_ars["sigma_limit"]),

                                mean_decay=float(config_ars["mean_decay"]))
        return self.optimizer

    
    def __init_openes(self, config_path_file:str, parameters_size:int):
        config_openes:Dict[str, Any] = TOOLS.config_function(config_path_file, ["OpenES-evosax"])["OpenES-evosax"]

        self.optimizer:OpenES = OpenES(popsize=self.pop_size,
                                    num_dims=parameters_size,
                                    opt_name=config_openes["opt_name"],
                                    use_antithetic_sampling=True if config_openes["antithetic_sampling"] == "True" else False,
                                    
                                    lrate_init=float(config_openes["learning_rate"]),
                                    lrate_decay=float(config_openes["learning_rate_decay"]),
                                    lrate_limit=float(config_openes["learning_rate_limit"]),
                                    
                                    sigma_init=float(config_openes["sigma_init"]),
                                    sigma_decay=float(config_openes["sigma_decay"]),
                                    sigma_limit=float(config_openes["sigma_limit"]),
                                    
                                    mean_decay=float(config_openes["mean_decay"])
                                    )
        return self.optimizer
        
    def __init_pepg(self, config_path_file:str, parameters_size:int):
        config_pepg:Dict[str, Any] = TOOLS.config_function(config_path_file, ["PEPG-evosax"])["PEPG-evosax"]

        self.optimizer:PGPE = PGPE(popsize=self.pop_size,
                                num_dims=parameters_size,
                                elite_ratio=float(config_pepg["elite_ratio"]),
                                opt_name=config_pepg["opt_name"],
                                
                                lrate_init=float(config_pepg["learning_rate"]),
                                lrate_decay=float(config_pepg["learning_rate_decay"]),
                                lrate_limit=float(config_pepg["learning_rate_limit"]),
                                
                                sigma_init=float(config_pepg["sigma_init"]),
                                sigma_decay=float(config_pepg["sigma_decay"]),
                                sigma_limit=float(config_pepg["sigma_limit"]),
                                
                                mean_decay=float(config_pepg["mean_decay"])
                                )
        return self.optimizer



    def run(self, global_population:Population) -> Population:

        self.population_manager = global_population
        if self.is_first_generation == True: 
            start_time = time.time()
            self.__first_generation(self.population_manager)
            self.__update_population_parameter_jax(self.population_manager)
            global_population = self.population_manager
            print(self.name+": First generation time:", time.time() - start_time, "s")
            return global_population
        
        # 1 - Update
        self.__update_es_by_fitness(self.population_manager)

        # 2 - Update population parameters
        self.__update_population_parameter_jax(self.population_manager)

        # 3 - Update population
        global_population.population = self.population_manager.population

        return global_population

            
    def __first_generation(self, population_manager:Population) -> None:
        self.is_first_generation = False
        population:Dict[int, Genome_NN] = population_manager.population
        parameters_names:List[str] = self.attributes_manager.mu_parameters.keys()

        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_es["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=False, weight_random=True)
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


    def __update_population_parameter_jax(self, population_manager:Population) -> None:
        # 1 - Get parameters from CMA-ES algorithms
        self.jax_seed, self.jax_seed_gen, self.jax_seed_eval = jax.random.split(self.jax_seed, 3)
        self.population_parameters, self.state = self.optimizer.ask(self.jax_seed_gen, self.params_state, self.es_hyperparameters)


        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            nn:NN = genome.nn
            if nn.parameters["weight"].flags.writeable == False: # Can happen when the genome is from another thread (i.e in parallel computing)
                nn.parameters["weight"] = nn.parameters["weight"].copy()

            # 2.2 Update network parameters
            if self.network_type == "ANN" and self.is_neuron_param == True:
                nn.parameters["weight"][nn.synapses_actives_indexes] = np.array(self.population_parameters[index][:self.synapse_parameters_size])

                if nn.parameters["bias"].flags.writeable == False: nn.parameters["bias"] = nn.parameters["bias"].copy()
                nn.parameters["bias"] = np.array(self.population_parameters[index][self.synapse_parameters_size:])  
                

            elif self.network_type == "SNN" or (self.network_type == "ANN" and self.is_neuron_param == False):
                nn.parameters["weight"][nn.synapses_actives_indexes] = np.array(self.population_parameters[index])

            
    def __update_es_by_fitness(self, population_manager:Population) -> None:
        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        fitnesses:List[int] = []
        for genome in genomes_dict.values():
            fitnesses.append(genome.fitness.score)
        fitnesses:np.ndarray = np.array(fitnesses)

        fit_shaper = FitnessShaper(
                        # centered_rank=True,
                        # z_score=True,
                        # w_decay=0.1,
                        maximize=True if self.optimization_type == "maximize" else False,
                        )
        
        fitnesses = fit_shaper.apply(self.population_parameters, fitnesses)

        # print("elites_indexes", elites_indexes, "size", elites_indexes.size)
        # print("fitnesses", fitnesses, "size", fitnesses.size)
        # print("fitness_max:", fitnesses[elites_indexes[0]], "fitness_min:", fitnesses[elites_indexes[-1]])

        self.params_state = self.optimizer.tell(self.population_parameters, fitnesses, self.params_state, self.es_hyperparameters)