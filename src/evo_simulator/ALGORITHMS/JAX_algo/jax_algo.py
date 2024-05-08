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


class Jax_algo(Algorithm):
    def __init__(self, config_path_file:str, algo_name:str) -> None:
        Algorithm.__init__(self, config_path_file, algo_name)
        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["JAX", "Genome_NN","NEURO_EVOLUTION", "Runner_Info"])

        self.pop_size:int = int(self.config_es["JAX"]["pop_size"])
        self.is_first_generation:bool = True
        self.genome_core:Genome_NN = Genome_NN(-1, self.config_es["Genome_NN"], self.attributes_manager)
        self.verbose:bool = True if self.config_es["JAX"]["verbose"] == "True" else False
        self.optimization_type:str = self.config_es["NEURO_EVOLUTION"]["optimization_type"]
        self.algo_name:str = algo_name
        self.is_neuron_param:bool = True if "bias" in self.attributes_manager.mu_parameters else False
        self.network_type:str = self.config_es["Genome_NN"]["network_type"]
        if self.network_type == "SNN":
            self.decay_method:str = self.config_es["Runner_Info"]["decay_method"]
        
        
        # Initialize es_algorithms
        self.init_es_jax()
 
    def init_es_jax(self):
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
            # self.optimizer:SNES = SNES(popsize=self.pop_size, num_dims=parameters_size, sigma_init=2.0) # SNN (Walker2d-SNN)
            self.optimizer:SNES = SNES(popsize=self.pop_size, num_dims=parameters_size, sigma_init=1.0) # SNN (Walker2d-SNN)
            # self.optimizer:SNES = SNES(popsize=self.pop_size, num_dims=parameters_size, sigma_init=0.25) # SNN (lunar lander)
            # self.optimizer:SNES = SNES(popsize=self.pop_size, num_dims=parameters_size, sigma_init=0.03) # ANN
        elif self.algo_name == "ARS-evosax":
            self.optimizer:ARS = ARS(popsize=self.pop_size, num_dims=parameters_size, opt_name="adam")
        elif self.algo_name == "OPENES-evosax":
            self.optimizer:OpenES = OpenES(popsize=self.pop_size, num_dims=parameters_size, opt_name="adam", sigma_init=0.03, lrate_init=0.015)
        elif self.algo_name == "PEPG-evosax":
            self.optimizer:PGPE = PGPE(popsize=self.pop_size, num_dims=parameters_size, sigma_init=0.03, lrate_init=0.015)
        else:
            raise Exception("Algo name not found -> only available DE-evosax, NES-evosax, ARS-evosax, OpenES-evosax, PGPE-evosax")

        self.es_hyperparameters = self.optimizer.default_params
        self.params_state = self.optimizer.initialize(self.jax_seed, self.es_hyperparameters)



    def run(self, global_population:Population) -> Population:

        self.population_manager = global_population
        if self.is_first_generation == True: 
            start_time = time.time()
            self.first_generation(self.population_manager)
            self.__update_population_parameter_jax(self.population_manager)
            global_population = self.population_manager
            print(self.name+": First generation time:", time.time() - start_time, "s")
            return global_population
        self.ajust_population(self.population_manager)
        
        # 1 - Update
        self.__update_es_by_fitness(self.population_manager)

        # 2 - Update population parameters
        self.__update_population_parameter_jax(self.population_manager)

        # 3 - Update population
        global_population.population = self.population_manager.population

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
                if self.decay_method == "lif":
                    new_genome.nn.parameters["tau"][:] = 200
                else:
                    new_genome.nn.parameters["tau"][:] = 0.1
            population[new_genome.id] = new_genome


    def __update_population_parameter_jax(self, population_manager:Population) -> None:
        # 1 - Get parameters from CMA-ES algorithms
        self.jax_seed, self.jax_seed_gen, self.jax_seed_eval = jax.random.split(self.jax_seed, 3)
        self.population_parameters, self.state = self.optimizer.ask(self.jax_seed_gen, self.params_state, self.es_hyperparameters)


        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            nn:NN = genome.nn

            # 2.2 Update network parameters
            if self.network_type == "ANN" and self.is_neuron_param == True:
                nn.parameters["weight"] = nn.parameters["weight"].copy()
                nn.parameters["weight"][nn.synapses_actives_indexes] = np.array(self.population_parameters[index][:self.synapse_parameters_size])
                nn.parameters["bias"] = np.array(self.population_parameters[index][self.synapse_parameters_size:])  
                

            elif self.network_type == "SNN" or (self.network_type == "ANN" and self.is_neuron_param == False):
                nn.parameters["weight"] = nn.parameters["weight"].copy()
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