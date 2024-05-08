from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_Decoder
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Distance import Distance
from evo_simulator.GENERAL.Reproduction import Reproduction_Decoder
from GENERAL.Mutation_NN import Mutation

from typing import Dict, Any, List, Callable
import numpy as np
import random
import time


class Decoder_GA(Algorithm):
    def __init__(self, config_path_file:str, name:str = "Decoder_GA") -> None:
        Algorithm.__init__(self, config_path_file, name)
        # Initialize configs
        self.config_decoder:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Decoder", "Genome_Decoder"])
        self.verbose:bool = True if self.config_decoder["Decoder"]["verbose"] == "True" else False

        self.pop_size:int = int(self.config_decoder["Decoder"]["pop_size"])
        self.is_first_generation:bool = True
        self.nb_neurons:int = 0
        self.distance:Distance = Distance(config_path_file)
        self.reproduction:Reproduction_Decoder = Reproduction_Decoder(config_path_file, self.attributes_manager)
        self.mutation:Mutation = Mutation(config_path_file, self.attributes_manager)

        self.init_config()



    def init_config(self) -> None:
        self.decoder_mu_neuron:np.ndarray = np.array([self.attributes_manager.mu_parameters[param][0] for param in self.attributes_manager.parameters_neuron_names])
        self.decoder_sigma_neuron:np.ndarray = np.array([self.attributes_manager.sigma_parameters[param][0] for param in self.attributes_manager.parameters_neuron_names])
        self.decoder_min_neuron:np.ndarray = np.array([self.attributes_manager.min_parameters[param][0] for param in self.attributes_manager.parameters_neuron_names])
        self.decoder_max_neuron:np.ndarray = np.array([self.attributes_manager.max_parameters[param][0] for param in self.attributes_manager.parameters_neuron_names])

        self.decoder_mu_synapse:np.ndarray = np.array([self.attributes_manager.mu_parameters[param][0] for param in self.attributes_manager.parameters_synapse_names])
        self.decoder_sigma_synapse:np.ndarray = np.array([self.attributes_manager.sigma_parameters[param][0] for param in self.attributes_manager.parameters_synapse_names])
        self.decoder_min_synapse:np.ndarray = np.array([self.attributes_manager.min_parameters[param][0] for param in self.attributes_manager.parameters_synapse_names])
        self.decoder_max_synapse:np.ndarray = np.array([self.attributes_manager.max_parameters[param][0] for param in self.attributes_manager.parameters_synapse_names])

        self.mu_bias:float = 0.0
        self.sigma_bias:float = 1.0
        self.sigma_decay:float = float(self.config_decoder["Decoder"]["sigma_decay"])
        self.sigma_min:float = float(self.config_decoder["Decoder"]["sigma_min"])

    def run(self, global_population:Population) -> Population:

        self.population_manager.population = global_population.population
        self.first_generation(self.population_manager)
        self.ajust_population(self.population_manager)
        
        # 1 - Reproduction
        start_time = time.time()
        self.__reproduction(self.population_manager)
        print(self.name+": Reproduction time:", time.time() - start_time, "s")

        # 2 - Mutation
        start_time = time.time()
        self.__mutation_decoder(self.population_manager, with_mu=False)
        print(self.name+": Mutation time:", time.time() - start_time, "s")

        # 3 - Decay sigma
        self.decoder_sigma_neuron[self.decoder_sigma_neuron > self.sigma_min] *= self.sigma_decay
        self.decoder_sigma_synapse[self.decoder_sigma_synapse > self.sigma_min] *= self.sigma_decay
        print("Decoder: sigma_neuron:", self.decoder_sigma_neuron, ", sigma_synapse:", self.decoder_sigma_synapse)

        # 5 - Print stats
        self.__print_stats()

        # 4 - Update population
        global_population.population = self.population_manager.population

        return global_population

            
    def first_generation(self, population_manager:Population) -> None:
        if  population_manager.is_first_generation == True:
            start_time = time.time()
            self.is_first_generation = False
            self.ajust_population(population_manager)
            self.__mutation_decoder(population_manager, with_mu=True)
            population_manager.is_first_generation = False
            print("Decoder: First generation time:", time.time() - start_time, "s")

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_Decoder] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_Decoder = Genome_Decoder(get_new_genome_id(), self.config_decoder["Genome_Decoder"])
            new_genome.info["is_elite"] = False
            population[new_genome.id] = new_genome


    def __reproduction(self, population:Population) -> None:
        # 1- Reproduction
        population.population = self.reproduction.reproduction(population, self.pop_size)


    def __mutation_decoder(self, population:Population, with_mu:bool = False) -> None:
        # 1 - Mutation (attributes only)
        pop_dict:Dict[int, Genome_Decoder] = population.population

        # 1.1 - Mutation
        if with_mu == True:
            for genome in pop_dict.values():
                if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_neuron_params:
                    genome.decoder_neuron = self.mutation.attributes.epsilon_mu_sigma_jit(genome.decoder_neuron, self.decoder_mu_neuron, self.decoder_sigma_neuron, self.decoder_min_neuron, self.decoder_max_neuron, self.mu_bias, self.sigma_bias)
                if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_synapse_params:
                    genome.decoder_synapse = self.mutation.attributes.epsilon_mu_sigma_jit(genome.decoder_synapse, self.decoder_mu_synapse, self.decoder_sigma_synapse, self.decoder_min_synapse, self.decoder_max_synapse, self.mu_bias, self.sigma_bias)
        else:
            for genome in pop_dict.values():
                if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_neuron_params:
                    genome.decoder_neuron = self.mutation.attributes.epsilon_sigma_jit(genome.decoder_neuron, self.decoder_sigma_neuron, self.decoder_min_neuron, self.decoder_max_neuron, self.mu_bias, self.sigma_bias)
                if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_synapse_params:
                    genome.decoder_synapse = self.mutation.attributes.epsilon_sigma_jit(genome.decoder_synapse, self.decoder_sigma_synapse, self.decoder_min_synapse, self.decoder_max_synapse, self.mu_bias, self.sigma_bias)


    def __get_info_stats_population(self):
        stats:List[List[int, float, float, int]] = []
        # self.population_manager.update_info()
        best_fitness:float = self.population_manager.fitness.score
        mean_fitness:float = self.population_manager.fitness.mean
        stagnation:float = self.population_manager.stagnation
        best_genome:Genome_Decoder = self.population_manager.best_genome
        param_size:int = best_genome.decoder_neuron.size + best_genome.decoder_synapse.size
        stats.append([0, len(self.population_manager.population), (best_genome.id, round(best_fitness, 3), param_size), round(mean_fitness, 3), stagnation])
        return stats

    # def __get_info_distance(self):
    #     elite_id:int = self.population_manager.best_genome.id
    #     population_ids:List[int] = self.population_manager.population.keys()
    #     pop_dict:Dict[int, Genome_Decoder] = self.population_manager.population
        # self.distance.distance_genomes_list([elite_id], population_ids, pop_dict, reset_cache=True)
        # print("global distance:", self.distance.mean_distance["global"], ", local distance:", self.distance.mean_distance["local"])
        # mean_distance:float = self.distance.mean_distance["global"]
        # print("Mean_distance (compared with one elite only):", round(mean_distance, 3))
        # print("--------------------------------------------------------------------------------------------------------------------")

    def __print_stats(self):
        if self.verbose == False: return
        self.population_manager.update_info()
        # self.__get_info_distance()
        print("--------------------------------------------------------------------------------------------------------------------->>> " +self.name)
        titles = [[self.name, "Size", "Best(id, fit, param_size)", "Avg", "Stagnation"]]
        titles.extend(self.__get_info_stats_population())
        col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for row in titles:
            print("".join(str(word).ljust(col_width) for word in row))
        print("\n")
