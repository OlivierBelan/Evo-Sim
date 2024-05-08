from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN, Genome_Classic, Genome
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id, get_new_population_id
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Distance import Distance
from evo_simulator.GENERAL.Reproduction import Reproduction_NN, Reproduction_Classic
from GENERAL.Mutation_NN import Mutation
from evo_simulator.GENERAL.Optimizer import minimize, maximize, closest_to_zero
from typing import Dict, Any, List, Callable, Set, Tuple
import numpy as np
import random
import time
import math

class NSGA(Algorithm):
    def __init__(self, config_path_file:str, name:str = "NSGA") -> None:
        Algorithm.__init__(self, config_path_file, name)
        # Initialize configs
        self.config_nsga:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NSGA", "Genome_NN", "Genome_Classic"])
        self.verbose:bool = True if self.config_nsga["NSGA"]["verbose"] == "True" else False
        self.pop_size:int = int(self.config_nsga["NSGA"]["pop_size"])
        self.fitness_dimension:int = int(self.config_nsga["NSGA"]["fitness_dimension"])
        self.optimization_type:str = self.config_nsga["NSGA"]["optimization_type"]

        # Initialize tools
        self.distance:Distance = Distance(config_path_file)
        # self.reproduction:Reproduction_NN = Reproduction_NN(config_path_file, self.attributes_manager)
        self.reproduction:Reproduction_Classic = Reproduction_Classic(config_path_file, self.attributes_manager, self.__build_new_genome)
        self.mutation:Mutation = Mutation(config_path_file, self.attributes_manager)

        # Mutable parameters
        self.is_first_generation:bool = True
        self.population_rank:List[List[Genome]] = [[]]
        self.population_parent:Population = Population(get_new_population_id, config_path_file)
        self.population_child:Population = Population(get_new_population_id, config_path_file)
        self.population_merge:Population = Population(get_new_population_id, config_path_file)
        self.generation = 0

    def run(self, global_population:Population, evaluation_function) -> Population:

        self.first_generation(self.population_parent, evaluation_function)
        self.ajust_population(self.population_parent)
        
        # 1 - Reproduction
        self.population_child = self.__reproduction(self.population_parent, criteria="crowding_distance")

        # 2 - Mutation
        self.__mutation(self.population_child)

        # 4 - Evaluation
        self.test_kursawe(self.population_child)

        # 6 - NSGA algorithm
        self.population_parent = self.__NSGA(self.population_parent, self.population_child)

        # # 5 - Print stats
        # self.__print_stats()

        # 4 - Update population
        global_population.population = self.population_parent.population
        self.population_manager = self.population_parent
        self.generation += 1
        if self.generation == 100:
            self.plot(self.population_parent)

        return global_population

            
    def first_generation(self, population_manager:Population, evaluation_function:Callable) -> None:
        if  population_manager.is_first_generation == True:
            start_time = time.time()
            # population:Dict[int, Genome_NN] = population_manager.population
            self.is_first_generation = False
            self.ajust_population(population_manager)
            # self.mutation.attributes.first_mutation_attributes_mu_sigma(population, global_parameters.nn_neuron_parameters, global_parameters.nn_synapse_parameters)
            population_manager.is_first_generation = False
            # self.__evalutation(population_manager, evaluation_function)
            self.test_kursawe(population_manager)
            print(self.name+": First generation time:", time.time() - start_time, "s")

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = self.__build_new_genome()
            population[new_genome.id] = new_genome

    def __build_new_genome(self) -> Genome:
        # new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_nsga["Genome_NN"])
        new_genome:Genome_Classic = Genome_Classic(get_new_genome_id(), self.config_nsga["Genome_Classic"], self.attributes_manager)
        new_genome.info["is_elite"] = False
        # NSGA parameters
        new_genome.info["rank"]:int = -1
        new_genome.info["crowding_distance"]:float = 0.0
        new_genome.info["domination_count"]:int = 0
        new_genome.info["domination_set"]:Set[Genome_Classic] = set()
        # new_genome.info["domination_set"]:Set[Genome_NN] = set()
        new_genome.parameter = np.random.uniform(-5, 5, new_genome.parameter.shape) # for test
        new_genome.info["fitnesses"]:np.ndarray = np.zeros(self.fitness_dimension, dtype=np.float32)
        new_genome.info["fitnesses_normalized"]:np.ndarray = np.zeros(self.fitness_dimension, dtype=np.float32)
        return new_genome

    def __evalutation(self, population_manager:Population, evaluation_function:Callable) -> None:
        if evaluation_function is not None:
            evaluation_function(population_manager.population)
        else:
            self.evaluation_function(population_manager.population)

    def __reproduction(self, population_parent:Population, criteria:str = "crowding_distance") -> Population:
        # 1- Reproduction
        self.population_child.population = self.reproduction.reproduction(population_parent, self.pop_size, criteria=criteria, replace=False)
        return self.population_child

    def __mutation(self, population:Population) -> None:
        # # 1 - Mutation (attributes only)
        # pop_dict:Dict[int, Genome_NN] = population.population
        # # 1.1 - Mutation Neuron (attributes only)
        # population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in pop_dict.items() if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_neuron_params}
        # self.mutation.attributes.neurons_sigma(population_to_mutate, global_parameters.nn_neuron_parameters)
        # # 1.2 - Mutation Synapse (attributes only)
        # population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in pop_dict.items() if genome.info["is_elite"] == False and random.random() < self.mutation.prob_mutate_synapse_params}
        # self.mutation.attributes.synapses_sigma(population_to_mutate, global_parameters.nn_synapse_parameters)
        pop_dict:Dict[int, Genome_Classic] = population.population
        for genome in pop_dict.values():
            # print("before mut id:", genome.id, "parameter", genome.parameter, "fitness:", genome.info["fitnesses"])
            if self.mutation.prob_mutation > random.random():
                genome.parameter = self.mutation.attributes.epsilon_sigma_jit(genome.parameter, self.attributes_manager.sigma_parameters["classic"], self.attributes_manager.min_parameters["classic"], self.attributes_manager.max_parameters["classic"], 0, 1)
            # print("after mut id:", genome.id, "parameter", genome.parameter, "fitness:", genome.info["fitnesses"])
        # exit()


    def __non_dominated_sorting(self, population:Population) -> List[List[Genome]]:
        population_dict:Dict[int, Genome_NN] = population.population
        ranking_count:int = 0
        ranking:List[List[int]] = [[]]
        self.population_rank:List[List[Genome]] = [[]]
        for id_1, genome_1 in population_dict.items():
            # 1 - Initialize
            genome_1.info["domination_set"] = set()
            genome_1.info["domination_count"] = 0

            # 2 - Compare with all other genomes
            for id_2, genome_2 in population_dict.items():
                if id_1 == id_2: continue # Skip if same genome

                # 2.1 - Compare fitnesses for domination
                if self.__dominates(genome_1, genome_2):
                    genome_1.info["domination_set"].add(id_2)

                elif self.__dominates(genome_2, genome_1):
                    genome_1.info["domination_count"] += 1
                        

            # 3 - Check if genome is rank 0 (best rank)
            if genome_1.info["domination_count"] == 0:
                genome_1.info["rank"] = 0
                ranking[0].append(id_1)
                self.population_rank[0].append(genome_1)

        # 4 - Rank all genomes
        while len(ranking[ranking_count]) > 0:
            ranking.append([])
            self.population_rank.append([])
            for id_1 in ranking[ranking_count]:
                genome_1:Genome_NN = population_dict[id_1]
                for id_2 in genome_1.info["domination_set"]:
                    genome_2:Genome_NN = population_dict[id_2]
                    genome_2.info["domination_count"] -= 1
                    if genome_2.info["domination_count"] == 0:
                        genome_2.info["rank"] = ranking_count + 1
                        ranking[ranking_count + 1].append(id_2)
                        self.population_rank[ranking_count + 1].append(genome_2)
            ranking_count += 1

        return self.population_rank
                
    def __dominates(self, genome_1:Genome_NN, genome_2:Genome_NN) -> bool:
        '''
            Return True if genome_1 dominates genome_2
        '''
        for i in range(self.fitness_dimension):

            if self.optimization_type == "closest_to_zero":
                if abs(genome_1.info["fitnesses"][i]) > abs(genome_2.info["fitnesses"][i]):
                    return False

            elif self.optimization_type == "maximize":
                if genome_1.info["fitnesses"][i] < genome_2.info["fitnesses"][i]:
                    return False

            elif self.optimization_type == "minimize":
                if genome_1.info["fitnesses"][i] > genome_2.info["fitnesses"][i]:
                    return False
        return True        

    def __crowding_distance(self, population_list:List[Genome]) -> List[Genome]:
        if len(population_list) <= 4: return
        for genome in population_list:
            genome.info["crowding_distance"] = 0

        for i in range(self.fitness_dimension):
            self.__crowding_distance_dimension(population_list, i)
        
    def __crowding_distance_dimension(self, population_list:List[Genome], dimension:int) -> None:
        # 1 - Sort population by fitness in dimension
        population_sorted:List[Genome_NN] = sorted(population_list, key=lambda genome: genome.info["fitnesses"][dimension])
        max_fitness:float = population_sorted[-1].info["fitnesses"][dimension]
        min_fitness:float = population_sorted[0].info["fitnesses"][dimension]

        # 2 - Set crowding distance
        population_sorted[0].info["crowding_distance"] = -float("inf")
        population_sorted[-1].info["crowding_distance"] = -float("inf")

        for i in range(1, len(population_sorted) - 1):
            population_sorted[i].info["crowding_distance"] += float(abs(population_sorted[i + 1].info["fitnesses"][dimension] - population_sorted[i - 1].info["fitnesses"][dimension]) / (max_fitness - min_fitness + 1e-10))
    

    def __crowding_comparison(self, genome_1:Genome_NN, genome_2:Genome_NN) -> Genome:
        # 1 - Check the rank
        if genome_1.info["rank"] < genome_2.info["rank"]:
            return genome_1
        elif genome_1.info["rank"] > genome_2.info["rank"]:
            return genome_2

        # 2 - If same rank, then check the crowding distance
        elif genome_1.info["crowding_distance"] > genome_2.info["crowding_distance"]:
            return genome_1
        else:
            return genome_2


    def __NSGA(self, population_parent:Population, population_child:Population) -> Population:

        # 1 - Merge
        self.population_merge.replace(population_parent)
        self.population_merge.merge(population_child)

        # 2 - Non dominated sorting
        self.population_rank = self.__non_dominated_sorting(self.population_merge)

        # 3 - Build new parent population
        new_population_parent:Dict[int, Genome] = {}
        for rank in self.population_rank:
            # 3.1 - Add all genomes from this rank to parent population if does not exceed pop_size
            if len(new_population_parent) + len(rank) <= self.pop_size:
                self.__crowding_distance(rank)
                for genome in rank:
                    new_population_parent[genome.id] = genome

            # 3.2 - Otherwise, add best (crowding_distance) genomes from this rank to parent population
            else:
                # 3.2.1 - Compute crowding distance
                self.__crowding_distance(rank)

                # 3.2.2 - Sort by crowding distance
                rank.sort(key=lambda genome: genome.info["crowding_distance"], reverse=True)

                # 3.2.3 - Add best genomes to parent population
                for genome in rank:
                    if len(new_population_parent) < self.pop_size:
                        new_population_parent[genome.id] = genome
                    else:
                        break
        # 4 - Update parent population
        population_parent.population = new_population_parent
        # print("--------------------------------------")
        # print("PARENT")
        # for genome in population_parent.population.values():
        #     print("id:", genome.id, "parameter", genome.parameter, "fitness:", genome.info["fitnesses"], "rank:", genome.info["rank"], "crowding_distance:", genome.info["crowding_distance"])
        # exit()

        # 5 - Free memory
        self.population_merge.population = {}
        self.population_child.population = {}

        return population_parent




    # Multiobjectives kursawe problem
    def kursawe(self,individual:np.ndarray) -> Tuple[float, float]:
        r"""Kursawe multiobjective function.

        :math:`f_{\text{Kursawe}1}(\mathbf{x}) = \sum_{i=1}^{N-1} -10 e^{-0.2 \sqrt{x_i^2 + x_{i+1}^2} }`

        :math:`f_{\text{Kursawe}2}(\mathbf{x}) = \sum_{i=1}^{N} |x_i|^{0.8} + 5 \sin(x_i^3)`

        .. plot:: code/benchmarks/kursawe.py
        :width: 100 %
        """
        f1 = sum(-10 * math.exp(-0.2 * math.sqrt(x * x + y * y)) for x, y in zip(individual[:-1], individual[1:]))
        f2 = sum(abs(x)**0.8 + 5 * math.sin(x * x * x) for x in individual)
        return f1, f2

    def test_kursawe(self, population:Population):
        population_dict:Dict[int, Genome_Classic] = population.population
        for genome in population_dict.values():
            genome.info["fitnesses"] = np.array(self.kursawe(genome.parameter), dtype=np.float32)

    def plot(self, population:Population) -> None:
        import matplotlib.pyplot as plt
        
        population_dict:Dict[int, Genome_Classic] = population.population
        f1 = [genome.info["fitnesses"][0] for genome in population_dict.values()]
        f2 = [genome.info["fitnesses"][1] for genome in population_dict.values()]
        print(f1)
        print(f2)
        # plt.scatter(f1, f2)
        # plt.show()
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter(x=f1, y=f2, mode='markers')])
        fig.show()
        

    def __get_info_stats_population(self):
        stats:List[List[int, float, float, int]] = []
        # self.population_manager.update_info()
        best_fitness:float = self.population_manager.fitness.score
        mean_fitness:float = self.population_manager.fitness.mean
        stagnation:float = self.population_manager.stagnation
        best_genome:Genome_NN = self.population_manager.best_genome
        nb_neurons:int = len(best_genome.nn.hiddens["neurons_indexes_active"])
        nb_synapses:int = best_genome.nn.synapses_actives_indexes[0].size
        stats.append([0, len(self.population_manager.population), (best_genome.id, round(best_fitness, 3), nb_neurons, nb_synapses), round(mean_fitness, 3), stagnation])
        return stats

    def __get_info_distance(self):
        elite_id:int = self.population_manager.best_genome.id
        population_ids:List[int] = self.population_manager.population.keys()
        pop_dict:Dict[int, Genome_NN] = self.population_manager.population
        self.distance.distance_genomes_list([elite_id], population_ids, pop_dict, reset_cache=True)
        print("global distance:", self.distance.mean_distance["global"], ", local distance:", self.distance.mean_distance["local"])
        mean_distance:float = self.distance.mean_distance["global"]
        print("Mean_distance (compared with one elite only):", round(mean_distance, 3))
        print("--------------------------------------------------------------------------------------------------------------------->>> " +self.name)

    def __print_stats(self):
        if self.verbose == False: return
        self.population_manager.update_info()
        self.__get_info_distance()
        titles = [[self.name, "Size", "Best(id, fit, neur, syn)", "Avg", "Stagnation"]]
        titles.extend(self.__get_info_stats_population())
        col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for row in titles:
            print("".join(str(word).ljust(col_width) for word in row))
        print("\n")



    def __non_dominated_sorting_cache(self, population:Population) -> None:
        population_dict:Dict[int, Genome_NN] = population.population
        ranking:List[List[int]] = [[]]
        ranking_count:int = 0
        self.cache_dominance:Dict[Tuple[int, int], bool] = {}
        for id_1, genome_1 in population_dict.items():
            # 1 - Initialize
            genome_1.info["domination_set"] = set()
            genome_1.info["domination_count"] = 0

            # 2 - Compare with all other genomes
            for id_2, genome_2 in population_dict.items():
                if id_1 == id_2: continue # Skip if same genome

                # 2.1 - Compare fitnesses for domination
                if (id_1, id_2) in self.cache_dominance: # Check if already computed
                    if self.cache_dominance[(id_1, id_2)] == True:
                        genome_1.info["domination_set"].add(id_2)
                    elif self.cache_dominance[(id_2, id_1)] == True:
                        genome_1.info["domination_count"] += 1
                else:
                    if self.__dominates(genome_1, genome_2):
                        genome_1.info["domination_set"].add(id_2)

                    elif self.__dominates(genome_2, genome_1):
                        genome_1.info["domination_count"] += 1
                        

            # 3 - Check if genome is rank 0 (best rank)
            if genome_1.info["domination_count"] == 0:
                genome_1.info["rank"] = 0
                ranking[0].append(id_1)

        # 4 - Rank all genomes
        while len(ranking[ranking_count]) > 0:
            ranking.append([])
            for id_1 in ranking[ranking_count]:
                genome_1:Genome_NN = population_dict[id_1]
                for id_2 in genome_1.info["domination_set"]:
                    genome_2:Genome_NN = population_dict[id_2]
                    genome_2.info["domination_count"] -= 1
                    if genome_2.info["domination_count"] == 0:
                        genome_2.info["rank"] = ranking_count + 1
                        ranking[ranking_count + 1].append(id_2)
            ranking_count += 1

        # for genome in population_dict.values():
        #     print("id:", genome.id, "fitness:", genome.info["fitnesses"],"rank",genome.info["rank"])

