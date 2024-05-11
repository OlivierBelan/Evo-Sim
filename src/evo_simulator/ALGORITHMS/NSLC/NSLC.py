from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN, Genome_Classic, Genome
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id, get_new_population_id
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Reproduction import Reproduction_NN, Reproduction_Classic
from evo_simulator.ALGORITHMS.NEAT.Mutation import Mutation_NEAT
from GENERAL.Mutation_NN import Mutation

from evo_simulator.GENERAL.Archive import Archive
from typing import Dict, Any, Tuple
import numpy as np
import numba as nb
import random
import math
import time


class NSLC(Algorithm):
    def __init__(self, config_path_file:str, name:str = "NSLC") -> None:
        Algorithm.__init__(self, config_path_file, name)
        # Initialize configs
        self.config_path_file:str = config_path_file
        self.config_NSLC:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NSLC", "Archive", "Genome_NN", "NEURO_EVOLUTION", "Reproduction", "Mutation"])
        self.verbose:bool = True if self.config_NSLC["NSLC"]["verbose"] == "True" else False
        self.optimization_type:str = self.config_NSLC["NEURO_EVOLUTION"]["optimization_type"] # maximize, minimize, closest_to_zero

        self.pop_size:int = int(self.config_NSLC["NSLC"]["pop_size"])
        self.neighbour_size:int = np.floor(self.pop_size * float(self.config_NSLC["NSLC"]["neighbourhood_ratio"])).astype(int)
        self.is_local_competition:bool = True if self.config_NSLC["NSLC"]["is_local_competition"] == "True" else False
        self.update_population_type:str = self.config_NSLC["NSLC"]["update_population_type"]
        if self.update_population_type not in ["archive_best", "archive_random", "population"]:
            raise Exception("update_population_type must be 'archive_best', 'archive_random' or 'population'")
        
        self.is_first_generation:bool = True

        self.population_manager:Population = Population(get_new_population_id(), config_path_file)
        self.reproduction:Reproduction_NN = Reproduction_NN(config_path_file, self.attributes_manager)


        self.mutation_type:str = self.config_NSLC["Mutation"]["mutation_type"] # classic, topology, sbx
        self.mutation_neat:Mutation_NEAT = Mutation_NEAT(config_path_file, self.attributes_manager)
        self.mutation_ga:Mutation = Mutation(config_path_file, self.attributes_manager)
        self.is_sbx:str = True if self.config_NSLC["Reproduction"]["is_sbx"] == "True" else False

        self.archive:Archive = Archive(
                                        config_section_name="Archive",
                                        config_path_file=config_path_file,
                                        genome_builder_function=self.__build_genome_function_nn
                                        )

        # self.__init_rastrigin(config_path_file) # If we want to run the Rastrigin function (FOR TESTING RASTRIGIN FUNCTION & DEBUGGING)


    def first_generation(self, population_manager:Population) -> None:
        self.is_first_generation = False
        population:Dict[int, Genome_NN] = population_manager.population

        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_NSLC["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=True)
            population[new_genome.id] = new_genome
    
    def __build_genome_function_nn(self) -> Genome_NN:
        new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_NSLC["Genome_NN"], self.attributes_manager)
        new_genome.nn.set_arbitrary_parameters(is_random=True)
        return new_genome


    def run(self, global_population:Population) -> Population:
        # return self.run_rastrigin(global_population) ## FOR TESTING RASTRIGIN FUNCTION & DEBUGGING

        # 0 - First generation - Build population
        self.population_manager.population = global_population.population
        if self.is_first_generation == True:
            self.first_generation(self.population_manager)
            return self.population_manager


        # 1 - Update archive
        start_time = time.time()
        self.__update_archive(self.population_manager, "fitness")
        print("NSLC: Update archive time:", time.time() - start_time, "s")

        # 2 - Novelty and local competition
        start_time = time.time()
        self.__novelty_and_local_competition(self.population_manager)
        print("NSLC: Novelty and local competition time:", time.time() - start_time, "s")

        # 3 - Reproduction
        start_time = time.time()
        self.__reproduction(self.population_manager, self.pop_size, "novelty_competition_score", optimization_type=self.optimization_type)
        print("NSLC: Reproduction time:", time.time() - start_time, "s")

        # 5 - Mutation
        start_time = time.time()
        self.__mutation(self.population_manager) # A REVOIR
        print("NSLC: Mutation time:", time.time() - start_time, "s")


        # 8 - Update population
        global_population.population = self.population_manager.population
        self.population_best:Population = self.archive.get_best_population()

        return global_population

    

    def __reproduction(self, population:Population, size:int, criteria:str, optimization_type:bool=True) -> None:
        # 1- Reproduction
        population.population = self.reproduction.reproduction(population=population, size=size, criteria=criteria, optimization_type=optimization_type)

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
            raise Exception("Mutation type not found, can be classic or tpology or sbx")

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

    def __update_archive(self, population:Population, update_criteria:str) -> None:
        self.archive.update(population, update_criteria)

    def __novelty_and_local_competition(self, population:Population) -> Population:

        # 1 - Build description_population, description_archive(current pop + archive) and fitness matrix
        description_population:np.ndarray = np.array([genome.info["description"] for genome in population.population.values()], dtype=np.float32)
        fitness_archive, description_archive, centroid_archive = self.archive.get_niches_info()
        description_archive = centroid_archive

        # 2 - Compute distance matrix
        if self.update_population_type == "archive_best" or self.update_population_type == "archive_random":
            distances_matrix:np.ndarray = self.euclidean_distance_matrix(description_archive, description_archive)
        else:
            distances_matrix:np.ndarray = self.euclidean_distance_matrix(description_population, description_archive)

        # 3 - Compute Novelty score and Local competition score (Higher score is better)
        if self.is_local_competition == True:
            competition_score, novelty_score = self.__compute_novelty_and_local_competition_score(distances_matrix, fitness_archive, self.neighbour_size, is_normalized=True, optimization_type=self.optimization_type)
        else:
            competition_score, novelty_score = self.__compute_novelty_and_global_competition_score(distances_matrix, fitness_archive, self.neighbour_size, is_normalized=True, optimization_type=self.optimization_type)

        # 5 - Get the n_pop_size indexes of the best combinations (novelty + competition)
        novelty_competition_score:np.ndarray = (novelty_score + competition_score) # (Higher is better)
        if self.update_population_type == "archive_best":
            novelty_competition_score_sorted_indexes:np.ndarray = np.argsort(novelty_competition_score)[::-1][:self.pop_size] # used if we want to build new population from best archive
        elif self.update_population_type == "archive_random":
            novelty_competition_score_sorted_indexes:np.ndarray = np.argsort(novelty_competition_score) # used if we want to build new population from random archive
            np.random.shuffle(novelty_competition_score_sorted_indexes)
            novelty_competition_score_sorted_indexes = novelty_competition_score_sorted_indexes[:self.pop_size]

        # 6 - Update Archive with Novelty and competition score and Build new population
        # novelty_score:List[float] = novelty_score.tolist() # cause its faster to iterate on list than on np.ndarray (in python mode)
        # competition_score:List[float] = competition_score.tolist() # cause its faster to iterate on list than on np.ndarray (in python mode)
        if self.update_population_type == "archive_best" or self.update_population_type == "archive_random":
            new_population:Dict[int, Genome] = {}
            for i, niche in enumerate(self.archive.niches_info.values()):
                # # # 6.1 - Update Archive with Novelty and competition score
                # niche.novelty_score = novelty_score[i]
                # niche.competition_score = competition_score[i]
                # niche.info["novelty_competition_score"] = float(novelty_competition_score[i]) # (Higher is better)

                # 6.2 - Build new population from archive best
                if i in novelty_competition_score_sorted_indexes:
                    loaded_genome:Genome = self.archive.load_genome_from_archive(niche.centroid)
                    loaded_genome.info["novelty_competition_score"] = float(novelty_competition_score[i]) # (Higher is better)
                    new_population[loaded_genome.id] = loaded_genome
                    if len(new_population) == self.pop_size: break
            population.population = new_population
            return population

        # 6.4 - Build (update) new population from current population
        elif self.update_population_type == "population":
            for i, genome in enumerate(population.population.values()):
                genome.info["novelty_competition_score"] = float(novelty_competition_score[i]) # (Higher is better)
            return population
        return population

    @staticmethod
    @nb.jit(nopython=True, cache=True, fastmath=True, nogil=True, parallel=True)
    def __compute_novelty_and_local_competition_score(distance_matrix:np.ndarray, fitness_archive:np.ndarray, neighbour_size_0:int, is_normalized:bool=True, optimization_type:str = "maximize") -> np.ndarray:

        # 1 - Compute Novelty and Competition score
        novelty_vector_neighbourhood:np.ndarray = np.empty((distance_matrix.shape[0]), dtype=np.float32)
        if neighbour_size_0 >= distance_matrix.shape[0]:
            neighbour_size:int = distance_matrix.shape[0] - 1
        else:
            neighbour_size:int = neighbour_size_0
        neighbour_size_plus_one:int = neighbour_size + 1
        fitness_matrix:np.ndarray = np.empty((distance_matrix.shape[0], neighbour_size_plus_one), dtype=np.float32)
        for i in nb.prange(distance_matrix.shape[0]):
            indexes_sorted_distance:np.ndarray = np.argsort(distance_matrix[i])
            # 1.1 - Compute Novelty score
            novelty_vector_neighbourhood[i] = np.mean(distance_matrix[i, indexes_sorted_distance[1:neighbour_size_plus_one]]) # +1 to remove itself distance
            # 1.2 - Compute the local competition score
            fitness_matrix[i] = np.argsort(fitness_archive[indexes_sorted_distance][:neighbour_size_plus_one])
        # 1.3 - Get local competition score -> get position 0 -> ranked local competition with neighbourhood
        local_competition_vector:np.ndarray = np.where(fitness_matrix == 0)[1].astype(np.float32) # get position 0 -> ranked local competition with neighbourhood

        # 2 - Normalize the Novelty score (if needed)
        if is_normalized == True:
            max_novelty:float = np.max(novelty_vector_neighbourhood)
            min_novelty:float = np.min(novelty_vector_neighbourhood)
            novelty_vector_neighbourhood = (novelty_vector_neighbourhood - min_novelty) / (max_novelty - min_novelty)

        # 3 - Check if the type of best fitness is higher or lower (cause it will change the competition score)
        if optimization_type in ["minimize", "closest_to_zero"]:
            local_competition_vector = -local_competition_vector + neighbour_size

        # 4 - Normalize the local competition score (if needed)
        if is_normalized == True: 
            # print("local_competition_vector / neighbour_size\n", local_competition_vector / neighbour_size)
            return (local_competition_vector / neighbour_size), novelty_vector_neighbourhood
        else: 
            return local_competition_vector, novelty_vector_neighbourhood

    @staticmethod
    @nb.jit(nopython=True, cache=True, fastmath=True, nogil=True, parallel=True)
    def __compute_novelty_and_global_competition_score(distance_matrix:np.ndarray, fitness_archive:np.ndarray, neighbour_size_0:int, is_normalized:bool=True, optimization_type:str = "maximize") -> np.ndarray:

        # 1 - Compute Novelty and Competition score
        novelty_vector_neighbourhood:np.ndarray = np.empty((distance_matrix.shape[0]), dtype=np.float32)
        if neighbour_size_0 >= distance_matrix.shape[0]:
            neighbour_size:int = distance_matrix.shape[0] - 1
        else:
            neighbour_size:int = neighbour_size_0
        neighbour_size_plus_one:int = neighbour_size + 1
        fitness_matrix:np.ndarray = np.empty((distance_matrix.shape[0], distance_matrix.shape[1]), dtype=np.float32)
        for i in nb.prange(distance_matrix.shape[0]):
            indexes_sorted_distance:np.ndarray = np.argsort(distance_matrix[i])
            # 1.1 - Compute Novelty score
            novelty_vector_neighbourhood[i] = np.mean(distance_matrix[i, indexes_sorted_distance[1:neighbour_size_plus_one]])
            # 1.2 - Compute the local competition score
            fitness_matrix[i] = np.argsort(fitness_archive[indexes_sorted_distance])
        # 1.3 - Get local competition score -> get position 0 -> ranked local competition with neighbourhood
        global_competition_vector:np.ndarray = np.where(fitness_matrix == 0)[1].astype(np.float32) # get position 0 -> ranked local competition with neighbourhood

        # 2 - Normalize the Novelty score (if needed)
        if is_normalized == True:
            max_novelty:float = np.max(novelty_vector_neighbourhood)
            min_novelty:float = np.min(novelty_vector_neighbourhood)
            novelty_vector_neighbourhood = (novelty_vector_neighbourhood - min_novelty) / (max_novelty - min_novelty)

        # 3 - Check if the type of best fitness is higher or lower (cause it will change the competition score)
        if optimization_type in ["minimize", "closest_to_zero"]:
            global_competition_vector = -global_competition_vector + (fitness_matrix.shape[1]-1)

        # 4 - Normalize the local competition score (if needed)
        if is_normalized == True: 
            return (global_competition_vector / (fitness_matrix.shape[1]-1)), novelty_vector_neighbourhood
        else: 
            return global_competition_vector, novelty_vector_neighbourhood


    @staticmethod
    @nb.jit(nopython=True, cache=True, fastmath=True, nogil=True, parallel=True)
    def __novelty_score(distance_matrix:np.ndarray, neighbour_size:int, is_normalized:bool=True) -> np.ndarray:
        # distances_matrix_sorted_without_itself_distance:np.ndarray = np.sort(distance_matrix, axis=1)[:, 1:] # remove itself distance   

        # novelty_vector_neighbourhood:np.ndarray = np.mean(distances_matrix_sorted_without_itself_distance[:, :neighbour_size], axis=1)
        # if is_normalized == True:        
        #     max_novelty:float = np.max(novelty_vector_neighbourhood)
        #     min_novelty:float = np.min(novelty_vector_neighbourhood)
        #     novelty_vector_normalized_neighbourhood_2:np.ndarray = (novelty_vector_neighbourhood - min_novelty) / (max_novelty - min_novelty)
        #     # return novelty_vector_normalized_neighbourhood
        #     print("novelty_vector_normalized_neighbourhood", novelty_vector_normalized_neighbourhood_2)
        # elif is_normalized == False:
        #     return novelty_vector_neighbourhood

        novelty_vector_neighbourhood:np.ndarray = np.empty((distance_matrix.shape[0]), dtype=np.float32)
        for i in nb.prange(distance_matrix.shape[0]):
            novelty_vector_neighbourhood[i] = np.mean(np.sort(distance_matrix[i])[1:neighbour_size+1])
        if is_normalized == True:
            max_novelty:float = np.max(novelty_vector_neighbourhood)
            min_novelty:float = np.min(novelty_vector_neighbourhood)
            return (novelty_vector_neighbourhood - min_novelty) / (max_novelty - min_novelty)
        elif is_normalized == False:
            return novelty_vector_neighbourhood

    @staticmethod
    @nb.jit(nopython=True, cache=True, fastmath=True, nogil=True, parallel=True)
    def __local_competition_score(fitness_archive:np.ndarray, distance_matrix:np.ndarray, neighbour_size:float, is_normalized:bool=True, optimization_type:str = "maximize") -> np.ndarray:
        #### START FUL EXPLANATION ####

        # # 0 - Get distance matrix sorted indexes
        # distance_matrix_sorted_indexes:np.ndarray = np.argsort(distance_matrix, axis=1)
        # # 1 - Get all fitnesses sorted by distance in order to get a fitness matrix
        # # where each row is the fitness of the individual we want to compare (idx=0) + the fitness of the archive (idx=1:nb_archive)
        # fitness_matrix:np.ndarray = fitness_archive[distance_matrix_sorted_indexes]
        # # 2 - fitnesses neighbourh -> +1 to include itself + neighbour (for the rank competion)
        # fitness_matrix_neighbourhood:np.ndarray = fitness_matrix[:, :neighbour_size+1] 
        # # 3 - Get index sorted fitnesses neighbourh -> the position (index) of the number 0 tell us the rank of the fitness (from neighbourhood)
        # fitness_sorted_indexes_neighbourhood:np.ndarray = np.argsort(fitness_matrix_neighbourhood, axis=1)# 
        # # 4 - Get local competition score -> get position 0 -> ranked local competition with neighbourhood
        # local_competition_vector:np.ndarray = np.where(fitness_sorted_indexes_neighbourhood == 0)[1]
        # print("fitness_archive", fitness_archive)
        # print("fitness_matrix\n", fitness_matrix)
        # print("fitness_matrix_neighbourhood\n", fitness_matrix_neighbourhood)
        # print("fitness_sorted_indexes_neighbourhood\n", fitness_sorted_indexes_neighbourhood)
        # print("optimization_type:", optimization_type)
        # print("1 - local_competition_vector\n", local_competition_vector)

        #### END FUL EXPLANATION ####
        # fitness_matrix_2:np.ndarray = fitness_archive[np.argsort(distance_matrix, axis=1)]
        # local_competition_vector_2:np.ndarray = np.where(np.argsort(fitness_matrix_2[:, :neighbour_size+1], axis=1) == 0)[1]# get position 0 -> ranked local competition with neighbourhood
        # print("original local_competition_vector\n", local_competition_vector_2)

        # 1 - Get all fitnesses sorted by distance in order to get a fitness matrix
        # where each row is the fitness of the individual we want to compare (idx=0) + the fitness of the archive (idx=1:nb_archive)
        # fitness_matrix:np.ndarray = fitness_archive[np.argsort(distance_matrix, axis=1)]
        neighbour_size_plus_one:int = neighbour_size + 1
        fitness_matrix:np.ndarray = np.empty((distance_matrix.shape[0], neighbour_size_plus_one), dtype=np.float64)
        for i in nb.prange(distance_matrix.shape[0]):
            fitness_matrix[i] = fitness_archive[np.argsort(distance_matrix[i])][:neighbour_size_plus_one]


        # 2 - Compute the local competition score (look up to have the full explanation)
        # local_competition_vector:np.ndarray = np.where(np.argsort(fitness_matrix[:, :neighbour_size+1], axis=1) == 0)[1]# get position 0 -> ranked local competition with neighbourhood
        fitness_sorted_indexes_neighbourhood:np.ndarray = np.empty((fitness_matrix.shape[0], neighbour_size_plus_one), dtype=np.int32)
        for i in nb.prange(fitness_matrix.shape[0]):
            fitness_sorted_indexes_neighbourhood[i] = np.argsort(fitness_matrix[i])
        local_competition_vector:np.ndarray = np.where(fitness_sorted_indexes_neighbourhood == 0)[1].astype(np.float32) # get position 0 -> ranked local competition with neighbourhood
        # print("local_competition_vector\n", local_competition_vector)
        # exit()

        # 3 - Check if the type of best fitness is higher or lower (cause it will change the competition score)
        if optimization_type in ["minimize", "closest_to_zero"]:
            local_competition_vector = -local_competition_vector + neighbour_size

        # 4 - Normalize the local competition score (if needed)
        if is_normalized == True: 
            # print("local_competition_vector / neighbour_size\n", local_competition_vector / neighbour_size)
            return local_competition_vector / neighbour_size
        else: 
            return local_competition_vector

    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __global_competition_score(fitness_archive:np.ndarray, distance_matrix:np.ndarray, is_normalized:bool=True, optimization_type:str="maximize") -> None:
        distance_matrix_sorted_indexes:np.ndarray = np.argsort(distance_matrix, axis=1)

        # fitness_matrix:np.ndarray = fitness_archive[distance_matrix_sorted_indexes] # fitnesses -> each row is the fitness of one individual (idx=0) + the fitness of the archive (idx=1:nb_archive)
        # fitness_sorted_indexes:np.ndarray = np.argsort(fitness_matrix, axis=1) # the position of the number 0 tell us the rank of the fitness (from all archive + population)
        # global_competition_vector:np.ndarray = np.where(fitness_sorted_indexes == 0)[1]# get position 0 -> ranked local competition with neighbourhood
        # print("fitness_archive", fitness_archive)
        # print("fitness_matrix\n", fitness_matrix)
        # print("fitness_sorted_indexes\n", fitness_sorted_indexes)
        # print("optimization_type:", optimization_type)
        # print("1 - global_competition_vector\n", global_competition_vector)
        # print("2 - global_competition_vector_normalized\n", global_competition_vector / (fitness_matrix.shape[1]-1))
        # print("fitness_matrix.shape[1]", fitness_matrix.shape[1])

        fitness_matrix:np.ndarray = fitness_archive[distance_matrix_sorted_indexes] # fitnesses -> each row is the fitness of one individual (idx=0) + the fitness of the archive (idx=1:nb_archive)
        global_competition_vector:np.ndarray = np.where(np.argsort(fitness_matrix, axis=1) == 0)[1] # get position 0 -> ranked local competition with neighbourhood

        if optimization_type in ["minimize", "closest_to_zero"]: 
            global_competition_vector = -global_competition_vector + (fitness_matrix.shape[1]-1)
        
        if is_normalized == True: 
            return global_competition_vector / (fitness_matrix.shape[1]-1)
        else: 
            return global_competition_vector


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def euclidean_distance_matrix(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
        diff = matrix_1[:, np.newaxis, :] - matrix_2[np.newaxis, :, :]
        distances = np.sqrt((diff**2).sum(axis=-1))
        return distances


    # START test rastrigin
    def first_generation_rastrigin(self, population_manager:Population) -> None:
        if self.is_first_generation == True:
            self.ajust_population_rastrigin(population_manager)
            self.__test_rastrigin(self.population_manager)
            self.__update_archive(self.population_manager, "fitness")
            self.__novelty_and_local_competition(self.population_manager)
            self.is_first_generation = False

    def ajust_population_rastrigin(self, population:Population) -> None:
        # population:Population = Population(get_new_population_id(), self.config_path_file)
        population_dict:Dict[int, Genome_Classic] = population.population
        while len(population_dict) < self.pop_size:
            new_genome:Genome_Classic = self.__build_genome_function_classic()
            population_dict[new_genome.id] = new_genome
        return population


    def __init_rastrigin(self, config_path_file:str) -> None:
        # test rastrigin start
        self.reproduction:Reproduction_Classic = Reproduction_Classic(config_path_file, self.attributes_manager)
        self.archive:Archive = Archive(
                                        config_section_name="Archive",
                                        config_path_file=config_path_file,
                                        genome_builder_function=self.__build_genome_function_classic, 
                                        )
        self.mutation:Mutation = Mutation(config_path_file, self.attributes_manager)
        self.config_NSLC.update(TOOLS.config_function(config_path_file, ["Genome_Classic"]))
        self.parameter_size:int = int(self.config_NSLC["Genome_Classic"]["parameter_size"])
        # test rastrigin end

    def run_rastrigin(self, global_population:Population) -> Population:
        # 0 - First generation
        self.first_generation_rastrigin(self.population_manager)
        
        # 1 - Reproduction
        self.__reproduction_rastrigin(self.population_manager, self.pop_size, "novelty_competition_score", optimization_type="maximize")

        # 2 - Mutation
        # self.__mutation_rastrigin(self.population_manager)

        # 3 - Evaluation
        self.__test_rastrigin(self.population_manager)

        # 5 - Update archive
        # self.archive.update(self.population_manager)
        self.__update_archive(self.population_manager, "fitness")

        # 6 - Novelty and local competition
        # start_time = time.time()
        self.__novelty_and_local_competition(self.population_manager)
        # print("NSLC: Novelty and local competition time:", time.time() - start_time, "s")


        # 7 - Update population
        global_population.population = self.archive.get_best_population().population

        return global_population

    def __test_rastrigin(self, population_manager:Population) -> None:
        population_dict:Dict[int, Genome_Classic] = population_manager.population
        for genome in population_dict.values():
            genome.fitness.score, genome.info["description"] = self.__rastrigin_NSLC(genome.parameter)

    def __rastrigin_NSLC(self, xx:np.ndarray) -> Tuple[float, np.ndarray]:
        x = xx * 10 - 5 # scaling to [-5, 5]
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f, np.array([xx[0], xx[1]])

    def __build_genome_function_classic(self) -> Genome_Classic:
        new_genome:Genome_Classic = Genome_Classic(get_new_genome_id(), self.config_NSLC["Genome_Classic"], self.attributes_manager)
        new_genome.info["is_elite"] = False
        # new_genome.parameter = self.mutation.attributes.epsilon_mu_sigma_jit(new_genome.parameter, mu_parameter=0, sigma_paramater=1, min=-1, max=1, mu_bias=0, sigma_coef=1)
        new_genome.parameter:np.ndarray = np.random.rand(self.parameter_size)
        return new_genome

    def __reproduction_rastrigin(self, population:Population, size:int, criteria:str, optimization_type:str="maximize") -> None:
        population.population = self.reproduction.reproduction(population=population, size=size, criteria=criteria, optimization_type=optimization_type)

    def __mutation_rastrigin(self, population:Population) -> None:
        self.__mutation_classic(population)
    # END test rastrigin
    