from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Population import Population_NN as Population
from evo_simulator.GENERAL.Distance import Distance
import evo_simulator.GENERAL.Globals as global_parameters
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES_algorithm import CMA_ES_algorithm
from typing import Dict, Any, List, Callable, Tuple
import numpy as np
import time
import jax
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD

from evosax import FitnessShaper
import math
from functools import reduce
import copy
from operator import mul

from pymoo.core.algorithm import Algorithm as pymoo_Algorithm
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem
from pymoo.problems.many import DTLZ1



class Algo_pymoo(Algorithm, Problem):
    def __init__(self, config_path_file:str, name:str = "algo_jax", pymoo_name_algo:str="NSGA2") -> None:
        Algorithm.__init__(self, config_path_file, name)


        # Initialize configs
        self.config_es:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["PYMOO", "Genome_NN","NEURO_EVOLUTION"])
        self.pop_size:int = int(self.config_es["PYMOO"]["pop_size"])
        self.is_first_generation:bool = True
        self.distance:Distance = Distance(config_path_file)
        self.genome_core:Genome_NN = Genome_NN(-1, self.config_es["Genome_NN"], self.attributes_manager)
        self.verbose:bool = True if self.config_es["PYMOO"]["verbose"] == "True" else False
        self.optimization_type:str = self.config_es["NEURO_EVOLUTION"]["optimization_type"]
        self.pymoo_name_algo:str = pymoo_name_algo
        
        
        # Initialize es_algorithms
        synapse_parameters_size:int = self.genome_core.nn.synapses_actives_indexes[0].size
        self.nb_generations:int = int(self.config_es["PYMOO"]["nb_generations"])
        self.nb_fitness:int = int(self.config_es["PYMOO"]["nb_fitness"])
        xl_array:np.ndarray = np.full(synapse_parameters_size, self.attributes_manager.min_parameters["weight"])
        xu_array:np.ndarray = np.full(synapse_parameters_size, self.attributes_manager.max_parameters["weight"])
        Problem.__init__(self, n_var=synapse_parameters_size, n_obj=self.nb_fitness, xl=xl_array, xu=xu_array)

        self.stop_criteria:Tuple = ("n_gen", self.nb_generations)

        if self.pymoo_name_algo == "NSGA2":
            self.algorithm:NSGA2 = NSGA2(pop_size=self.pop_size)

        elif self.pymoo_name_algo == "NSGA3":
            ref_dir = get_reference_directions("energy", self.nb_fitness, 90, seed=np.random.randint(0, 1000))
            # ref_dir =get_reference_directions("das-dennis", self.nb_fitness, n_partitions=12)
            self.algorithm:NSGA3 = NSGA3(pop_size=self.pop_size, ref_dirs=ref_dir)

        elif self.pymoo_name_algo == "MOEAD":
            n_neighbors:int = int(self.config_es["PYMOO"]["n_neighbors"])
            prob_neighbor_mating:float = float(self.config_es["PYMOO"]["prob_neighbor_mating"])
            ref_dir = get_reference_directions("energy", self.nb_fitness, 90, seed=np.random.randint(0, 1000))
            self.algorithm = MOEAD(ref_dir, n_neighbors=n_neighbors, prob_neighbor_mating=prob_neighbor_mating)
        else:
            raise Exception("Error: The algorithm name is not valid ->", self.pymoo_name_algo, "only NSGA2, NSGA3 and MOEAD are valid")

        # self.dtlz1:DTLZ1 = get_problem("dtlz1") # for testing


    def __run_pymoo(self, algorithm:pymoo_Algorithm, termination=None, copy_algorithm=True, copy_termination=True, **kwargs) -> None:

        # create a copy of the algorithm object to ensure no side-effects
        if copy_algorithm:
            algorithm = copy.deepcopy(algorithm)

        # initialize the algorithm object given a problem - if not set already
        if algorithm.problem is None:
            if termination is not None:

                if copy_termination:
                    termination = copy.deepcopy(termination)

                kwargs["termination"] = termination

            algorithm.setup(self, **kwargs)
            # algorithm.setup(self.dtlz1, **kwargs) # for testing

        # actually execute the algorithm
        res = algorithm.run()

        # store the deep copied algorithm in the result object
        res.algorithm = algorithm

        return res


    def _evaluate(self, designs, out, *args, **kwargs):
        self.fitness_shape = designs.shape[0]

        # 0 - Update population parameters
        self.__update_population_parameter(self.population_manager, designs)

        # 1 - Evaluation
        # self.__evalutation(self.population_manager, self.global_evaluation_function)
        self.test_kursawe(self.population_manager) # for testing
        # # self.test_dtlz1(self.population_manager) # for testing

        # 2 - Update
        out["F"] = self.__update_by_fitness(self.population_manager, designs)
        

        # For testing
        # self.test_dtlz1(self.population_manager)
        # print("self.n_obj", self.n_obj)
        # exit()
        # X_, X_M = designs[:, :self.n_obj - 1], designs[:, self.n_obj - 1:]
        # g = self.dtlz1.g1(X_M)
        # out["F"] = self.dtlz1.obj_func(X_, g) # for testing
        # End testing



    def run(self, global_population:Population, evaluation_function:Callable) -> Population:
        self.global_evaluation_function:Callable = evaluation_function
        self.population_manager = global_population
        self.first_generation(self.population_manager, evaluation_function)

        # 0 - Start
        res = self.__run_pymoo(self.algorithm, self.stop_criteria, verbose=False)
        self.__update_population_fitnesses(self.population_manager, res.F)

        # 1 - Update population
        global_population.population = self.population_manager.population

        Scatter().add(res.F).show()
        # self.plot(global_population)

        return global_population

            
    def first_generation(self, population_manager:Population, evaluation_function:Callable) -> None:
        if  population_manager.is_first_generation == True:
            start_time = time.time()
            self.is_first_generation = False
            self.ajust_population(population_manager)
            population_manager.is_first_generation = False
            # self.__evalutation(population_manager, evaluation_function)
            print(self.name+": First generation time:", time.time() - start_time, "s")

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            # NN VERSION
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_es["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=False, weight_random=True)
            population[new_genome.id] = new_genome

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
        population_dict:Dict[int, Genome_NN] = population.population
        for genome in population_dict.values():
            genome.info["fitnesses"] = np.array(self.kursawe(genome.nn.parameters["weight"]), dtype=np.float32)

    def test_dtlz1(self, population:Population):
        population_dict:Dict[int, Genome_NN] = population.population
        for genome in population_dict.values():
            genome.info["fitnesses"] = np.array(self.dtlz1(genome.nn.parameters["weight"]), dtype=np.float32)

    def dtlz1(self, individual, obj=3):

        r"""DTLZ1 multiobjective function. It returns a tuple of *obj* values.
        The individual must have at least *obj* elements.
        From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
        Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

        :math:`g(\mathbf{x}_m) = 100\left(|\mathbf{x}_m| + \sum_{x_i \in \mathbf{x}_m}\left((x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5))\right)\right)`

        :math:`f_{\text{DTLZ1}1}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m)) \prod_{i=1}^{m-1}x_i`

        :math:`f_{\text{DTLZ1}2}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m)) (1-x_{m-1}) \prod_{i=1}^{m-2}x_i`

        :math:`\ldots`

        :math:`f_{\text{DTLZ1}m-1}(\mathbf{x}) = \frac{1}{2} (1 + g(\mathbf{x}_m)) (1 - x_2) x_1`

        :math:`f_{\text{DTLZ1}m}(\mathbf{x}) = \frac{1}{2} (1 - x_1)(1 + g(\mathbf{x}_m))`

        Where :math:`m` is the number of objectives and :math:`\mathbf{x}_m` is a
        vector of the remaining attributes :math:`[x_m~\ldots~x_n]` of the
        individual in :math:`n > m` dimensions.

        """
        g = 100 * (len(individual[obj - 1:]) + sum((xi - 0.5)**2 - math.cos(20 * math.pi * (xi - 0.5)) for xi in individual[obj - 1:]))
        f = [0.5 * reduce(mul, individual[:obj - 1], 1) * (1 + g)]
        f.extend(0.5 * reduce(mul, individual[:m], 1) * (1 - individual[m]) * (1 + g) for m in reversed(range(obj - 1)))
        return f


    def __update_population_parameter(self, population_manager:Population, paramaters:np.ndarray) -> None:
        # NN VERSION
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        # 2 - Update parameters in the population
        for index, genome in enumerate(genomes_dict.values()):
            if index >= len(paramaters):
                index = 0
            nn:NN = genome.nn

            # 2.2 Update synapse parameters
            # nn.parameters["weight"][nn.synapses_actives_indexes] = np.array(paramaters[index])
            nn.parameters["weight"] = np.array(paramaters[index]) # for testing


    def __update_by_fitness(self, population_manager:Population, parameters:np.ndarray) -> np.ndarray:
        self.population_manager.update_info()
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        fitnesses:List[int] = []
        for genome in genomes_dict.values():
            # fitnesses.append(genome.fitness.score)
            fitnesses.append(genome.info["fitnesses"]) # for testing

        fitnesses:np.ndarray = np.array(fitnesses)
        fit_shaper = FitnessShaper(
                        # centered_rank=True,
                        # z_score=False,
                        # w_decay=0.1,
                        maximize=True if self.optimization_type == "maximize" else False,
                        )
        
        return fit_shaper.apply(parameters, fitnesses)[:self.fitness_shape]

    def __update_population_fitnesses(self, population_manager:Population, fitnesses:np.ndarray) -> None:
        # NN VERSION
        genomes_dict:Dict[int, Genome_NN] = population_manager.population
        # 2 - Update fitnesses in the population
        for index, genome in enumerate(genomes_dict.values()):
            if index >= len(fitnesses):
                index = 0

            # 2.2 Update synapse fitnesses
            genome.info["fitnesses"] = np.array(fitnesses[index])


    def __evalutation(self, population_manager:Population, evaluation_function:Callable) -> None:
        if evaluation_function is not None:
            evaluation_function(population_manager.population)
        else:
            self.evaluation_function(population_manager.population)


    def plot(self, population:Population) -> None:
        import matplotlib.pyplot as plt
        
        population_dict:Dict[int, Genome_NN] = population.population
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
