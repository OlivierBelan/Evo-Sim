from evo_simulator.GENERAL.Genome import Genome
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.SNN import SNN
from evo_simulator.GENERAL.NN import NN
from typing import Dict, Any, List
import numpy as np
import TOOLS

class Population:
    def __init__(self, id:int, config_path_file:str, extra_info:Dict[Any, Any] = {}) -> None:
        # General
        self.id:int = id
        self.population:Dict[int, Genome] = {}
        self.best_genome:Genome = None
        self.extra_info:Dict[str, Any] = extra_info
        self.config_path_file:str = config_path_file
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NEURO_EVOLUTION"])
        self.optimization_type:str = self.config["NEURO_EVOLUTION"]["optimization_type"] # maximize, minimize, closest_to_zero

        # Parameters
        self.age: int = 0
        self.stagnation: int = 0
        self.reproduction_size: int = 1
        self.size = 0

        # Fitness
        self.fitness: Fitness = Fitness()

    def update(self, population:"Population" | List["Population"]) -> None:
        if isinstance(population, List):
            for pop in population:
                self.population.update(pop.population)
        else:
            self.population.update(population.population)

    def replace(self, population:"Population") -> None:
        self.population = population.population

    def update_info(self, optimization_type:str = None) -> None:
        self.reset()
        if optimization_type is not None:
            if optimization_type not in ["maximize", "minimize", "closest_to_zero"]: raise ValueError("optimization_type must be 'maximize', 'minimize' or 'closest_to_zero'")
            self.optimization_type = optimization_type

        # 0 - Get current fitness and topology info of the population
        for index, genome in enumerate(self.population.values()):
            # 1 - Fitness  
            genome_score:float = genome.fitness.score if self.optimization_type != "closest_to_zero" else abs(genome.fitness.score)
            # 1.1 - Fitness max
            if index == 0 or genome_score > self.fitness.max:
                self.fitness.max = genome_score
                if self.optimization_type == "maximize":
                    self.best_genome = genome            
            # 1.2 - Fitness min
            if index == 0 or genome_score < self.fitness.min:
                self.fitness.min = genome_score
                if self.optimization_type in ["minimize", "closest_to_zero"]:
                    self.best_genome = genome
            # 1.3 - Fitness mean
            self.fitness.mean += genome_score

        # 3 - Update mean
        self.size:int = len(self.population)
        if self.size > 0:
            self.fitness.mean /= self.size

        # 4 - Fitness best score and history
        self.fitness.score = self.fitness.max if self.optimization_type == "maximize" else self.fitness.min
        self.fitness.history_best.append(self.fitness.score)
        self.fitness.history_mean.append(self.fitness.mean)
    
    def reset(self):
        self.fitness.reset()


class Population_SNN(Population):
    def __init__(self, id:int, config_path_file:str, extra_info:Dict[Any, Any] = {}) -> None:

        Population.__init__(self, id, config_path_file, extra_info)

        # Topology
        self.neurons_max:int = 0
        self.neurons_mean:int = 0
        self.neurons_min:int = 0

        self.synapses_max:int = 0
        self.synapses_mean:int = 0
        self.synapses_min:int = 0

    def update_info(self, optimization_type:str = None) -> None:
        self.reset()
        if optimization_type is not None:
            if optimization_type not in ["maximize", "minimize", "closest_to_zero"]: raise ValueError("optimization_type must be 'maximize', 'minimize' or 'closest_to_zero'")
            self.optimization_type = optimization_type


        # 0 - Get current fitness and topology info of the population
        for index, genome in enumerate(self.population.values()):
            snn:SNN = genome.snn

            # 1 - Fitness  
            genome_score:float = genome.fitness.score if self.optimization_type != "closest_to_zero" else abs(genome.fitness.score)
            # 1.1 - Fitness max
            if index == 0 or genome_score > self.fitness.max:
                self.fitness.max = genome_score
                if self.optimization_type == "maximize":
                    self.best_genome = genome            
            # 1.2 - Fitness min
            if index == 0 or genome_score < self.fitness.min:
                self.fitness.min = genome_score
                if self.optimization_type in ["minimize", "closest_to_zero"]:
                    self.best_genome = genome
            # 1.3 - Fitness mean
            self.fitness.mean += genome_score


            # 2 - Topology
            # 2.1 - Neurons
            nb_neurons:int = len(snn.hiddens["neurons_indexes_active"])
            if nb_neurons > self.neurons_max:
                self.neurons_max = nb_neurons
            if nb_neurons < self.neurons_min:
                self.neurons_min = nb_neurons
            self.neurons_mean += nb_neurons

            # 2.2 - Synapses
            nb_synapses:int = len(snn.synapses_actives_indexes[0])
            if nb_synapses > self.synapses_max:
                self.synapses_max = nb_synapses
            if nb_synapses < self.synapses_min:
                self.synapses_min = nb_synapses
            self.synapses_mean += nb_synapses 

        # 3 - Update mean
        self.size:int = len(self.population)
        if self.size > 0:
            self.fitness.mean /= self.size
            self.neurons_mean /= self.size
            self.synapses_mean /= self.size

        # 4 - Fitness best score and history
        self.fitness.score = self.fitness.max if self.optimization_type == "maximize" else self.fitness.min
        self.fitness.history_best.append(self.fitness.score)
        self.fitness.history_mean.append(self.fitness.mean)
    
    def reset(self):
        self.fitness.reset()
        self.neurons_max:int = 0
        self.neurons_mean:int = 0
        self.neurons_min:int = 0

        self.synapses_max:int = 0
        self.synapses_mean:int = 0
        self.synapses_min:int = 0


class Population_NN(Population):
    def __init__(self, id:int, config_path_file:str, extra_info:Dict[Any, Any] = {}) -> None:

        Population.__init__(self, id, config_path_file, extra_info)

        # Topology
        self.neurons_max:int = 0
        self.neurons_mean:int = 0
        self.neurons_min:int = 0

        self.synapses_max:int = 0
        self.synapses_mean:int = 0
        self.synapses_min:int = 0

    def update_info(self, optimization_type:str = None) -> None:
        self.reset()
        if optimization_type is not None:
            if optimization_type not in ["maximize", "minimize", "closest_to_zero"]: raise ValueError("optimization_type must be 'maximize', 'minimize' or 'closest_to_zero'")
            self.optimization_type = optimization_type

        nb_fitness_mean:int = 0
        # 0 - Get current fitness and topology info of the population
        for index, genome in enumerate(self.population.values()):
            nn:NN = genome.nn

            # 1 - Fitness  
            genome_score:float = genome.fitness.score if self.optimization_type != "closest_to_zero" else abs(genome.fitness.score)
            # 1.1 - Fitness max
            if index == 0 or genome_score > self.fitness.max:
                self.fitness.max = genome_score
                if self.optimization_type == "maximize":
                    self.best_genome = genome            
            # 1.2 - Fitness min
            if index == 0 or genome_score < self.fitness.min:
                self.fitness.min = genome_score
                if self.optimization_type in ["minimize", "closest_to_zero"]:
                    self.best_genome = genome
            # 1.3 - Fitness mean
            if genome_score != -np.inf and genome_score != np.inf:
                self.fitness.mean += genome_score
                nb_fitness_mean += 1

            # 2 - Topology
            # 2.1 - Neurons
            nb_neurons:int = len(nn.hiddens["neurons_indexes_active"])
            if nb_neurons > self.neurons_max:
                self.neurons_max = nb_neurons
            if nb_neurons < self.neurons_min:
                self.neurons_min = nb_neurons
            self.neurons_mean += nb_neurons

            # 2.2 - Synapses
            nb_synapses:int = len(nn.synapses_actives_indexes[0])
            if nb_synapses > self.synapses_max:
                self.synapses_max = nb_synapses
            if nb_synapses < self.synapses_min:
                self.synapses_min = nb_synapses
            self.synapses_mean += nb_synapses 

        # 3 - Update mean
        self.size:int = len(self.population)
        if self.size > 0 and nb_fitness_mean > 0:
            self.fitness.mean /= nb_fitness_mean
            self.neurons_mean /= self.size
            self.synapses_mean /= self.size

        # 4 - Fitness best score and history
        self.fitness.score = self.fitness.max if self.optimization_type == "maximize" else self.fitness.min
        self.fitness.history_best.append(self.fitness.score)
        self.fitness.history_mean.append(self.fitness.mean)
    
    def reset(self):
        self.fitness.reset()
        self.neurons_max:int = 0
        self.neurons_mean:int = 0
        self.neurons_min:int = 0

        self.synapses_max:int = 0
        self.synapses_mean:int = 0
        self.synapses_min:int = 0
