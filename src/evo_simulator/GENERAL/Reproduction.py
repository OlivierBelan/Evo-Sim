from typing import Dict, Any, List, Tuple, Callable, Set, Union
from evo_simulator.GENERAL.Genome import Genome, Genome_Decoder, Genome_Classic, Genome_NN
from evo_simulator.GENERAL.NN import NN
from evo_simulator.GENERAL.Distance import Distance
from evo_simulator.GENERAL.Population import Population
import evo_simulator.TOOLS as TOOLS
from evo_simulator.GENERAL.Index_Manager import get_new_genome_id
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
from evo_simulator.GENERAL.Optimizer import sort_population
import random
import numpy as np
import numba as nb
import time


class Reproduction:
    def __init__(self, config_path_file:str, attributes:Attribute_Paramaters) -> None:
        # Initialize configs
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Reproduction", "NEURO_EVOLUTION"])
        self.attributes:Attribute_Paramaters = attributes

        # Initialize reproduction configs
        self.nb_elites_ratio:float = float(self.config["Reproduction"]["nb_elites_ratio"])
        self.prob_reproduction_random:float = float(self.config["Reproduction"]["prob_reproduction_random"])
        self.prob_reproduction_elites:float = 1 - self.prob_reproduction_random
        self.prob_reproduction_dominance:float = float(self.config["Reproduction"]["prob_reproduction_dominance"])
        self.keep_elites:bool = True if self.config["Reproduction"]["keep_elites"] == "True" else False
        self.parents_similarities_ratio:float = float(self.config["Reproduction"]["parents_similarities_ratio"])

        # Fitness configs
        self.optimization_type:str = self.config["NEURO_EVOLUTION"]["optimization_type"]

        # Selection
        self.selection_type:str = self.config["Reproduction"]["selection_type"]
        if self.selection_type == "tournament":
            self.tournament_size:int = int(self.config["Reproduction"]["tournament_size"])

        # SBX
        self.is_sbx:bool = True if self.config["Reproduction"]["is_sbx"] == "True" else False
        self.sbx_eta:float = float(self.config["Reproduction"]["sbx_eta"])

        # Reproduction function (must be defined in the child class)
        self.reproduction_function:Callable = None


    def reproduction(self, population:Population, size:int, criteria:str="fitness", replace:bool= True, optimization_type:str=None, get_parents:bool=False) -> Union[Dict[int, Genome], Tuple[Dict[int, Genome], Dict[int, Genome]]]:

        # 1 - Initialize
        if optimization_type is not None: self.optimization_type = optimization_type
        new_population:Dict[int, Genome] = {}
        parent_population:Dict[int, Genome] = {}
        current_population:Dict[int, Genome] = population.population

        # 2 - Elites selection
        elite_size:int = int(np.ceil(len(current_population) * self.nb_elites_ratio)) # Number of elites (rounded up to the nearest integer minimum 1)
        if elite_size >= size: elite_size:int = int(np.ceil(size * self.nb_elites_ratio))
   
        elites:np.ndarray = self.selection(current_population, elite_size, criteria)
        if self.keep_elites == True:
            new_population:Dict[int, Genome] = {elite:current_population[elite] for elite in elites}
            size:int = size - elite_size
        if size <= 0:
            if replace == True: population.population = new_population # keep this line for the sub populations (e.g. NEAT (specie))
            return new_population

        # 3 - Parents selection
        parents:Tuple[np.ndarray] = (elites, np.array(list(current_population.keys()), dtype=np.int32)) # (elites, random) (could be more than 2 parents)
        parents_probs:np.ndarray = np.array([self.prob_reproduction_elites, self.prob_reproduction_random],dtype=np.float32)
        parents:np.ndarray = self.parents_selection_jit(parents, parents_probs, size)
        parents_unique:np.ndarray = np.unique(parents)
        parent_population.update({parent_id:current_population[parent_id] for parent_id in parents_unique})
        # print("elite")
        # for genome in parent_population.values():
        #     print("genome id:", genome.id, " parameters:", genome.parameter)


        # 4 - Reproduction
        if self.reproduction_function is None: raise Exception("Reproduction: reproduction_function is not defined")
        self.reproduction_function(parents, parent_population, new_population)

        # 5 - Return
        if replace == True: population.population = new_population # keep this line for the sub populations (e.g. NEAT (specie))
        if get_parents == True: return new_population, parent_population
        return new_population

    def selection(self, population:Dict[int, Genome], size:int, criteria:str="fitness") -> np.ndarray:
        if self.selection_type == "best":
            return self.best_selection(population, size, criteria)
        elif self.selection_type == "tournament":
            return self.tournament_selection(population, self.tournament_size, size, criteria)
        else:
            raise Exception("Reproduction: selection_type must be best or tournament")

    def best_selection(self, population:Dict[int, Genome], size:int, criteria:str="fitness") -> np.ndarray:
        '''
        Select the elites of the population
        '''
        if criteria != "fitness" and type(population[next(iter(population))].info[criteria]) != float: raise Exception("Reproduction: criteria must be a float")
        elites = sort_population(population, self.optimization_type, criteria)
        # if criteria == "fitness":
        #     if self.optimization_type == "minimize":
        #         elites:List[Genome] = sorted(population.values(), key=lambda x: x.fitness.score, reverse=False)
        #     elif self.optimization_type == "maximize":
        #         elites:List[Genome] = sorted(population.values(), key=lambda x: x.fitness.score, reverse=True)
        #     elif self.optimization_type == "closest_to_zero":
        #         elites:List[Genome] = sorted(population.values(), key=lambda x: abs(x.fitness.score), reverse=False)
        # else:
        #     if self.optimization_type == "minimize":
        #         elites:List[Genome] = sorted(population.values(), key=lambda x: x.info[criteria], reverse=False)
        #     elif self.optimization_type == "maximize":
        #         elites:List[Genome] = sorted(population.values(), key=lambda x: x.info[criteria], reverse=True)
        #     elif self.optimization_type == "closest_to_zero":
        #         elites:List[Genome] = sorted(population.values(), key=lambda x: abs(x.info[criteria]), reverse=False)


        elites_ids:List[int] = []
        for i in range(size):
            elites[i].info["is_elite"] = True
            elites_ids.append(elites[i].id)
        return np.array(elites_ids, dtype=np.int32)[:size]

    def tournament_selection(self, population:Dict[int, Genome], tournament_size:int, size:int, criteria:str="fitness") -> np.ndarray:
        if criteria != "fitness" and type(population[next(iter(population))].info[criteria]) != float: raise Exception("Reproduction: criteria must be a float")
        if tournament_size > size: tournament_size = size
        elites_ids:Set[int] = set()
        population_list:List[Genome] = list(population.values())
        while len(elites_ids) < size:
            # 1 - Get the parents, TODO: -> Possible to increase the number of parents to put more pressure on the selection
            parents:List[Genome] = np.random.choice(population_list, size=tournament_size, replace=False)

            # 2 - Choose parent by criteria
            if criteria == "fitness":
                values:np.ndarray = np.array([parent.fitness.score for parent in parents])
            else:
                values:np.ndarray = np.array([parent.info[criteria] for parent in parents])
            
            # 3 - Choose the parent
            if self.optimization_type == "maximize":
                best_parent:Genome = parents[np.argmax(values)]
            else:
                best_parent:Genome = parents[np.argmin(values)]
            
            # 4 - Add to elite population
            elites_ids.add(best_parent.id)
        return np.array(list(elites_ids), dtype=np.int32)
         

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def parents_selection_jit(parents:Tuple[np.ndarray], probs:np.ndarray, size:int) -> np.ndarray:
        # # Check if the arrays and the probabilities have the same length
        assert len(parents) == len(probs), "Arrays and probabilities must have the same length"
        assert np.sum(probs) < 1.1, "Probabilities must sum to 1"

        # 1- Initialize parents
        len_parents:int = len(parents)
        parents_array:np.ndarray = np.empty((len_parents, size), dtype=np.int32)        
        probs_cumsum:np.ndarray = np.cumsum(probs)

        # 2- Define arrays list once
        for i in nb.prange(size):
            for j in nb.prange(len_parents):
                # 2.1 - Choose the population to choose from
                population_choosen:np.ndarray = parents[np.searchsorted(probs_cumsum, np.random.random(), side="right")]
                # 2.2 - Choose the parent from the population choosen
                parents_array[j, i] = np.random.choice(population_choosen)
    
        return parents_array


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def sbx(parent_1:np.ndarray, parent_2:np.ndarray, new_parameter:np.ndarray, eta:float=10.0, lower_bound:float=0.0, upper_bound:float=1.0) -> np.ndarray:
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover

        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        eta:float = eta
        xl:float = lower_bound # lower bound
        xu:float = upper_bound # upper bound
        r1:np.ndarray = np.random.random(size=len(parent_1))
        r2:np.ndarray = np.random.random(size=len(parent_1))
        new_parameter[:] = parent_1[:] # copy parent 1
        # x1, x2, beta, alpha, rand, beta_q, c1, c2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for i in nb.prange(0, len(parent_1)):
            # 1 - If the values are different, then SBX
            if abs(parent_1[i] - parent_2[i]) > 1e-15:
                x1:float = min(parent_1[i], parent_2[i])
                x2:float = max(parent_1[i], parent_2[i])
                
                # Compute c1
                beta:float = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha:float = 2.0 - beta ** -(eta + 1)
                rand:float = r1[i]
                if rand <= 1.0 / alpha:
                    beta_q:float = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q:float = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c1:float = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                # Compute c2
                beta:float = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha:float = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q:float = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q:float = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2:float = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1:float = min(max(c1, xl), xu)
                c2:float = min(max(c2, xl), xu)

                if r2[i] <= 0.5:
                    new_parameter[i] = c2
                else:
                    new_parameter[i] = c1
            # # 2 - If the values are the same, then the child is the same as the parents
            # else:
            #     new_parameter[i] = parent_1[i]
        return new_parameter

    def __sbx_original(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover

        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        eta:float = 10.0
        xl = 0 # lower bound
        xu = 1 # upper bound
        z = x.copy()
        r1 = np.random.random(size=len(x))
        r2 = np.random.random(size=len(x))

        for i in range(0, len(x)):
            if abs(x[i] - y[i]) > 1e-15:
                x1 = min(x[i], y[i])
                x2 = max(x[i], y[i])

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                rand = r1[i]
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if r2[i] <= 0.5:
                    z[i] = c2
                else:
                    z[i] = c1
        return z

class Reproduction_Classic(Reproduction):
    def __init__(self, config_path_file:str, attributes:Attribute_Paramaters, build_genome_function:Callable = None) -> None:
        Reproduction.__init__(self, config_path_file, attributes)

        # Initialize configs
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Genome_Classic"])
        self.reproduction_function = self.__reproduction_classic
        self.build_genome_function:Callable = build_genome_function

    def __reproduction_classic(self, parents:np.ndarray, current_population:Dict[int, Genome_Classic], new_population:Dict[int, Genome_Classic]) -> Dict[int, Genome_Classic]:
        size:int = len(parents[0])
        for i in range(size):
            # 1 - Get the parents and the new genome
            parent_1:np.ndarray = current_population[parents[0][i]].parameter
            parent_2:np.ndarray = current_population[parents[1][i]].parameter

            if self.build_genome_function is not None:
                new_genome:Genome_Classic = self.build_genome_function()
            else:
                new_genome:Genome_Classic = Genome_Classic(get_new_genome_id(), self.config["Genome_Classic"], self.attributes)
            
            # 2 - Crossover
            if self.is_sbx == True:
                self.sbx(parent_1, parent_2, new_genome.parameter, eta=self.sbx_eta, lower_bound=self.attributes.min_parameters["classic"][0], upper_bound=self.attributes.max_parameters["classic"][0])
            else:
                self.crossover_jit(parent_1, parent_2, new_genome.parameter, 0.5) # 0.5 = 50% parents 1, 50% parents 2

            # 3 - Add to new population
            new_population[new_genome.id] = new_genome

        return new_population


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def crossover_jit(decoder_1:np.ndarray, decoder_2:np.ndarray, new_decoder:np.ndarray, prob:float) -> None:
        # 1 - Crossover
        for i in nb.prange(new_decoder.shape[0]):
            if np.random.random() < prob: 
                new_decoder[i] = decoder_1[i]
            else: 
                new_decoder[i] = decoder_2[i]


class Reproduction_Decoder(Reproduction):
    def __init__(self, config_path_file:str, attributes:Attribute_Paramaters) -> None:
        Reproduction.__init__(self, config_path_file, attributes)

        # Initialize configs
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Genome_Decoder"])
        self.reproduction_function = self.__reproduction_decoder

    def __reproduction_decoder(self, parents:np.ndarray, current_population:Dict[int, Genome_Decoder], new_population:Dict[int, Genome_Decoder]) -> Dict[int, Genome_Decoder]:
        size:int = len(parents[0])
        for i in range(size):
            # 1 - Get the parents and the new genome
            decoder_1_neuron:np.ndarray = current_population[parents[0][i]].decoder_neuron
            decoder_2_neuron:np.ndarray = current_population[parents[1][i]].decoder_neuron

            decoder_1_synpase:np.ndarray = current_population[parents[0][i]].decoder_synapse
            decoder_2_synpase:np.ndarray = current_population[parents[1][i]].decoder_synapse

            new_genome:Genome_Decoder = Genome_Decoder(get_new_genome_id(), self.config["Genome_Decoder"])
            new_decoder_neuron:np.ndarray = new_genome.decoder_neuron
            new_decoder_synapse:np.ndarray = new_genome.decoder_synapse
            
            # 2 - Crossover
            if self.is_sbx == True:
                new_genome.decoder_neuron = self.sbx(decoder_1_neuron, decoder_2_neuron, new_decoder_neuron, eta=self.sbx_eta)
                new_genome.decoder_synapse = self.sbx(decoder_1_synpase, decoder_2_synpase, new_decoder_synapse, eta=self.sbx_eta)
            else:
                self.crossover_jit(decoder_1_neuron, decoder_2_neuron, new_decoder_neuron, 0.5) # 0.5 = 50% parents 1, 50% parents 2
                self.crossover_jit(decoder_1_synpase, decoder_2_synpase, new_decoder_synapse, 0.5) # 0.5 = 50% parents 1, 50% parents 2

            # 3 - Add to new population
            new_population[new_genome.id] = new_genome
        return new_population

    
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def crossover_jit(decoder_1:np.ndarray, decoder_2:np.ndarray, new_decoder:np.ndarray, prob:float) -> None:
        # 1 - Crossover
        for i in nb.prange(new_decoder.shape[0]):
            if np.random.random() < prob: 
                new_decoder[i] = decoder_1[i]
            else: 
                new_decoder[i] = decoder_2[i]

class Reproduction_NN(Reproduction):
    def __init__(self, config_path_file:str, attributes:Attribute_Paramaters) -> None:
        Reproduction.__init__(self, config_path_file, attributes)

        # Initialize configs
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Genome_NN"])
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.parameters_name_neuron:List[str] = attributes.parameters_neuron_names
        self.parameters_name_synapse:List[str] = attributes.parameters_synapse_names
        
        if self.network_type == "SNN":
            # self.parameters_name_neuron:List[str] = ["voltage", "threshold", "tau", "input_current", "refractory"]
            # self.parameters_name_synapse:List[str] = ["weight", "delay"]
            self.reproduction_function = self.__reproduction_snn if self.is_sbx == False else self.__reproduction_sbx
        elif self.network_type == "ANN":
            # self.parameters_name_neuron:List[str] = ["bias"]
            # self.parameters_name_synapse:List[str] = ["weight"]
            self.reproduction_function = self.__reproduction_ann if self.is_sbx == False else self.__reproduction_sbx
        else:
            raise Exception("Reproduction_NN: network_type not recognized", self.network_type, "should be SNN or ANN")

    def __set_init_new_genome_score(self, new_genome:Genome_NN) -> None:
        if self.optimization_type == "maximize":
            new_genome.fitness.score = -np.inf
        else:
            new_genome.fitness.score = np.inf

    def __reproduction_snn(self, parents:np.ndarray, current_population:Dict[int, Genome_NN], new_population:Dict[int, Genome_NN]) -> Dict[int, Genome_NN]:
        size:int = len(parents[0])
        for i in range(size):
            # 1 - Get the parents and the new genome
            nn_1:NN = current_population[parents[0][i]].nn
            nn_2:NN = current_population[parents[1][i]].nn
            new_genome:Genome_NN = Genome_NN(
                                            id= get_new_genome_id(), 
                                            config_genome=self.config["Genome_NN"], 
                                            attributes_manager=self.attributes,
                                            hiddens_active=0, 
                                            )
            new_nn:NN = new_genome.nn
            self.__set_init_new_genome_score(new_genome)

            # 2 - Common neurons/synapses indexes between the two parents
            common_neurons_indexes, common_synapses_indexes = Distance.get_common_indexes(nn_1, nn_2, only_hidden=True)

            # 3 - Crossover
            self.crossover_snn_jit(
                nn_1.parameters["voltage"],       nn_2.parameters["voltage"],       new_nn.parameters["voltage"],       0.5, # 0.5 = 50% parents 1, 50% parents 2
                nn_1.parameters["threshold"],     nn_2.parameters["threshold"],     new_nn.parameters["threshold"],     0.5,
                nn_1.parameters["tau"],           nn_2.parameters["tau"],           new_nn.parameters["tau"],           0.5,
                nn_1.parameters["input_current"], nn_2.parameters["input_current"], new_nn.parameters["input_current"], 0.5,
                nn_1.parameters["weight"],        nn_2.parameters["weight"],        new_nn.parameters["weight"],        0.5,
                # nn_1.parameters["coeff"],        nn_2.parameters["coeff"],        new_nn.parameters["coeff"],        0.5,
                # nn_1.parameters["refractory"],    nn_2.parameters["refractory"],    new_nn.parameters["refractory"],    0.5,
                # nn_1.parameters["delay"],         nn_2.parameters["delay"],         new_nn.parameters["delay"],         0.5,
                new_nn.neurons_status, common_neurons_indexes, 1.0, # 1.0 = take 100% of parents neurons that are active
                new_nn.synapses_status, common_synapses_indexes, 1.0 # 1.0 = take 100% of parents synapses that are active (considering the neurons choosen from the parents (previous step)))
            )

            # 4 - update indexes
            new_nn.update_indexes()
            new_population[new_genome.id] = new_genome
        return new_population

    def __reproduction_ann(self, parents:np.ndarray, current_population:Dict[int, Genome_NN], new_population:Dict[int, Genome_NN]) -> Dict[int, Genome_NN]:
        size:int = len(parents[0])
        for i in range(size):
            # 1 - Get the parents and the new genome
            nn_1:NN = current_population[parents[0][i]].nn
            nn_2:NN = current_population[parents[1][i]].nn
            new_genome:Genome_NN = Genome_NN(
                                            id= get_new_genome_id(), 
                                            config_genome=self.config["Genome_NN"], 
                                            attributes_manager=self.attributes,
                                            hiddens_active=0, 
                                            )
            new_nn:NN = new_genome.nn
            self.__set_init_new_genome_score(new_genome)

            # 2 - Common neurons/synapses indexes between the two parents
            common_neurons_indexes, common_synapses_indexes = Distance.get_common_indexes(nn_1, nn_2, only_hidden=True)

            # 3 - Crossover
            self.crossover_ann_jit(
                nn_1.parameters["bias"],          nn_2.parameters["bias"],          new_nn.parameters["bias"],       0.5, # 0.5 = 50% parents 1, 50% parents 2
                nn_1.parameters["weight"],        nn_2.parameters["weight"],        new_nn.parameters["weight"],      0.5,
                new_nn.neurons_status, common_neurons_indexes, 1.0, # 1.0 = take 100% of parents neurons that are active
                new_nn.synapses_status, common_synapses_indexes, 1.0 # 1.0 = take 100% of parents synapses that are active (considering the neurons choosen from the parents (previous step)))
            )

            # 4 - update indexes
            new_nn.update_indexes()
            new_population[new_genome.id] = new_genome
        return new_population



    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def crossover_nn_neuron(
            parameters_1:np.ndarray, parameters_2:np.ndarray, new_parameters:np.ndarray, prob:float,
            neurons_states_new:np.ndarray, common_neurons_indexes:np.ndarray, prob_neurons_states:float,
    ):
        # 1 - Crososver topology and get topology neurons/synapses indexes
        random_indexes:np.ndarray = (np.random.uniform(0, 1, common_neurons_indexes.shape[0]) < prob_neurons_states).astype(np.int32)
        neurons_states_new[common_neurons_indexes[random_indexes == 1]] = True
        new_neurons_indexes:np.ndarray = np.where(neurons_states_new)[0]

        for i in range(new_neurons_indexes.shape[0]):
            if np.random.random() < prob: new_parameters[new_neurons_indexes[i]] = parameters_1[new_neurons_indexes[i]]
            else: new_parameters[new_neurons_indexes[i]] = parameters_2[new_neurons_indexes[i]]


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def crossover_nn_synapse(
            parameters_1:np.ndarray, parameters_2:np.ndarray, new_parameters:np.ndarray, prob:float,
            neurons_states_new:np.ndarray, synapses_states_new:np.ndarray, common_synapses_indexes:np.ndarray, prob_synapses_states:float,
    ):
        new_neurons_indexes:np.ndarray = np.where(neurons_states_new)[0]
        new_neurons_indexes_set:set = set(new_neurons_indexes)
        for i in range(common_synapses_indexes.shape[1]):
            x:float = common_synapses_indexes[0, i]
            y:float = common_synapses_indexes[1, i]
            if (x in new_neurons_indexes_set or y in new_neurons_indexes_set) and np.random.random() < prob_synapses_states:
                synapses_states_new[x, y] = True
                if np.random.random() < prob: new_parameters[x, y] = parameters_1[x, y]
                else: new_parameters[x, y] = parameters_2[x, y]

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def crossover_snn_jit(
        voltage_1:np.ndarray, voltalge_2:np.ndarray, voltage_new:np.ndarray, prob_voltage:float,
        threshold_1:np.ndarray, threshold_2:np.ndarray, threshold_new:np.ndarray, prob_threshold:float,
        tau_1:np.ndarray, tau_2:np.ndarray, tau_new:np.ndarray, prob_tau:float,
        input_current_1:np.ndarray, input_current_2:np.ndarray, input_current_new:np.ndarray, prob_input_current:float,
        weight_1:np.ndarray, weight_2:np.ndarray, weight_new:np.ndarray, prob_weight:float,
        # coeff_1:np.ndarray, coeff_2:np.ndarray, coeff_new:np.ndarray, prob_coeff:float,
        # refractory_1:np.ndarray, refractory_2:np.ndarray, refractory_new:np.ndarray, prob_refractory:float,
        # delay_1:np.ndarray, delay_2:np.ndarray, delay_new:np.ndarray, prob_delay:float,
        neurons_states_new:np.ndarray, common_neurons_indexes:np.ndarray, prob_neurons_states:float,
        synapses_states_new:np.ndarray, common_synapses_indexes:np.ndarray, prob_synapses_states:float
        ):
        # 1 - Crososver topology and get topology neurons/synapses indexes
        random_indexes:np.ndarray = (np.random.uniform(0, 1, common_neurons_indexes.shape[0]) < prob_neurons_states).astype(np.int32)
        neurons_states_new[common_neurons_indexes[random_indexes == 1]] = True
        new_neurons_indexes:np.ndarray = np.where(neurons_states_new)[0]

        # 2 - Crossover neurons
        for i in nb.prange(new_neurons_indexes.shape[0]):
            # 2.1 - Voltages
            if np.random.random() < prob_voltage: voltage_new[new_neurons_indexes[i]] = voltage_1[new_neurons_indexes[i]]
            else: voltage_new[new_neurons_indexes[i]] = voltalge_2[new_neurons_indexes[i]]

            # 2.2 - thresholds
            if np.random.random() < prob_threshold: threshold_new[new_neurons_indexes[i]] = threshold_1[new_neurons_indexes[i]]
            else: threshold_new[new_neurons_indexes[i]] = threshold_2[new_neurons_indexes[i]]

            # 2.3 - Tau
            if np.random.random() < prob_tau: tau_new[new_neurons_indexes[i]] = tau_1[new_neurons_indexes[i]]
            else: tau_new[new_neurons_indexes[i]] = tau_2[new_neurons_indexes[i]]

            # 2.4 - Input current
            if np.random.random() < prob_input_current: input_current_new[new_neurons_indexes[i]] = input_current_1[new_neurons_indexes[i]]
            else: input_current_new[new_neurons_indexes[i]] = input_current_2[new_neurons_indexes[i]]

            # # 2.5 - Coeff
            # if np.random.random() < prob_coeff: coeff_new[new_neurons_indexes[i]] = coeff_1[new_neurons_indexes[i]]
            # else: coeff_new[new_neurons_indexes[i]] = coeff_2[new_neurons_indexes[i]]

            # 2.5 - Refractory
            # if np.random.random() < prob_refractory: refractory_new[new_neurons_indexes[i]] = refractory_1[new_neurons_indexes[i]]
            # else: refractory_new[new_neurons_indexes[i]] = refractory_2[new_neurons_indexes[i]]

        # 3 - Crossover synapses
        new_neurons_indexes_set:set = set(new_neurons_indexes)
        for i in nb.prange(common_synapses_indexes.shape[1]):
            x:float = common_synapses_indexes[0, i]
            y:float = common_synapses_indexes[1, i]
            if (x in new_neurons_indexes_set or y in new_neurons_indexes_set) and np.random.random() < prob_synapses_states:
                # 3.1 - Status
                synapses_states_new[x, y] = True
                # 3.2 - Weight
                if np.random.random() < prob_weight: weight_new[x, y] = weight_1[x, y]
                else: weight_new[x, y] = weight_2[x, y]

                # 3.3 - Delay
                # if np.random.random() < prob_delay: delay_new[x, y] = delay_1[x, y]
                # else: delay_new[x, y] = delay_2[x, y]

        # print("random_indexes:\n", random_indexes, "len:", len(random_indexes))
        # print("common_neurons_indexes:\n", common_neurons_indexes, "len:", len(common_neurons_indexes))
        # print("new_neurons_indexes:\n", new_neurons_indexes, "len:", len(new_neurons_indexes))
        # print("new_neurons_indexes_set:\n", new_neurons_indexes_set, "len:", len(new_neurons_indexes_set))
        # print("common_synapses_indexes:\n", common_synapses_indexes[0], "\n", common_synapses_indexes[1])
        # print("synapses_states_new:\n", synapses_states_new)
        # print("synpases_indexes:\n", np.where(synapses_states_new))

    
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def crossover_ann_jit(
        bias_1:np.ndarray, bias_2:np.ndarray, bias_new:np.ndarray, prob_bias:float,
        weight_1:np.ndarray, weight_2:np.ndarray, weight_new:np.ndarray, prob_weight:float,
        neurons_states_new:np.ndarray, common_neurons_indexes:np.ndarray, prob_neurons_states:float,
        synapses_states_new:np.ndarray, common_synapses_indexes:np.ndarray, prob_synapses_states:float
        ):
        # 1 - Crososver topology and get topology neurons/synapses indexes
        random_indexes:np.ndarray = (np.random.uniform(0, 1, common_neurons_indexes.shape[0]) < prob_neurons_states).astype(np.int32)
        neurons_states_new[common_neurons_indexes[random_indexes == 1]] = True
        new_neurons_indexes:np.ndarray = np.where(neurons_states_new)[0]

        # 2 - Crossover neurons
        for i in nb.prange(new_neurons_indexes.shape[0]):
            # 2.1 - Bias
            if np.random.random() < prob_bias: bias_new[new_neurons_indexes[i]] = bias_1[new_neurons_indexes[i]]
            else: bias_new[new_neurons_indexes[i]] = bias_2[new_neurons_indexes[i]]


        # 3 - Crossover synapses
        new_neurons_indexes_set:set = set(new_neurons_indexes)
        for i in nb.prange(common_synapses_indexes.shape[1]):
            x:float = common_synapses_indexes[0, i]
            y:float = common_synapses_indexes[1, i]
            if (x in new_neurons_indexes_set or y in new_neurons_indexes_set) and np.random.random() < prob_synapses_states:
                # 3.1 - Status
                synapses_states_new[x, y] = True
                # 3.2 - Weight
                if np.random.random() < prob_weight: weight_new[x, y] = weight_1[x, y]
                else: weight_new[x, y] = weight_2[x, y]


        # print("random_indexes:\n", random_indexes, "len:", len(random_indexes))
        # print("common_neurons_indexes:\n", common_neurons_indexes, "len:", len(common_neurons_indexes))
        # print("new_neurons_indexes:\n", new_neurons_indexes, "len:", len(new_neurons_indexes))
        # print("new_neurons_indexes_set:\n", new_neurons_indexes_set, "len:", len(new_neurons_indexes_set))
        # print("common_synapses_indexes:\n", common_synapses_indexes[0], "\n", common_synapses_indexes[1])
        # print("synapses_states_new:\n", synapses_states_new)
        # print("synpases_indexes:\n", np.where(synapses_states_new))

    
    def __reproduction_sbx(self, parents:np.ndarray, current_population:Dict[int, Genome_NN], new_population:Dict[int, Genome_NN]) -> Dict[int, Genome_NN]:
        size:int = len(parents[0])
        for i in range(size):
            # 1 - Get the parents and the new genome
            nn_1:NN = current_population[parents[0][i]].nn
            nn_2:NN = current_population[parents[1][i]].nn
            new_genome:Genome_NN = Genome_NN(
                                            id=get_new_genome_id(), 
                                            config_genome=self.config["Genome_NN"], 
                                            attributes_manager=self.attributes,
                                            hiddens_active=0, 
                                            )
            new_nn:NN = new_genome.nn
            self.__set_init_new_genome_score(new_genome)

            # 2 - Common neurons/synapses indexes between the two parents
            common_neurons_indexes, common_synapses_indexes = Distance.get_common_indexes(nn_1, nn_2, only_hidden=False)
            new_nn.neurons_status[common_neurons_indexes] = True
            new_nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]] = True

            # 3 - Crossover
            # 3.1 - Neurons Crossover
            for parameter_name in self.parameters_name_neuron:
                new_nn.parameters[parameter_name][common_neurons_indexes] = self.sbx(
                    nn_1.parameters[parameter_name][common_neurons_indexes], 
                    nn_2.parameters[parameter_name][common_neurons_indexes], 
                    new_nn.parameters[parameter_name][common_neurons_indexes],
                    eta=self.sbx_eta,
                    lower_bound=self.attributes.min_parameters[parameter_name][0], # min
                    upper_bound=self.attributes.max_parameters[parameter_name][0]  # max
                    )
            # 3.2 - Synapses Crossover
            for parameter_name in self.parameters_name_synapse:
                new_nn.parameters[parameter_name][common_synapses_indexes[0], common_synapses_indexes[1]] = self.sbx(
                    nn_1.parameters[parameter_name][common_synapses_indexes[0], common_synapses_indexes[1]], 
                    nn_2.parameters[parameter_name][common_synapses_indexes[0], common_synapses_indexes[1]], 
                    new_nn.parameters[parameter_name][common_synapses_indexes[0], common_synapses_indexes[1]],
                    eta=self.sbx_eta,
                    lower_bound=self.attributes.min_parameters[parameter_name][0], # min
                    upper_bound=self.attributes.max_parameters[parameter_name][0]  # max
                    )
            
            # print("nn_1.parameters[voltage][common_neurons_indexes]", nn_1.parameters["tau"][common_neurons_indexes])
            # print("nn_2.parameters[voltage][common_neurons_indexes]", nn_2.parameters["tau"][common_neurons_indexes])
            # print("new_nn.parameters[voltage][common_neurons_indexes]", new_nn.parameters["tau"][common_neurons_indexes])

            # # print("nn_1.parameters[weight][common_synapses_indexes[0], common_synapses_indexes[1]]\n", nn_1.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]])
            # # print("nn_2.parameters[weight][common_synapses_indexes[0], common_synapses_indexes[1]]\n", nn_2.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]])
            # # print("new_nn.parameters[weight]\n", new_nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]])
            # exit()
            # 4 - update indexes
            new_nn.update_indexes()
            new_population[new_genome.id] = new_genome
        return new_population