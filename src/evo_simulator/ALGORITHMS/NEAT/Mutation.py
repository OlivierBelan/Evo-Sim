import numpy as np
import numba as nb
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
from evo_simulator.GENERAL.Mutation_NN import Mutation
from typing import Dict, List, Tuple
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
import random

class Mutation_NEAT(Mutation):
    def __init__(self, config_path_file:str, attributes_manager:Attribute_Paramaters) -> None:
        Mutation.__init__(self, config_path_file, attributes_manager)
        self.neuron_parameters_names:List[str] = attributes_manager.parameters_neuron_names
        self.synapse_parameters_names:List[str] = attributes_manager.parameters_synapse_names

    def mutation_neat(self, population:Population) -> Population:
        population_dict:Dict[int, Genome_NN] = population.population
        first_key:int = next(iter(population_dict))
        self.nb_neurons:int = population_dict[first_key].nn.nb_neurons
        self.__mutation_attribute_neat(population.population)
        self.__mutation_topology_neat(population.population)
        return population

    def __mutation_attribute_neat(self, population:Dict[int, Genome_NN]) -> None:
        population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in population.items() if genome.info["is_elite"] == False and random.random() < self.prob_mutate_neuron_params}
        self.attributes.neurons_sigma(population_to_mutate, self.neuron_parameters_names)
        population_to_mutate:Dict[int, Genome_NN] = {id:genome for id, genome in population.items() if genome.info["is_elite"] == False and random.random() < self.prob_mutate_synapse_params}
        self.attributes.synapses_sigma(population_to_mutate, self.synapse_parameters_names)

    def __mutation_topology_neat(self, population:Dict[int, Genome_NN]) -> None:
        self.__update_neuron_status_population(population)
        for genome in population.values():
            if genome.info["is_elite"] == False:
                nn:NN = genome.nn
                # Mutation neuron
                if random.random() < self.prob_add_neuron:
                    self.__add_neuron(nn)
                if random.random() < self.prob_delete_neuron:
                    self.__del_neuron(nn)
                if random.random() < self.prob_activate_neuron:
                    self.__activate_neuron(nn)
                if random.random() < self.prob_deactivate_neuron:
                    self.__deactivate_neuron(nn)

                # Mutation synapse
                if random.random() < self.prob_add_synapse:
                    self.__add_synapse(nn)
                if random.random() < self.prob_delete_synapse:
                    self.__del_synapse(nn)
                if random.random() < self.prob_activate_synapse:
                    self.__activate_synapse(nn)
                if random.random() < self.prob_deactivate_synapse:
                    self.__deactivate_synapse(nn)

    def __update_neuron_status_population(self, population:Dict[int, Genome_NN]) -> None:
        population:Dict[int, Genome_NN] = population
        self.neurons_status_population:np.ndarray = np.zeros(self.nb_neurons, dtype=np.bool8)
        for genome in population.values():
            self.neurons_status_population[genome.nn.hiddens["neurons_indexes_active"]] = True
           
    def __add_neuron(self, nn:NN, size:int=1):
        if nn.nb_hiddens - nn.hiddens["neurons_indexes_active"].size < size: return
        if nn.nb_hiddens == nn.hiddens["neurons_indexes_active"].size: return
        if nn.synapses_actives_indexes[0].size == 0: return
        if random.random() < self.prob_creation_mutation:
            neurons_population_used_indexes:np.ndarray = np.where(self.neurons_status_population == True)[0]
            neuron_population_unused:np.ndarray = np.setdiff1d(neurons_population_used_indexes, nn.hiddens["neurons_indexes_active"])
        else: 
            neuron_population_unused:np.ndarray = np.array([], dtype=np.int64)
        new_neurons, neurons_in, neurons_out = self.__add_neuron_jit(nn.neurons_status, nn.synapses_actives_indexes, neuron_population_unused, size)
        self.neurons_status_population[new_neurons] = True
        self.topologies.add_neuron_betwen_two_neurons(nn, new_neurons, neurons_in, neurons_out)
  
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __add_neuron_jit(neurons_status:np.ndarray, synapses_actives_indexes:Tuple[np.ndarray, np.ndarray], neuron_population_unused:np.ndarray, size:int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        random_active_synapses:np.ndarray = np.unique(np.random.randint(0, synapses_actives_indexes[0].shape[0], size=size))
        neurons_in:np.ndarray = synapses_actives_indexes[0][random_active_synapses]
        neurons_out:np.ndarray = synapses_actives_indexes[1][random_active_synapses]
        if neuron_population_unused.shape[0] > 0:
            new_neurons:np.ndarray = np.random.choice(neuron_population_unused, size=size, replace=False)
        else:
            new_neurons:np.ndarray = np.random.choice(np.where(~neurons_status)[0], size=size, replace=False) # Equivalent to np.where(neurons_status == False)
        return new_neurons, neurons_in, neurons_out


    def __del_neuron(self, nn:NN, size:int=1):
        if nn.hiddens["neurons_indexes_active"].size == 0 or nn.hiddens["neurons_indexes_active"].size < size: return
        neurons_to_del:np.ndarray = self.__del_neuron_jit(nn.hiddens["neurons_indexes_active"], size)
        nn.neuron_deactived_status[neurons_to_del - (nn.nb_inputs + nn.nb_outputs)] = False
        self.topologies.del_neurons(nn, neurons_to_del)
  
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __del_neuron_jit(hidden_neurons_indexes_active:np.ndarray, size=1) -> np.ndarray:
        neurons_to_del:np.ndarray = np.random.choice(hidden_neurons_indexes_active, size=size, replace=False)
        return neurons_to_del


    def __activate_neuron(self, nn:NN, size:int=1):
        neurons_to_activate:np.ndarray = self.__activate_neuron_jit(nn.neuron_deactived_status, size)
        if neurons_to_activate is None: return
        nn.neuron_deactived_status[neurons_to_activate] = False
        neurons_to_activate += nn.nb_inputs + nn.nb_outputs
        self.topologies.activate_neurons(nn, neurons_to_activate)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __activate_neuron_jit(neuron_deactived:np.ndarray, size:int=1):
        unactive_neurons_indexes:np.ndarray = np.where(neuron_deactived)[0] # should be only hiddens neurons as input and output neurons are always active
        if unactive_neurons_indexes.size == 0: return None
        return np.random.choice(unactive_neurons_indexes, size=size, replace=False)


    def __deactivate_neuron(self, nn:NN, size:int=1):
        if nn.hiddens["neurons_indexes_active"].size < size: return None
        neurons_to_deactivate:np.ndarray = self.__deactivate_neuron_jit(nn.hiddens["neurons_indexes_active"], size)
        nn.neuron_deactived_status[neurons_to_deactivate - (nn.nb_inputs + nn.nb_outputs)] = True
        self.topologies.deactivate_neurons(nn, neurons_to_deactivate)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __deactivate_neuron_jit(hidden_neurons_indexes_active:np.ndarray, size:int=1) -> np.ndarray:
        neurons_to_deactivate:np.ndarray = np.random.choice(hidden_neurons_indexes_active, size=size, replace=False)
        return neurons_to_deactivate


    def __add_synapse(self, nn:NN, size:int=1):
        neurons_in, neurons_out = self.__add_synapse_jit(nn.synapses_unactives_indexes, size)
        if neurons_in is None: return
        self.topologies.add_synapses(nn, neurons_in, neurons_out)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __add_synapse_jit(synapses_unactive_indexes:Tuple[np.ndarray, np.ndarray], size:int=1) -> Tuple[np.ndarray, np.ndarray]:
        if synapses_unactive_indexes[0].size == 0: return None, None
        random_unactive_synapses:np.ndarray = np.unique(np.random.randint(0, synapses_unactive_indexes[0].shape[0], size=size))
        neurons_in:np.ndarray = synapses_unactive_indexes[0][random_unactive_synapses]
        neurons_out:np.ndarray = synapses_unactive_indexes[1][random_unactive_synapses]
        return neurons_in, neurons_out


    def __del_synapse(self, nn:NN, size:int=1):
        if nn.synapses_actives_indexes[0].size == 0: return
        neurons_in, neurons_out = self.__del_synapse_jit(nn.synapses_actives_indexes, size)
        self.topologies.del_synapses(nn, neurons_in, neurons_out)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __del_synapse_jit(synapses_actives_indexes:Tuple[np.ndarray, np.ndarray], size:int=1) -> Tuple[np.ndarray, np.ndarray]:
        random_active_synapses:np.ndarray = np.unique(np.random.randint(0, synapses_actives_indexes[0].shape[0], size=size))
        neurons_in:int = synapses_actives_indexes[0][random_active_synapses]
        neurons_out:int = synapses_actives_indexes[1][random_active_synapses]
        return neurons_in, neurons_out


    def __activate_synapse(self, nn:NN, size:int=1):
        neurons_in, neurons_out = self.__add_synapse_jit(nn.synapses_unactives_weight_indexes, size)
        if neurons_in is None: return
        self.topologies.activate_synapses(nn, neurons_in, neurons_out)

    def __deactivate_synapse(self, nn:NN, size:int=1):
        if nn.synapses_actives_indexes[0].size == 0: return
        neurons_in, neurons_out = self.__del_synapse_jit(nn.synapses_actives_indexes, size)
        self.topologies.deactivate_synapses(nn, neurons_in, neurons_out)
