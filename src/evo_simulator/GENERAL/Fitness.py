import numpy as np
import numba as nb
from typing import Dict, List
import sys

class Fitness():
    def __init__(self) -> None:
        
        self.score: float = sys.float_info.min
        self.max:float = sys.float_info.min
        self.mean:float = 0.0
        self.min:float = sys.float_info.max
        
        self.history_best: List[float] = [] # has to not be reset
        self.history_mean: List[float] = [] # has to not be reset

        # Extra info
        self.extra_info:Dict[str, float] = {}
    
    def reset(self) -> None:
        self.score: float = sys.float_info.min
        self.max:float = sys.float_info.min
        self.mean:float = 0.0
        self.min:float = sys.float_info.max

        # Extra info
        self.extra_info:Dict[str, float] = {}

    def copy(self):
        fitness: Fitness = Fitness()
        fitness.score = self.score
        fitness.max = self.max
        fitness.mean = self.mean
        fitness.min = self.min
        fitness.history_best = self.history_best.copy()
        fitness.history_mean = self.history_mean.copy()
        fitness.extra_info = {}.update(self.extra_info)
        return fitness
    
class Fitness_Manager:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def deterministic_neuron_output_accuracy_numerical(output_network:np.ndarray, output_ground:np.ndarray) -> float:
        '''
            Return the output of the network by taking the neuron with the highest output
        '''
        output_network = np.argmax(output_network, axis=1)
        accuracy:float = np.sum(output_network == output_ground) / output_ground.shape[0]
        return accuracy

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def deterministic_neuron_output_accuracy_spikes(output_indexes:np.ndarray, neurons_output:np.ndarray, output_ground:np.ndarray) -> float:
        '''
            Return the output of the network by taking the neuron with the highest output
        '''
        values:np.ndarray = np.empty(output_indexes.shape[0], dtype=np.float32)
        accuracy:float = 0.0
        record = np.empty(output_ground.shape[0], dtype=np.int32)

        for i in nb.prange(output_ground.shape[0]):
            for j in nb.prange(output_indexes.shape[0]):
                values[j] = np.sum(neurons_output[i][output_indexes[j]])
            output_max = np.argmax(values)
            record[i] = output_max
            if output_max == output_ground[i]:
                accuracy += 1
        return accuracy / output_ground.shape[0]
    
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def probabilistic_neuron_output_accuracy(output_indexes:np.ndarray, neurons_output:np.ndarray, output_ground:np.ndarray) -> float:
        '''
            Return the output of the network by taking a random neuron based on the probability of each neuron
        '''
        summed_values:np.ndarray = np.empty(output_indexes.shape[0], dtype=np.int32)
        probabilities:np.ndarray = np.empty(output_indexes.shape[0], dtype=np.float32)
        total_summed_values:float = 0.0
        accuracy:float = 0.0

        for i in nb.prange(output_ground.shape[0]):
            # 1 - Sum all output values based on the output_indexes
            total_summed_values = 0.0
            for j in nb.prange(output_indexes.shape[0]):
                summed_values[j] = np.sum(neurons_output[i][output_indexes[j]])
                total_summed_values += summed_values[j]

            # 2 - Calculate the probability of each output
            for j in nb.prange(output_indexes.shape[0]):
                probabilities[j] = summed_values[j] / total_summed_values

            # 3 - Choose a random output based on the probability and check if it is correct
            choice:int = np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right") # equivalent to np.random.choice(output_indexes, p=probabilities)
            if choice == output_ground[i]:
                accuracy += 1        
        return accuracy / output_ground.shape[0]