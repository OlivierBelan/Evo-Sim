import numpy as np
import numba as nb
import evo_simulator.TOOLS as TOOLS
from typing import Dict, List, Any

class Decoder:
    def __init__(self, config_path:str) -> None:
        config:Dict[str, Any] = TOOLS.config_function(config_path, ["Runner_Info", "Genome_NN"])
        self.run_time:int = int(config["Runner_Info"]["run_time"])
        self.nb_neurons_per_categories:int = int(config["Genome_NN"]["outputs_multiplicator"])
        self.output_spike_average:List[float] = []
        self.output_spike_average_summed:List[float] = []
    
    def max_spikes(neurons_output:np.ndarray, nb_classes:int) -> int:
        '''
            Return the index of the neuron with the highest output
        '''
        # 1 - split the neurons output in nb_classes parts
        neurons_output_split:np.ndarray = np.array(TOOLS.split(neurons_output, nb_classes))
        # 2 - sum the output of each part
        summed_values:np.ndarray = np.sum(neurons_output_split, axis=1)
        # 3 - return the index of the neuron with the highest output
        return np.argmax(summed_values)

    # @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def augmented(self, neurons_output:np.ndarray, nb_classes:int, spike_max:float) -> np.ndarray:
        '''
            Return the spiking rate of each neuron
        '''
        # 1 - split the neurons output in nb_classes parts
        neurons_output_split:np.ndarray = np.array(np.split(neurons_output, nb_classes), dtype=np.float32)

        # 2 - sum the output of each part
        summed_values:np.ndarray = np.sum(neurons_output_split, axis=1)

        # self.output_spike_average.extend(neurons_output) # for testing
        # self.output_spike_average_summed.extend(summed_values) # for testing

        # 3 - convert the summed values to rate -> rate = nb_of_spikes_of_neuron_categorie / (run_time * nb_neurons)
        for i in range(nb_classes):
            summed_values[i] = np.clip(summed_values[i] / spike_max, 0.0, 1.0)
        return summed_values

    def rate(self, neurons_output:np.ndarray, nb_classes:int, ratio_max_output_spike:float=1.0) -> np.ndarray:
        '''
            Return the spiking rate of each neuron
        '''
        # 1 - split the neurons output in nb_classes parts
        neurons_output_split:np.ndarray = np.array(np.split(neurons_output, nb_classes), dtype=np.float32)

        # 2 - sum the output of each part
        summed_values:np.ndarray = np.sum(neurons_output_split, axis=1)

        # self.output_spike_average.extend(neurons_output) # for testing
        # self.output_spike_average_summed.extend(summed_values) # for testing

        # 3 - convert the summed values to rate -> rate = nb_of_spikes_of_neuron_categorie / (run_time * nb_neurons)
        for i in range(nb_classes):
            # formula is -> rate = nb_of_spikes_of_neuron_categorie / (nb_neurons_per_categories * run_time * ratio_max_spikes)
            summed_values[i] = np.clip(summed_values[i] / (self.nb_neurons_per_categories * self.run_time * ratio_max_output_spike), 0.0, 1.0)
        return summed_values


    # @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    # def voltage(neurons_output_voltage:np.ndarray, voltage_min:np.ndarray, voltage_max:np.ndarray) -> np.ndarray:
    def voltage(self, neurons_output_voltage:np.ndarray, voltage_min:np.ndarray, voltage_max:np.ndarray) -> np.ndarray:
        '''
            Return the voltage output of each neuron clip/interpolate between 0 and 1
        '''
        # print("voltage_min", voltage_min, "shape", voltage_min.shape)
        # print("voltage_max", voltage_max, "shape", voltage_max.shape)
        # print("neurons_output_voltage", neurons_output_voltage, "shape", neurons_output_voltage.shape)
        for i in nb.prange(neurons_output_voltage.shape[1]):
            neurons_output_voltage[:, i] = np.interp(neurons_output_voltage[:, i], (voltage_min[i], voltage_max[i]), (0, 1))
        return neurons_output_voltage


    def coefficient(self, neurons_output:np.ndarray, coefficients:np.ndarray) -> np.ndarray:
        '''
            Return the output of the network by taking the output of each neuron and multiply it by the coefficient
        '''
        # 0 - multiply the outputs by the coefficients
        neurons_output = neurons_output * coefficients
        # print("coef neurons_output", neurons_output)

        # 1 - clip the neurons output between 0 and 1
        neurons_output = np.clip(neurons_output, 0.0, 1.0)


        # exit()

        return neurons_output

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