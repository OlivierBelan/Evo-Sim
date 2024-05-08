import numpy as np
from typing import Dict, Tuple, List
from evo_simulator.GENERAL.Genome import Genome_NN as Genome
from evo_simulator.GENERAL.NN import NN
import TOOLS
import sys
import numba as nb



class Distance():
    def __init__(self, config_path_file:str) -> None:
        self.distances_genomes_cache:Dict[Tuple[int, int], float] = {} # key = (genome1_id, genome2_id), value = distance between genomes
        self.distances_min_genomes_cache:Dict[int, Dict[str, float]] = {} # key = genome_id, value = (key = (id/distance/distance_local), value = (closest_genome_id/distance))
        self.mean_distance:Dict[str, float] = {"global":0.0, "local":0.0}

        # Parameters to mutate
        self.config:Dict[str, Dict[str, str]] = TOOLS.config_function(config_path_file, ["Distance", "Genome_NN"])

        # Parameters coefficients
        self.parameter_coeff:float = float(self.config["Distance"]["parameter_coeff"])
        self.topology_coeff:float = float(self.config["Distance"]["topology_coeff"])
        self.network_type:str = self.config["Genome_NN"]["network_type"]


    def reset_cache(self):
        self.distances_genomes_cache:Dict[Tuple[int, int], float] = {}
        self.distances_min_genomes_cache:Dict[int, Dict[str, float]] = {}
        self.mean_distance:Dict[str, float] = {"global":0.0, "local":0.0}
        self.mean_distance_cumulator:float = 0
        self.nb_genomes_cumulator:int = 0


    def distance_genomes_list(self, genomes_1_list:List[int], genomes_2_list:List[int], populaton:Dict[int, Genome], reset_cache:bool=True) -> Dict[Tuple[int, int], float]:
        if reset_cache == True: self.reset_cache()

        genome_1, genome_2 = None, None
        closest_id_local, min_distance_local = -1, sys.float_info.max
        distance, nb_genomes, cumulative_distance = 0, 0, 0
        min_distance_dict:Dict[str, float] = None

        for id_1 in genomes_1_list:
            genome_1:Genome = populaton[id_1]
            min_distance_local:float = sys.float_info.max
            closest_id_local:int = -1

            for id_2 in genomes_2_list:
                if id_1 != id_2: # Avoid distance with itself
                    genome_2:Genome = populaton[id_2]
                    distance:float = self.distance_genomes(genome_1, genome_2)
                    cumulative_distance += distance
                    nb_genomes += 1 
                    if distance < min_distance_local:
                        min_distance_local:float = distance
                        closest_id_local:int = genome_2.id

            if closest_id_local != -1:
                min_distance_dict:Dict[str, float] = self.distances_min_genomes_cache.get(genome_1.id)
                if min_distance_dict != None:
                    min_distance_dict["id_local"] = closest_id_local
                    min_distance_dict["distance_local"] = min_distance_local
                    if min_distance_local < min_distance_dict["distance_global"]:
                        min_distance_dict["id_global"] = closest_id_local
                        min_distance_dict["distance_global"] = min_distance_local
                else:
                    self.distances_min_genomes_cache[genome_1.id] = {"id_global":closest_id_local, "distance_global":min_distance_local, "id_local":closest_id_local, "distance_local":min_distance_local}
                    


        nb_genomes:int = nb_genomes if nb_genomes != 0 else 1 # Avoid division by zero
        self.mean_distance_cumulator += cumulative_distance
        self.nb_genomes_cumulator += nb_genomes
        self.mean_distance["local"] = cumulative_distance / nb_genomes
        self.mean_distance["global"] = self.mean_distance_cumulator / self.nb_genomes_cumulator

        return self.distances_genomes_cache


    def distance_genomes(self, genome_1:Genome, genome_2:Genome) -> float:
        # Check if distance has been already computed
        if self.distances_genomes_cache.get((genome_1.id, genome_2.id)) is not None:
            return self.distances_genomes_cache[(genome_1.id, genome_2.id)]

        # 1 - Common neurons/synapses indexes
        synapses_actives_concatenate:np.ndarray = self.get_synapses_actives_concatenate_jit(genome_1.nn.synapses_actives_indexes, genome_2.nn.synapses_actives_indexes)
        synapses_actives_unique, unique_indices = np.unique(synapses_actives_concatenate, return_index=True, axis=0)
        common_neurons_indexes, common_synapses_indexes = self.get_common_indexes_jit(genome_1.nn.neuron_actives_indexes, genome_2.nn.neuron_actives_indexes, synapses_actives_unique, unique_indices)


        # 2 - Parameters distance
        if self.network_type == "SNN":
            parameter_distance, topology_distance = self.euclidean_distance_genome_snn_jit(
                # 2.1- Parameters distance
                genome_1.nn.parameters["voltage"], genome_2.nn.parameters["voltage"],
                genome_1.nn.parameters["threshold"], genome_2.nn.parameters["threshold"],
                genome_1.nn.parameters["tau"], genome_2.nn.parameters["tau"],
                genome_1.nn.parameters["input_current"], genome_2.nn.parameters["input_current"],
                # genome_1.nn.parameters["refractory"], genome_2.nn.parameters["refractory"],
                genome_1.nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]], genome_2.nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]],
                # 2.2- Topology distance
                genome_1.nn.neurons_status, genome_2.nn.neurons_status,
                genome_1.nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]], genome_2.nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]],
                # 2.3- Common indexes
                common_neurons_indexes
                )
        elif self.network_type == "ANN":
            parameter_distance, topology_distance = self.euclidean_distance_genome_ann_jit(
                # 2.1- Parameters distance
                genome_1.nn.parameters["bias"], genome_2.nn.parameters["bias"],
                genome_1.nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]], genome_2.nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]],
                # 2.2- Topology distance
                genome_1.nn.neurons_status, genome_2.nn.neurons_status,
                genome_1.nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]], genome_2.nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]],
                # 2.3- Common indexes
                common_neurons_indexes
                )

        # 3- Ecludien (weighted) distance
        ecludien_distance:float = (parameter_distance * self.parameter_coeff) + (topology_distance * self.topology_coeff)

        # # Debug
        # self.debug_print(genome_1, genome_2, parameter_distance, common_neurons_indexes, common_synapses_indexes, topology_distance, ecludien_distance)

        # Sava distance in cache
        self.distances_genomes_cache[(genome_1.id, genome_2.id)] = ecludien_distance
        self.distances_genomes_cache[(genome_2.id, genome_1.id)] = ecludien_distance
        return ecludien_distance

    @staticmethod
    def get_common_indexes(nn_1:NN, nn_2:NN, only_hidden:bool = True) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Return the common neurons and synapses indexes between two NN
        first return: common_neurons_indexes
        second return: common_synapses_indexes
        '''
        synapses_actives_concatenate:np.ndarray = Distance.get_synapses_actives_concatenate_jit(nn_1.synapses_actives_indexes, nn_2.synapses_actives_indexes)
        synapses_actives_unique, unique_indices = np.unique(synapses_actives_concatenate, return_index=True, axis=0)
        if only_hidden == True:
            return Distance.get_common_indexes_jit(nn_1.hiddens["neurons_indexes_active"], nn_2.hiddens["neurons_indexes_active"], synapses_actives_unique, unique_indices)
        return Distance.get_common_indexes_jit(nn_1.neuron_actives_indexes, nn_2.neuron_actives_indexes, synapses_actives_unique, unique_indices)


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def get_synapses_actives_concatenate_jit(synapses_actives_indexes_1:Tuple[np.ndarray, np.ndarray], synapses_actives_indexes_2:Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        synapses_actives_indexes_1_T:np.ndarray = np.empty((2, len(synapses_actives_indexes_1[0])), dtype=np.int32)
        synapses_actives_indexes_1_T[0] = synapses_actives_indexes_1[0]
        synapses_actives_indexes_1_T[1] = synapses_actives_indexes_1[1]

        
        synapses_actives_indexes_2_T:np.ndarray = np.empty((2, len(synapses_actives_indexes_2[0])), dtype=np.int32)
        synapses_actives_indexes_2_T[0] = synapses_actives_indexes_2[0]
        synapses_actives_indexes_2_T[1] = synapses_actives_indexes_2[1]

        synapses_actives_indexes_1_T:np.ndarray = synapses_actives_indexes_1_T.T
        synapses_actives_indexes_2_T:np.ndarray = synapses_actives_indexes_2_T.T

        synapses_actives_concatenate:np.ndarray = np.concatenate((synapses_actives_indexes_1_T, synapses_actives_indexes_2_T), axis=0)
        return synapses_actives_concatenate

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def get_synapses_actives_concatenate_2(synapses_actives_indexes_1:Tuple[np.ndarray, np.ndarray], synapses_actives_indexes_2:Tuple[np.ndarray, np.ndarray]):

        # Concatenate the arrays along the first axis.
        return np.concatenate((synapses_actives_indexes_1.T, synapses_actives_indexes_2.T), axis=0)


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def get_common_indexes_jit(neuron_actives_indexes_1:np.ndarray, neuron_actives_indexes_2:np.ndarray, synapses_actives_unique:np.ndarray, unique_indices:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        common_neurons_indexes:np.ndarray = np.unique(np.concatenate((neuron_actives_indexes_1, neuron_actives_indexes_2)))
        common_synapses_indexes:np.ndarray = synapses_actives_unique[np.argsort(unique_indices)].T
        return common_neurons_indexes, common_synapses_indexes

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def euclidean_distance_genome_snn_jit(
        # 1- Parameters distance
        voltage_1:np.ndarray, voltage_2:np.ndarray, 
        threshold_1:np.ndarray, threshold_2:np.ndarray, 
        tau_1:np.ndarray, tau_2:np.ndarray, 
        input_current_1:np.ndarray, input_current_2:np.ndarray, 
        # refractory_1:np.ndarray, refractory_2:np.ndarray, 
        weight_1:np.ndarray, weight_2:np.ndarray,
        # 2- Topology distance
        neuron_status_1:np.ndarray, neuron_status_2:np.ndarray,
        synapse_status_1:np.ndarray, synapse_status_2:np.ndarray,
        # 3- Common indexes
        common_neurons_indexes:np.ndarray
        ) -> float:
        return (
        # 1- Parameters distance
        # Voltage
        (np.linalg.norm(voltage_1[common_neurons_indexes]-voltage_2[common_neurons_indexes])
        # Threshold
        + np.linalg.norm(threshold_1[common_neurons_indexes]-threshold_2[common_neurons_indexes])
        # Tau
        + np.linalg.norm(tau_1[common_neurons_indexes]-tau_2[common_neurons_indexes])
        # Input current
        + np.linalg.norm(input_current_1[common_neurons_indexes]-input_current_2[common_neurons_indexes]) 
        # Refractory
        # + np.linalg.norm(refractory_1[common_neurons_indexes]-refractory_2[common_neurons_indexes]) 
        # Weight
        + np.linalg.norm(weight_1-weight_2)),
        # 2- Topology distance
        # Neuron status
        (np.linalg.norm(neuron_status_1[common_neurons_indexes].astype(np.float32)-neuron_status_2[common_neurons_indexes].astype(np.float32))
        # Synapse status
        + np.linalg.norm(synapse_status_1.astype(np.float32)-synapse_status_2.astype(np.float32)))
        )

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def euclidean_distance_genome_ann_jit(
        # 1- Parameters distance
        bias_1:np.ndarray, bias_2:np.ndarray, 
        weight_1:np.ndarray, weight_2:np.ndarray,
        # 2- Topology distance
        neuron_status_1:np.ndarray, neuron_status_2:np.ndarray,
        synapse_status_1:np.ndarray, synapse_status_2:np.ndarray,
        # 3- Common indexes
        common_neurons_indexes:np.ndarray
        ) -> float:
        return (
        # 1- Parameters distance
        # Bias
        (np.linalg.norm(bias_1[common_neurons_indexes]-bias_2[common_neurons_indexes])
        # Weight
        + np.linalg.norm(weight_1-weight_2)),
        # 2- Topology distance
        # Neuron status
        (np.linalg.norm(neuron_status_1[common_neurons_indexes].astype(np.float32)-neuron_status_2[common_neurons_indexes].astype(np.float32))
        # Synapse status
        + np.linalg.norm(synapse_status_1.astype(np.float32)-synapse_status_2.astype(np.float32)))
        )

    # @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    # def euclidean_distance_matrix(matrix_1:np.ndarray, matrix_2:np.ndarray) -> np.ndarray:
    #     # 1- Create distances matrix
    #     distances:np.ndarray = np.zeros((matrix_1.shape[0], matrix_2.shape[0]), dtype=np.float32)

    #     # 2- Compute euclidean distances
    #     for i in range(matrix_1.shape[0]):
    #         for j in range(matrix_2.shape[0]):
    #             distances[i, j] = np.linalg.norm(matrix_1[i]-matrix_2[j])

    #     return distances
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def euclidean_distance_matrix(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
        diff = matrix_1[:, np.newaxis, :] - matrix_2[np.newaxis, :, :]
        distances = np.sqrt((diff**2).sum(axis=-1))
        return distances

    @staticmethod
    def cosine_distance_array_jit(array_1:np.ndarray, array_2:np.ndarray) -> float:
        a:np.ndarray = array_1.flatten() + 1e-8 # Add 1e-8 to avoid division by zero
        b:np.ndarray = array_2.flatten() + 1e-8 # Add 1e-8 to avoid division by zero
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def euclidean_distance_array_jit(array_1:np.ndarray, array_2:np.ndarray) -> float:
        return np.linalg.norm(array_1-array_2)

    def debug_print(self, genome_1:Genome, genome_2:Genome, parameter_distance:float, common_neurons_indexes:np.ndarray, common_synapses_indexes:np.ndarray, topology_distance:float, ecludien_distance:float) -> None:
        print("genome_1.neuron_actives_indexes", genome_1.nn.neuron_actives_indexes)
        print("genome_2.neuron_actives_indexes", genome_2.nn.neuron_actives_indexes)
        print("common_neurons_indexes", common_neurons_indexes)
        print("genome_1.nn.synapses_actives_indexes:\n", genome_1.nn.synapses_actives_indexes[0], '\n', genome_1.nn.synapses_actives_indexes[1])
        print("genome_2.nn.synapses_actives_indexes:\n", genome_2.nn.synapses_actives_indexes[0], '\n', genome_2.nn.synapses_actives_indexes[1])
        print("common_synapses_indexes:\n",common_synapses_indexes)
        print("genome_1.nn.neurons_status[common_neurons_indexes]\n", genome_1.nn.neurons_status[common_neurons_indexes])
        print("genome_2.nn.neurons_status[common_neurons_indexes]\n", genome_2.nn.neurons_status[common_neurons_indexes])
        print("genome_1.nn.neurons_status[common_synapses_indexes[0], common_synapses_indexes[1]]\n", genome_1.nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]])
        print("genome_2.nn.neurons_status[common_synapses_indexes[0], common_synapses_indexes[1]]\n", genome_2.nn.synapses_status[common_synapses_indexes[0], common_synapses_indexes[1]])
        print("genome_1.nn.param[weight][common_synapses_indexes[0], common_synapses_indexes[1]]\n", genome_1.nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]])
        print("genome_2.nn.param[weight][common_synapses_indexes[0], common_synapses_indexes[1]]\n", genome_2.nn.parameters["weight"][common_synapses_indexes[0], common_synapses_indexes[1]])

        print("parameter_distance", parameter_distance)
        print("topology_distance", topology_distance)
        print("ecludien_distance", ecludien_distance)
