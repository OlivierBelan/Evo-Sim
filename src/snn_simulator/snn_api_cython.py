
import numpy as np
class SNN:
    def __init__(self, id:int = 0) -> None:
        self.id:int = id

        # 0 - Init info indexes
        self.nb_input:np.ndarray = None
        self.nb_output:np.ndarray = None
        self.nb_hidden:np.ndarray = None
        self.total_neurons:np.ndarray = None
        self.input_indexes:np.ndarray = None
        self.output_indexes:np.ndarray = None
        self.hidden_indexes:np.ndarray = None
        self.neuron_active_indexes:np.ndarray = None
        self.synapse_active_indexes:np.ndarray = None

        # 1- Init Neurons
        self.voltage_init:np.ndarray = None
        self.threshold:np.ndarray = None
        self.tau:np.ndarray = None
        self.constant_current:np.ndarray = None
        self.refractory:np.ndarray = None

        #2 - Init Synapses
        self.weight:np.ndarray = None
        self.delay:np.ndarray = None

        # 3 Extra
        # self.spike_matrix = np.empty(self.voltage_init_matrix.shape[0], dtype=np.uint8)
        # self.record.spike_number = np.zeros(self.voltage_init.shape[0], dtype=np.int32)
