#!python
# cython: embedsignature=True, binding=True

import cython
import numpy as np
cimport numpy as np
np.import_array()


cdef class SNN_cython:    

    def __init__(self, int id = 0):
        # PULBIC VARIABLE
        self.id = id
        

    #  PUBLIC METHODE     
    cpdef void init_network(self, 
        # Parameters
        dict parameter, 
        int nb_input, 
        int nb_output, 
        int nb_hidden,
        np.ndarray[np.int32_t, ndim=1] input_indexes, 
        np.ndarray[np.int32_t, ndim=1] output_indexes, 
        np.ndarray[np.int32_t, ndim=1] hidden_indexes, 
        # np.ndarray[np.int32_t, ndim=1] neuron_active_indexes, 
        # np.ndarray[np.int32_t, ndim=2] synapse_active_indexes,
        np.ndarray[np.int32_t, ndim=1] neuron_active_global_indexes, 
        # np.ndarray[np.int32_t, ndim=2] synapse_active_global_indexes,
        np.ndarray[np.int32_t, ndim=2] synapse_unactive_indexes,
        np.ndarray[np.int32_t, ndim=1] neuron_unactive_indexes,
        bint is_delay,
        bint is_refractory
        ):

        cdef np.ndarray flat_indices, rows, cols

        # 0 - Init info indexes
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.nb_hidden = nb_hidden
        self.total_neurons = nb_input + nb_hidden + nb_output
        self.input_indexes = input_indexes
        self.output_indexes = output_indexes
        self.hidden_indexes = hidden_indexes
        # self.neuron_active_indexes = neuron_active_indexes
        # self.synapse_active_indexes = synapse_active_indexes
        # self.neuron_active_global_indexes = neuron_active_global_indexes
        # self.synapse_active_global_indexes = synapse_active_global_indexes

        weight = parameter["weight"].copy()
        threshold = parameter["threshold"].copy()
        weight[synapse_unactive_indexes[0], synapse_unactive_indexes[1]] = 0.0 # set unactive synapses to 0.0
        threshold[neuron_unactive_indexes] = 1e14 # (just in case) set very big threshold to not have any spikes on unactive neurons

        # 1- Init Neurons
        self.voltage_init = parameter["voltage"][neuron_active_global_indexes]
        self.threshold = threshold[neuron_active_global_indexes]
        self.tau = parameter["tau"][neuron_active_global_indexes]
        self.constant_current = parameter["input_current"][neuron_active_global_indexes]

        #2 - Init Synapses
        flat_indices = np.unique(neuron_active_global_indexes)
        rows, cols = np.meshgrid(flat_indices, flat_indices)
        self.weight = weight[cols, rows]
        # self.weight = parameter["weight"][cols, rows]


        # 3 - Init Time varying variables
        if is_refractory == True:
            self.refractory = parameter["refractory"][neuron_active_global_indexes]
        if is_delay == True:
            # self.delay = parameter["delay"][rows, cols]
            self.delay = parameter["delay"][cols, rows]
