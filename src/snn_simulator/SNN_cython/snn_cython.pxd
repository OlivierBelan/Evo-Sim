# from cython.cimports.libcpp import bint

import cython
import numpy as np
cimport numpy as np
np.import_array()

cdef class SNN_cython:    
    # PULBIC VARIABLE
    cdef public int id

    cdef public int nb_input
    cdef public int nb_output
    cdef public int nb_hidden
    cdef public int total_neurons
    cdef public np.ndarray input_indexes
    cdef public np.ndarray hidden_indexes
    cdef public np.ndarray output_indexes
    cdef public np.ndarray neuron_active_indexes
    cdef public np.ndarray synapse_active_indexes
    cdef public np.ndarray neuron_active_global_indexes
    cdef public np.ndarray synapse_active_global_indexes
        
    #  PRIVATE VARIABLE
    # Neurons variables
    cdef public np.ndarray voltage_init
    cdef public np.ndarray tau
    cdef public np.ndarray current
    cdef public np.ndarray threshold
    cdef public np.ndarray refractory
    cdef public np.ndarray constant_current
    # Synapses variables
    cdef public np.ndarray weight
    cdef public np.ndarray delay


    cpdef void init_network(self, 
                            dict parameter, 
                            int nb_input, 
                            int nb_hidden, 
                            int nb_output, 
                            np.ndarray input_indexes, 
                            np.ndarray hidden_indexes, 
                            np.ndarray output_indexes, 
                            # np.ndarray neuron_active_indexes, 
                            # np.ndarray synapse_active_indexes, 
                            np.ndarray neuron_active_global_indexes, 
                            # np.ndarray synapse_active_global_indexes, 
                            np.ndarray synapse_unactive_indexes, 
                            np.ndarray neuron_unactive_indexes,
                            bint is_delay,
                            bint is_refractory
                            )


