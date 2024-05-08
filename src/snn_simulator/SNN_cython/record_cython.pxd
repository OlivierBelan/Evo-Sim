cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef class Record_cython:
    cdef public size_t id
    cdef public int score
    cdef public list neurons
    cdef public np.ndarray spike_number
    cdef public np.ndarray spike_time
    cpdef list neurons_max_spikes_indexes(self, set neurons_types = *)
    cpdef list neurons_nb_spikes_list(self, set neurons_types = *)
    cpdef dict neurons_nb_spikes_dict(self, set neurons_types = *)
    cpdef list neurons_outputs_stochastic(self, size_t nb_outputs, set neurons_types = *)
    cpdef list neurons_outputs_deterministic(self, size_t nb_outputs, set neurons_types = *)