#!python
# cython: embedsignature=True, binding=True

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

cdef class Record_cython:
    def __init__(self, id = 0):
        self.id = id
        self.score = 0
        self.neurons = []
        self.spike_number = np.empty((0,0), dtype=np.int32)
        self.spike_time = np.empty((0,0), dtype=np.int32)

    cpdef list neurons_max_spikes_indexes(self, set neurons_types = {"output"}):
        cdef size_t i
        cdef size_t len_neurons = len(self.neurons)
        cdef np.ndarray spikes_array
        cdef list neurons_spikes = [], output_index = []
        
        for i in range(len_neurons):
            if self.neurons[i].group_name in neurons_types:
                # print("neuron.id", self.neurons[i].id, "neuron.gene_id", self.neurons[i].gene_id,"neurons.spikes_number_record", self.neurons[i].spikes_number_record)
                neurons_spikes.append(list(self.neurons[i].spikes_number_record.values()))
        spikes_array = np.array(neurons_spikes, dtype=np.int32)
        
        for i in range(<size_t>spikes_array.shape[1]):
            output_index.append(np.argmax(spikes_array[:,i]))
        return output_index

    cpdef list neurons_nb_spikes_list(self, set neurons_types = {"output"}):
        cdef size_t i
        cdef size_t len_neurons = len(self.neurons)
        cdef list neurons_spikes = [] # [nb_spikes]

        for i in range(len_neurons):
            if self.neurons[i].group_name in neurons_types:
                neurons_spikes.append(sum(list(self.neurons[i].spikes_number_record.values())))

        return neurons_spikes

    cpdef dict neurons_nb_spikes_dict(self, set neurons_types = {"output"}):
        cdef size_t i
        cdef size_t len_neurons = len(self.neurons)
        cdef dict neurons_spikes = {} # {gene_id: nb_spikes}

        for i in range(len_neurons):
            if self.neurons[i].group_name in neurons_types:
                neurons_spikes[self.neurons[i].gene_id] = sum(list(self.neurons[i].spikes_number_record.values()))
        return neurons_spikes

    cpdef list neurons_outputs_stochastic(self, size_t nb_outputs, set neurons_types = {"output"}):
        cdef size_t i,j
        cdef size_t len_neurons = len(self.neurons)
        cdef np.ndarray spikes_array
        cdef list neurons_spikes = [], output_index = []
        
        for i in range(len_neurons):
            if self.neurons[i].group_name in neurons_types:
                neurons_spikes.append(list(self.neurons[i].spikes_number_record.values()))

        spikes_array = np.array(neurons_spikes, dtype=np.float32)
        # print("spikes_array", spikes_array, "type", type(spikes_array))
        spikes_array = np.array(np.array_split(spikes_array.transpose(), nb_outputs, axis=1))
        # print("spikes_array.transpose().split", spikes_array, "type", type(spikes_array))

        spikes_array = spikes_array.sum(axis=2, keepdims=False)
        # print("np.sum(spikes_array)", spikes_array)
        # print("np.sum(spikes_array).T", spikes_array.transpose())
        spikes_array = spikes_array.transpose()

        for i in range(<size_t>len(spikes_array)):
            if np.sum(spikes_array[i]) > 0:
                spikes_array[i] /= np.sum(spikes_array[i])
            else: # case where no spike has been recorded (all zeros, avoid division by zero)
                spikes_array[i] = np.ones_like(spikes_array[i]) / len(spikes_array[i])
            # print('indexes', list(range(len(spikes_array[i]))))
            # print("probs", spikes_array[i])
            output_index.append(np.random.choice(list(range(len(spikes_array[i]))), p=spikes_array[i]))


        # print("output_index", output_index)
        return output_index

    cpdef list neurons_outputs_deterministic(self, size_t nb_outputs, set neurons_types = {"output"}):
        cdef size_t i,j
        cdef size_t len_neurons = len(self.neurons)
        cdef np.ndarray spikes_array
        cdef list neurons_spikes = [], output_index = []
        
        for i in range(len_neurons):
            if self.neurons[i].group_name in neurons_types:
                neurons_spikes.append(list(self.neurons[i].spikes_number_record.values()))

        spikes_array = np.array(neurons_spikes, dtype=np.float32)
        # print("spikes_array", spikes_array, "type", type(spikes_array))
        spikes_array = np.array(np.array_split(spikes_array.transpose(), nb_outputs, axis=1))
        # print("spikes_array.transpose().split", spikes_array, "type", type(spikes_array))

        spikes_array = spikes_array.sum(axis=2, keepdims=False)
        # print("np.sum(spikes_array)", spikes_array)
        # print("np.sum(spikes_array).T", spikes_array.transpose())
        spikes_array = spikes_array.transpose()

        for i in range(<size_t>len(spikes_array)):
            output_index.append(np.argmax(spikes_array[i]))


        # print("output_index", output_index)
        return output_index
