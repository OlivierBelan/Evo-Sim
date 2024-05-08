#!python
# cython: embedsignature=True, binding=True

cimport cython
from snn_cython cimport SNN_cython as SNN
import numpy as np
cimport numpy as np
np.import_array()
from tools_cython cimport get_time
cimport libc.math as math
from cython.parallel import prange
import random

from libc.stdlib cimport rand
from libc.stdlib cimport RAND_MAX

cdef class Runner_RL_cython:


    cpdef void init(self, list networks_list, size_t run_time = 100, size_t run_time_margin = 0, float dt = 1.0, size_t nb_episode = 1, bint online=False, str neuron_reset="voltage_reset", list record_layer=["output"], bint disable_output_threshold = False, str decay_method="lif", bint is_delay=False, bint is_refractory=False, size_t delay_max = 100, set record_decoding_method = {"spike"}):
        self.dt = dt # time step
        self.run_time_original = run_time # simulation time
        self.run_time_margin = run_time + run_time_margin
        delay_max = <size_t>max(0, <long>delay_max) # in case its negative (even its type of size_t)
        self.delay_max = delay_max + 1 # max delay
        self.run_time_delay_max = self.run_time_margin + delay_max  # simulation time + margin + max delay
        self.online = online
        self.is_threshold_reset = True if neuron_reset == "threshold_reset" else False
        self.is_combinatorial_encoder_decoder_init = False


        self.record_layer = record_layer
        self.disable_output_threshold = disable_output_threshold
        self.decay_method = True if decay_method == "lif" else False
        self.is_delay = is_delay
        self.is_refractory = is_refractory

        # private variables
        self.input_spike_amplitude = 100.0 # spike amplitude
        self.first_time_step = 0

        self.nb_episode = nb_episode # nb_episode size
        self.population_len = len(networks_list)
                
        self.record_spikes = {}
        self.record_voltages = {}
        self.record_augmented = {}
        self.nb_neurons = len((<SNN>networks_list[0]).voltage_init)

        self.networks_list = networks_list
        self.init_network(self.networks_list)

        self.is_record_spike = "spike" in record_decoding_method
        self.is_record_voltage = "voltage" in record_decoding_method
        self.is_record_augmented = "augmented" in record_decoding_method

        if self.is_augmented_decoder == True:
            self.spike_max_time_step = <int>max(np.rint(self.spikes_max/(self.run_time_margin-1)), 1)
            if self.spike_distribution_importance > 0:
                self.spikes_importance = self.augmented_linear_spike_importance(self.spike_distribution_importance, self.run_time_margin-1, is_descending=self.linear_spike_importance_type)
            else:
                self.spikes_importance = np.zeros(self.run_time_margin-1, dtype=np.float32)
   

    #  PUBLIC METHODS
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void run(self, str encoder_type, np.ndarray inputs_data, size_t spike_rate = 2, float spike_amplitude = 100.0, size_t max_nb_spikes = 3, int reduce_noise = 100, int combinatorial_factor = 1, size_t combinaison_size=1, size_t combinaison_size_max=1, float combinatorial_combinaison_noise=0.0, bint combinatorial_roll = True):
        # 0 - Init inputs data and networks
        if self.init_data_set(encoder_type, inputs_data, spike_rate, spike_amplitude, max_nb_spikes, reduce_noise, combinatorial_factor, combinaison_size, combinaison_size_max, combinatorial_combinaison_noise, combinatorial_roll) == -1: return
        self.run_C()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void run_C(self):
        
        # 0 - Init networks for this step
        self.init_network_step()

        for self.time_step in range(self.run_time_margin):
            
            # 1 - Input Data in Networks
            self.input_data_to_network(self.time_step) # init network

            # 2 - Update Networks with LIF and Apply Refractory + Save Spike
            if self.is_refractory == True:
                self.LIF_update_voltage_refractory(self.time_step) # Check Spike + Update (LIF) Voltage
            else:
                self.LIF_update_voltage_no_refractory(self.time_step) # Check Spike + Update (LIF) Voltage
            

            # 3 - Update Networks with Current Kernel (Weights + Delay)
            if self.is_delay == True:
                self.current_kernel_update_voltage_delay(self.time_step) # Refractory + Weight + Delay + Update Voltage
            else:
                self.current_kernel_update_voltage_no_delay(self.time_step) # Refractory + Weight + Update Voltage

            # 4 - Augmented Decoder Update
            if self.is_augmented_decoder == True:
                self.augmented_update(self.time_step)

        # 5 - Apply importance in Augmented Decoder
        if self.is_augmented_decoder == True:
            self.augmented_importance(self.networks_list)

        # 6 - Record Networks (Spikes)
        self.record()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int init_data_set(self, str encoder_type, np.ndarray inputs_data, size_t rate, float spike_amplitude, size_t max_nb_spikes, int reduce_noise, int combinatorial_factor, size_t combinaison_size=1, size_t combinaison_size_max=1, float combinatorial_combinaison_noise=0.0, bint combinatorial_roll = True):
        if encoder_type == "poisson":
            self.poisson_encoder(inputs_data, rate, spike_amplitude, max_nb_spikes)
        elif encoder_type == "binomial":
            self.binomial_encoder(inputs_data, spike_amplitude, max_nb_spikes, reduce_noise)
        elif encoder_type == "exact":
            self.exact_encoder(inputs_data, spike_amplitude, max_nb_spikes)
        elif encoder_type == "rate":
            self.rate_encoder(inputs_data, spike_amplitude)
        elif encoder_type == "combinatorial":
            self.combinatorial_encoder(inputs_data, combinatorial_factor, combinaison_size, combinaison_size_max, combinatorial_combinaison_noise, combinatorial_roll, spike_amplitude)
        elif encoder_type == "latency":
            self.raw_encoder(inputs_data, spike_amplitude)
        else:
            print("Error: encoder_type not found")
            return -1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void poisson_encoder(self, np.ndarray[np.float64_t, ndim=3] inputs_data, size_t rate, float spike_amplitude = 100.0, size_t max_nb_spikes = 3):
        self.input_spike_amplitude = spike_amplitude
        cdef size_t run_time = self.run_time_original
        self.input_data = np.zeros([inputs_data.shape[0], inputs_data.shape[1], inputs_data.shape[2], <int>self.run_time_original], dtype=np.float32)
        cdef long i, j, k, m, len_indexes
        cdef np.ndarray[np.int_t, ndim=1] poisson
        cdef int[:] indexes
        max_nb_spikes = max_nb_spikes if max_nb_spikes < run_time else run_time

        for i in range(inputs_data.shape[0]):
            for m in range(inputs_data.shape[1]):
                for j in range(inputs_data.shape[2]):
                    poisson = np.random.poisson(lam=inputs_data[i, m, j]*run_time, size=rate)
                    indexes = self.n_most_frequent_np(poisson, max_nb_spikes)
                    len_indexes = len(indexes)
                    for k in range(len_indexes):
                        if indexes[k] < (<int>run_time):
                            self.input_data[i, m, j, indexes[k]] = spike_amplitude
        # Add padding (if needed)
        self.add_padding_input_data()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void binomial_encoder(self, np.ndarray[np.float64_t, ndim=3] inputs_data, float spike_amplitude = 100.0, size_t max_nb_spikes = 3, int reduce_noise = 100):
        self.input_spike_amplitude = spike_amplitude
        cdef size_t run_time = self.run_time_original - 1
        self.input_data = np.zeros([inputs_data.shape[0], inputs_data.shape[1], inputs_data.shape[2], <int>self.run_time_original], dtype=np.float32)
        cdef long i, j, k, m, len_indexes
        cdef np.ndarray[np.int_t, ndim=1] binomial
        cdef int[:] indexes
        max_nb_spikes = max_nb_spikes if max_nb_spikes < run_time else run_time

        for i in range(inputs_data.shape[0]):
            for m in range(inputs_data.shape[1]):
                for j in range(inputs_data.shape[2]):
                    binomial = np.random.binomial(n=run_time, p=inputs_data[i, m, j], size=max_nb_spikes + reduce_noise)
                    indexes = self.n_most_frequent_np(binomial, max_nb_spikes)
                    len_indexes = len(indexes)
                    for k in range(len_indexes):
                        self.input_data[i, m, j, indexes[k]] = spike_amplitude
        # Add padding (if needed)
        self.add_padding_input_data()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void rate_encoder(self, np.ndarray[np.float64_t, ndim=3] inputs_data, float spike_amplitude = 100.0):
        self.input_spike_amplitude = spike_amplitude
        cdef size_t run_time = self.run_time_original
        self.input_data = np.zeros([inputs_data.shape[0], inputs_data.shape[1], inputs_data.shape[2], <int>self.run_time_original], dtype=np.float32)
        cdef size_t i, j, k, m
        # cdef np.ndarray[np.int_t, ndim=1] binomial
        cdef size_t len_0 = inputs_data.shape[0]
        cdef size_t len_1 = inputs_data.shape[1]
        cdef size_t len_2 = inputs_data.shape[2]
        cdef double[:, :, :] inputs_data_view = inputs_data

        for i in range(len_0):
            for m in range(len_1):
                for j in range(len_2):
                    # binomial = np.random.binomial(n=1, p=inputs_data[i, m, j], size=<long>run_time)
                    for k in range(run_time):
                        # if binomial[k] == 1:
                        # if random.random() < inputs_data_view[i, m, j]:
                        # if self.bernoulli_trial(inputs_data_view[i, m, j]) == 1:
                        if rand() < (inputs_data_view[i, m, j] * RAND_MAX): # bernoulli_trial (binomial) C version
                            self.input_data[i, m, j, k] = spike_amplitude
        # Add padding (if needed)
        self.add_padding_input_data()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void exact_encoder(self, np.ndarray[np.float64_t, ndim=3] inputs_data, float spike_amplitude = 100.0, int max_nb_spikes = 3):
        self.input_spike_amplitude = spike_amplitude
        cdef double[:, :, :] inputs_data_view = inputs_data * self.run_time_original
        self.input_data = np.zeros([inputs_data.shape[0], inputs_data.shape[1], inputs_data.shape[2], <int>self.run_time_original], dtype=np.float32)
        cdef int shift_max, shift_min, middle
        cdef int shift_max_operation = (max_nb_spikes//2) +1
        cdef int shift_min_operation = (max_nb_spikes//2)
        cdef int i, j, k

        for i in range(inputs_data.shape[0]):
            for k in range(inputs_data.shape[1]):
                for j in range(inputs_data.shape[2]):
                    middle = <int>math.ceil(inputs_data_view[i, k, j]) - 1
                    shift_min = 0 if 0 > (middle - shift_min_operation) else (middle - shift_min_operation)
                    shift_max = <int>self.run_time_original if (<int>self.run_time_original) < (middle + shift_max_operation) else (middle + shift_max_operation)
                    self.input_data[i, k, j, shift_min:shift_max] = spike_amplitude
        # Add padding (if needed)
        self.add_padding_input_data()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void raw_encoder(self, np.ndarray inputs_data, float spike_amplitude = 100.0):
        self.input_data = inputs_data * spike_amplitude
        # Add padding (if needed)
        self.add_padding_input_data()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef void combinatorial_encoder(self, np.ndarray inputs_data, int combinatorial_factor = 1, size_t combinaison_size=1, size_t combinaison_size_max=1, float combinaison_noise=0.0, bint combinatorial_roll=True, float spike_amplitude = 100.0):
        self.combinatorial_encoder_init(combinatorial_factor, combinaison_size, combinaison_size_max, combinatorial_roll)

        self.input_data = np.zeros([inputs_data.shape[0], inputs_data.shape[1], inputs_data.shape[2], <int>self.run_time_margin], dtype=np.float32) # add padding by using run_time_margin
        cdef int encoder_size = self.combinatorial_encoder_table.shape[0] - 1
        # print("inputs_data", inputs_data, "shape", np.shape(inputs_data))
        inputs_data = np.rint(np.interp(inputs_data, [0, 1], [0, encoder_size])).astype(np.int32)
        cdef int[:, :, :] inputs_data_view = inputs_data
        cdef int i, j, k
        cdef int index

        for i in range(inputs_data.shape[0]):
            for j in range(inputs_data.shape[1]):
                for k in range(inputs_data.shape[2]):
                    index = inputs_data_view[i, j, k]
                    if combinaison_noise != 0.0:
                        index = max(0, min(np.round(np.random.normal(index, combinaison_noise)).astype(np.int32), encoder_size))
                    for m in range(self.run_time_original): # add padding by using run_time_original
                        self.input_data[i, j, k, m] = self.combinatorial_encoder_table_view[index, m] * spike_amplitude

        # print("combinatorial_encoder_table_view\n", np.array(self.combinatorial_encoder_table_view), "shape", np.shape(self.combinatorial_encoder_table_view))
        # print("inputs_data\n", inputs_data, "shape", np.shape(inputs_data))
        # print("input_data encoded\n", np.array(self.input_data), "shape", np.shape(self.input_data))
        # print("spike_amplitude", spike_amplitude)
        # print("combinaison_noise:", combinaison_noise)
        # exit()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void combinatorial_encoder_init(self, int combinatorial_factor=1, size_t combinaison_size=1, size_t combinaison_size_max=1, bint combinatorial_roll=True):
        if self.is_combinatorial_encoder_decoder_init == True: return
        cdef size_t i = 0
        cdef str binary_repr
        cdef int first_one_index
        cdef size_t run_len = self.run_time_original // combinatorial_factor
        cdef size_t combinatorial_encoder_len = min(2**run_len, combinaison_size_max) if run_len < 50 else min(2**50, combinaison_size_max) # to avoid memory error
        cdef np.ndarray[np.float32_t, ndim=1] row_sums 
        cdef long[:] sorted_indices, indices_to_keep

        # Binary Encoder -> Binary Order
        self.combinatorial_encoder_table = np.empty((combinatorial_encoder_len, run_len), dtype=np.float32)
        for i in range(combinatorial_encoder_len):
            binary_repr = format(i, f'0{run_len}b')
            self.combinatorial_encoder_table[i] = [float(x) for x in binary_repr]

        # self.combinatorial_encoder_table = self.combinatorial_encoder_table[np.lexsort((self.combinatorial_encoder_table @ (2**np.arange(run_len)), self.combinatorial_encoder_table.sum(axis=1)))]

        # Pruning (sort by the number of 1 and keep the n combinaison_size)
        row_sums = np.sum(self.combinatorial_encoder_table, axis=1)
        sorted_indices = np.lexsort((-np.arange(len(self.combinatorial_encoder_table)), -row_sums))[::-1]
        indices_to_keep = sorted_indices[:combinaison_size]
        self.combinatorial_encoder_table = self.combinatorial_encoder_table[indices_to_keep]

        # Repeat (if possible and asked)
        if combinatorial_factor > 1:
            self.combinatorial_encoder_table = np.repeat(self.combinatorial_encoder_table, combinatorial_factor, axis=1)
      
        # Sort (by the first 1 index)
        self.combinatorial_encoder_table = np.array(sorted(self.combinatorial_encoder_table, key=self.find_first_one_index, reverse=True))

        # Roll (if possible and asked)
        if combinatorial_roll == True:
            first_one_index = np.argmax(np.any(self.combinatorial_encoder_table==1, axis=0))
            self.combinatorial_encoder_table = np.roll(self.combinatorial_encoder_table, -first_one_index)

        # # Nothing
        # self.combinatorial_encoder_table = np.repeat(self.combinatorial_encoder_table, combinatorial_factor, axis=1)[:combinaison_size]

        self.combinatorial_encoder_table_view = self.combinatorial_encoder_table
        self.is_combinatorial_encoder_decoder_init = True
        # print("combinatorial_encoder\n", self.combinatorial_encoder_table, "shape", np.shape(self.combinatorial_encoder_table), "nb_one", np.sum(self.combinatorial_encoder_table))
        # exit()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef int find_first_one_index(self, float[:] row):
        # try:
        #     return np.where(row == 1)[0][0]
        # except IndexError:
        #     return len(row)
        # print("row", row, "shape", np.shape(row), "type", type(row))
        # exit()
        # cdef long[:] indices = np.where(row != 0)[0]
        # return <int>indices[0] if len(indices) > 0 else len(row)
        cdef size_t i, row_len
        row_len = len(row)
        for i in range(row_len):
            if row[i] != 0:
                return i
        return row_len

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add_padding_input_data(self):
        # print("1 before padding - self.input_data\n", np.array(self.input_data[:2]), "shape", np.shape(self.input_data))
        if self.run_time_margin > self.run_time_original:
            self.input_data = np.pad(self.input_data, ((0, 0), (0, 0), (0, 0), (0, <int>(self.run_time_margin - self.run_time_original))), 'constant')
        # print("2 after padding  - self.input_data\n", np.array(self.input_data[:2]), "shape", np.shape(self.input_data))


    # PRIVATES METHODS
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void input_data_to_network(self, size_t current_time):
        self.voltage[:, :, self.input_indexes, current_time] += self.input_data[:, :, :, current_time]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void LIF_update_voltage_refractory(self, size_t current_time):
        cdef bint index = 1 if current_time + 1 < self.run_time_delay_max else 0

        # CYTHON view version
        self.voltage_sub_view = self.voltage_view[:, :, :, current_time]
        self.voltage_sub_view_next = self.voltage_view[:, :, :, current_time + 1]
        self.spike_state_view_sub = self.spike_state_view[:, :, :, current_time]
        self.spike_state_sub = self.spike_state[:, :, :, current_time]
        cdef size_t i, j, k
        cdef float threshold, voltage_next

        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                for k in range(self.nb_neurons):

                    # 1 - Refractory Check (beacause of the delay spikes)
                    if self.refractory_active_view[i, j, k] > 0:
                        self.voltage_sub_view[i, j, k] = self.voltage_reset_view[j, k] # not working/perfect in case if there is a delay weight add
                        continue
                    
                    # 2 - Check/Save Spike and Check/Update Refractory
                    threshold = self.threshold_view[j, k] # cython (little) optimization
                    if self.voltage_sub_view[i, j, k] > threshold:
                        self.spike_state_view_sub[i, j, k] = 1
                        self.refractory_active_view[i, j, k] = self.refractory_view[j, k]
                        if index == 1:
                            if self.is_threshold_reset == True:
                                self.voltage_sub_view_next[i, j, k] = self.voltage_sub_view_next[i, j, k] - threshold
                            else:
                                self.voltage_sub_view_next[i, j, k] = self.voltage_reset_view[j, k]

                    # 3 - Update Voltage for next time step -> LIF (Leaky Integrate and Fire)
                    else: 
                        if index == 1:
                            # 3 - LIF
                            # self.voltage_sub_view_next[i, j, k] += self.voltage_sub_view[i, j, k]
                            voltage_next = self.voltage_sub_view_next[i, j, k] + self.voltage_sub_view[i, j, k]
                            if self.decay_method == True: #  Voltage = Voltage + (-Voltage + Constant Current * Resistance) / Tau * dt) here Resistance = 1 and dt = 1
                                self.voltage_sub_view_next[i, j, k] = voltage_next + (-voltage_next + self.constant_current_view[j, k]) / self.tau_view[j, k] * self.dt
                            else: # V * leak * dt + current
                                self.voltage_sub_view_next[i, j, k] =  voltage_next * self.tau_view[j, k] * self.dt + self.constant_current_view[j, k]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void LIF_update_voltage_no_refractory(self, size_t current_time):
        cdef bint index = 1 if current_time + 1 < self.run_time_delay_max else 0

        # CYTHON view version
        self.voltage_sub_view = self.voltage_view[:, :, :, current_time]
        self.voltage_sub_view_next = self.voltage_view[:, :, :, current_time + 1]
        self.spike_state_view_sub = self.spike_state_view[:, :, :, current_time]
        self.spike_state_sub = self.spike_state[:, :, :, current_time]
        cdef size_t i, j, k
        cdef float voltage, threshold, voltage_next

        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                for k in range(self.nb_neurons):
                    voltage = self.voltage_sub_view[i, j, k]
                    threshold = self.threshold_view[j, k]

                    # 1 - Check/Save Spike
                    if voltage > threshold:
                        self.spike_state_view_sub[i, j, k] = 1

                        if index == 1:
                            if self.is_threshold_reset == True:
                                self.voltage_sub_view_next[i, j, k] = voltage - threshold
                            else:
                                self.voltage_sub_view_next[i, j, k] = self.voltage_reset_view[j, k]

                    # 2 - Update Voltage for next time step -> LIF (Leaky Integrate and Fire)
                    else:
                        if index == 1:
                            # 3 - LIF
                            voltage_next = self.voltage_sub_view_next[i, j, k] + voltage
                            if self.decay_method == True: # Voltage = Voltage + (-Voltage + Constant Current * Resistance) / Tau * dt) here Resistance = 1 and dt = 1
                                self.voltage_sub_view_next[i, j, k] =  voltage_next + (-voltage_next + self.constant_current_view[j, k]) / self.tau_view[j, k] * self.dt
                            else: # V * leak * dt + current
                                self.voltage_sub_view_next[i, j, k] =  voltage_next * self.tau_view[j, k] * self.dt + self.constant_current_view[j, k]

                            # self.voltage_sub_view_next[i, j, k] += voltage # add the previous voltage to the new voltage (important to keep this form for the delay weights)
                            # self.voltage_sub_view_next[i, j, k] = self.voltage_sub_view_next[i, j, k] + (-self.voltage_sub_view_next[i, j, k] + self.constant_current_view[j, k]) / self.tau_view[j, k] * self.dt

    @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef void current_kernel_update_voltage_delay(self, size_t current_time):

        # 1 - Filter the weight with the spike (active) state and create the current_kernel (weight)
        # cdef np.ndarray spike_reshape = self.spike_state_sub.reshape(self.spike_state_sub.shape[0], self.spike_state_sub.shape[1], self.spike_state_sub.shape[2], -1)
        # self.current_kernel = np.multiply(spike_reshape, self.weight)
        # up or this below (equivalent):
        # self.current_kernel = self.weight * self.spike_state_sub.reshape(self.spike_state_sub.shape[0], self.spike_state_sub.shape[1], self.spike_state_sub.shape[2], -1)
        self.current_kernel = self.weight * self.spike_state_sub[: ,: ,:, np.newaxis]


        # 2 - Filter the current_kernel with the delay and create the current_kernel_delay
        if self.is_delay == True:
            self.current_kernel_indexes = np.array(np.where(self.current_kernel != 0))
            self.current_kernel_delay_indexes = np.empty(np.shape(self.current_kernel_indexes), dtype=np.int32)
            self.current_kernel_delay_indexes[:-1] = self.current_kernel_indexes[[0, 1, 3]]
            self.current_kernel_delay_indexes[-1] = self.delay[tuple(self.current_kernel_indexes[1:])]

            self.current_kernel_delay = np.zeros(np.shape(self.current_kernel)[:-1] + (self.delay_max,))
            # self.current_kernel_delay = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.delay_max), dtype=np.float32)
            np.add.at(self.current_kernel_delay, tuple(self.current_kernel_delay_indexes), self.current_kernel[tuple(self.current_kernel_indexes)])
        else:
            self.current_kernel_delay = np.reshape(np.sum(self.current_kernel, axis=2), np.shape(self.current_kernel)[:-1] + (self.delay_max,))
            # self.current_kernel_delay = np.reshape(np.sum(self.current_kernel, axis=2), (self.nb_episode, self.nb_networks, self.nb_neurons, self.delay_max))

        # print("weight:\n", self.weight, "shape:", np.shape(self.weight))
        # print("current_kernel\n:", self.current_kernel, "shape:", np.shape(self.current_kernel))
        # print("sum(current_kernel)\n:", np.reshape(np.sum(self.current_kernel, axis=2),np.shape(self.current_kernel)[:-1] + (self.delay_max,)) , "shape:", np.shape(np.reshape(np.sum(self.current_kernel, axis=2),np.shape(self.current_kernel)[:-1] + (self.delay_max,))))
        # print("current_kernel_delay:\n", self.current_kernel_delay, "shape:", np.shape(self.current_kernel_delay))
        # print("refractory_active:\n", self.refractory_active, "shape:", np.shape(self.refractory_active))
        # print("delay:\n", self.delay, "shape:", np.shape(self.delay))
        # exit()

        # 3 - Update the voltage with the current_kernel_delay
        cdef size_t start = min(current_time+1, self.run_time_delay_max)
        cdef size_t end = min(start+self.delay_max, self.run_time_delay_max)
        self.voltage[:, :, :, start:end] += self.current_kernel_delay[:, :, :, :end-start]
        if self.is_augmented_decoder == True: self.voltage_decoder += self.voltage[:, :, self.output_indexes, start]

        # 4 - Update the refractory_active
        if self.is_refractory == True:
            self.refractory_active[self.refractory_active > 0] -= <int>self.dt


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void current_kernel_update_voltage_no_delay(self, size_t current_time):
        if current_time + 1 >= self.run_time_delay_max: return

        cdef size_t start = min(current_time+1, self.run_time_delay_max)
        cdef size_t end = min(start+1, self.run_time_delay_max)
        cdef float[:,:,:,:] voltage_weight_view = self.voltage_view[:, :, :, start:end]

        # self.current_kernel = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, 1), dtype=np.float32)
        # cdef float[:, :, :, :] current_kernel_view = self.current_kernel

        cdef int[:, :, :] spike_state_sub_view = self.spike_state_sub
        cdef size_t i, j, k, l, a, connections
        connections = self.synapses_actives_indexes_view.shape[1]

        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                # for k in range(self.nb_neurons):
                #     for l in range(self.nb_neurons):
                for a in range(connections):
                    k = self.synapses_actives_indexes_view[0, a]
                    if spike_state_sub_view[i, j, k] == 1:
                        l = self.synapses_actives_indexes_view[1, a]
                        # current_kernel_view[i, j, l, 0] += self.weight_view[j, k, l]
                        voltage_weight_view[i, j, l, 0] += self.weight_view[j, k, l]
    
                        if l >= self.output_start and l <= self.output_end and self.is_augmented_decoder == True:
                            self.voltage_decoder_view[i, j, l-self.input_size] += self.weight_view[j, k, l]

        # cdef size_t start = min(current_time+1, self.run_time_delay_max)
        # cdef size_t end = min(start+1, self.run_time_delay_max)
        # self.voltage[:, :, :, start:end] += self.current_kernel[:, :, :, :end-start]

        # 4 - Update the refractory_active
        if self.is_refractory == True:
            self.refractory_active[self.refractory_active > 0] -= <int>self.dt

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_network(self, list snn):
        
        self.nb_networks = len(snn)
        self.nb_neurons = len(snn[0].voltage_init)
        cdef size_t i

        # 1 - Dynamic variables
        self.voltage = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_delay_max), dtype=np.float32)

        # 2 - Static variables
        self.weight = np.empty((self.nb_networks, self.nb_neurons, self.nb_neurons), dtype=np.float32)
        self.voltage_reset = np.empty((self.nb_networks, self.nb_neurons), dtype=np.float32)
        self.tau = np.empty((self.nb_networks, self.nb_neurons), dtype=np.float32)
        self.constant_current_kernel = np.empty((self.nb_networks, self.nb_neurons), dtype=np.float32)
        self.threshold = np.empty((self.nb_networks, self.nb_neurons), dtype=np.float32)

        if self.is_delay == True:
            self.delay = np.empty((self.nb_networks, self.nb_neurons, self.nb_neurons), dtype=np.float32)
        if self.is_refractory == True:
            self.refractory = np.empty((self.nb_networks, self.nb_neurons), dtype=np.float32)

        self.input_indexes = snn[0].input_indexes
        self.hidden_indexes = snn[0].hidden_indexes
        self.output_indexes = snn[0].output_indexes

        cdef np.ndarray[np.int32_t, ndim=2] synapses_status = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.int32)
        for i in range(self.nb_networks):
            self.voltage[:, i, :, 0] += snn[i].voltage_init # init voltage matrix
            self.voltage_reset[i] = snn[i].voltage_init
            self.tau[i] = snn[i].tau
            self.constant_current_kernel[i] = snn[i].constant_current
            self.threshold[i] = snn[i].threshold
            self.weight[i] = snn[i].weight
            synapses_status[np.where((snn[i].weight > 0.001) | (snn[i].weight < -0.001))] = 1
            if self.is_delay == True:
                self.delay[i] = snn[i].delay
            if self.is_refractory == True:
                self.refractory[i] = snn[i].refractory

        # cdef np.ndarray[np.int32_t, ndim=2] synapses_actives_indexes_2 = np.array(np.where(synapses_status == 1), dtype=np.int32)
        # print("is_equal", np.array_equal(synapses_actives_indexes_2, self.synapses_actives_indexes_view))
        self.synapses_actives_indexes_view = np.array(np.where(synapses_status == 1), dtype=np.int32)

        # 2 - Check if there is delay or refractory or bad values
        if self.is_delay == True:
            self.delay = np.ceil(self.delay).astype(np.int32)
            self.delay[self.delay > self.run_time_delay_max] = self.run_time_delay_max - 1
            self.delay[self.delay < 0] = 0 # in case of negative delay
            self.is_delay = np.any(self.delay != 0)
            # self.delay_max = np.max(self.delay) + 1

        if self.is_refractory:
            self.refractory = np.ceil(self.refractory).astype(np.int32)
            self.refractory_active = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons), dtype=np.int32)
            self.is_refractory = np.any(self.refractory != 0)
            self.refractory_view = self.refractory
            self.refractory_active_view = self.refractory_active

        if self.decay_method == True:
            self.tau[self.tau <= 0] = 1.0 # in case of tau <= 0
        else:
            self.tau = np.clip(self.tau, 0.0, 1.0) # in case tau < 0 or tau > 1
            self.tau = 1.0 - self.tau # because tau = 1/tau in the LIF equation
        
        if self.disable_output_threshold == True: self.threshold[:, self.output_indexes] = 1e14 # set very high voltage to disable output reset

        # 3 - Transfer numpy memory to cython memoryview
        self.voltage_view = self.voltage
        self.voltage_reset_view = self.voltage_reset
        self.tau_view = self.tau
        self.constant_current_view = self.constant_current_kernel
        self.threshold_view = self.threshold
        self.spike_state = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)
        self.spike_state_view = self.spike_state
        self.weight_view = self.weight
        
        # 4 - Set the first time step to 0
        self.first_time_step == 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_network_step(self):
        cdef size_t i
        cdef size_t output_len = len(self.output_indexes)
        
        # 0 - If it is the first time step, do nothing
        if self.first_time_step == 0:
            # 0.1 Get the neuron to record indexes
            self.neuron_to_record_indexes = np.array([], dtype=np.int32)
            if "input" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.input_indexes))
            if "output" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.output_indexes))
            if "hidden" in self.record_layer: self.neuron_to_record_indexes = np.concatenate((self.neuron_to_record_indexes, self.hidden_indexes))
            # 0.2 Reset record_dict
            for i in range(self.population_len):
                if self.is_record_spike == True:
                    self.record_spikes[<SNN>self.networks_list[i].id] = np.zeros((self.nb_episode, self.neuron_to_record_indexes.shape[0]), np.int32)

                if self.is_record_voltage == True:
                    self.record_voltages[<SNN>self.networks_list[i].id] = np.zeros((self.nb_episode, self.neuron_to_record_indexes.shape[0]), np.float32)

                if self.is_record_augmented == True:
                    self.record_augmented[<SNN>self.networks_list[i].id] = np.zeros((self.nb_episode, output_len), np.float32)

            if self.is_record_augmented == True:
                self.init_augmented_run()

            self.first_time_step = 1
            return
        # 1 - Reset the voltage and add the previous last step voltage to the new voltage
        cdef np.ndarray[np.float32_t, ndim=4] new_voltage = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_delay_max), dtype=np.float32)
        if self.online == True:
            if self.delay_max == 1:
                new_voltage[:, :, :, 0] = self.voltage[:, :, :, self.time_step]
            else:
                new_voltage[:, :, :, :self.delay_max-1] = self.voltage[:, :, :, self.time_step+1:]
        else:
            self.voltage_decoder = np.zeros((self.nb_episode, self.nb_networks, len(self.output_indexes)), dtype=np.float32)
            self.voltage_decoder_view = self.voltage_decoder
        # print("voltage_decoder", self.voltage_decoder, "shape", np.shape(self.voltage_decoder))
        # exit()

        self.voltage = new_voltage
        self.voltage_view = self.voltage

        # 2 - Reset spike_state in order to record the new spikes
        self.spike_state = np.zeros((self.nb_episode, self.nb_networks, self.nb_neurons, self.run_time_margin), dtype=np.int32)
        self.spike_state_view = self.spike_state

        self.spike_state_decoder = np.zeros((self.nb_episode, self.nb_networks, len(self.output_indexes), self.run_time_margin), dtype=np.float32)

        # 3 - Reset record_dict
        for i in range(self.population_len):
            if self.is_record_spike == True:
                self.record_spikes[<SNN>self.networks_list[i].id] = np.zeros((self.nb_episode, self.neuron_to_record_indexes.shape[0]), np.int32)

            if self.is_record_voltage == True:
                self.record_voltages[<SNN>self.networks_list[i].id] = np.zeros((self.nb_episode, self.neuron_to_record_indexes.shape[0]), np.float32)

            if self.is_record_augmented == True:
                self.record_augmented[<SNN>self.networks_list[i].id] = np.zeros((self.nb_episode, output_len), np.float32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void augmented_update(self, size_t current_time):
        if current_time + 1 >= self.run_time_margin: return
        cdef int time_step = current_time + 1
        cdef double spike_floor
        cdef size_t i, j, k, output_len
        cdef float[:,:,:] spike_state_decoder_view = self.spike_state_decoder[:, :, :, time_step]
        output_len = self.output_indexes.shape[0]


        for i in range(self.nb_episode):
            for j in range(self.nb_networks):
                for k in range(output_len):
                    
                    # 1 - Update the spike_state_decoder with the voltage_decoder (LIF Version)  -> (Σ_W/threshold) * (1 - (1/tau) * (1/(1+Refractory)) ; tau -> [1, +inf], refractory -> [0, +inf]
                    if self.decay_method == True and self.is_refractory == False:
                        spike_state_decoder_view[i, j, k] = (self.voltage_decoder_view[i, j, k]/self.threshold_decoder_view[j, k]) * (1-(1/(self.tau_decoder_view[j, k]))) 
                    
                    if self.decay_method == True and self.is_refractory == True:
                        spike_state_decoder_view[i, j, k] = (self.voltage_decoder_view[i, j, k]/self.threshold_decoder_view[j, k]) * (1-(1/(self.tau_decoder_view[j, k]))) * (1/(1+self.refractory_decoder_view[i, j]))


                    # 1 - Update the spike_state_decoder with the voltage_decoder (Beta Version) -> (Σ_W/threshold) * (tau) * (1/(1+Refractory)) ; tau -> [0, 1], refractory -> [0, +inf]
                    elif self.decay_method == False and self.is_refractory == False:
                        spike_state_decoder_view[i, j, k] = (self.voltage_decoder_view[i, j, k]/self.threshold_decoder_view[j, k]) * self.tau_decoder_view[j, k]

                    elif self.decay_method == False and self.is_refractory == True:
                        spike_state_decoder_view[i, j, k] = (self.voltage_decoder_view[i, j, k]/self.threshold_decoder_view[j, k]) * self.tau_decoder_view[j, k] * (1/(1+self.refractory_decoder_view[i, j]))
                    
                    spike_floor = math.floor(spike_state_decoder_view[i, j, k])
                    self.voltage_decoder_view[i, j, k] = spike_state_decoder_view[i, j, k] - spike_floor
                    # self.voltage_decoder_view[i, j, k] = 0


                    # 2 - Add the rounded spike to the spike_state_decoder
                    # 2.1 - Positive spike only
                    if self.spike_format_type == 0:
                        spike_state_decoder_view[i, j, k] = spike_floor if spike_floor >= 0 else 0

                    # 2.2 - Abosolute spike only                    
                    elif self.spike_format_type == 1:
                        spike_state_decoder_view[i, j, k] = math.fabs(spike_floor)

                    # 2.3 - Both positive and negative spikes
                    elif self.spike_format_type == 2:
                        spike_state_decoder_view[i, j, k] = spike_floor


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void augmented_importance(self, list snn):
        cdef size_t snn_len = len(snn)
        cdef size_t i, j, k, l, index
        cdef size_t nb_outputs = self.output_indexes.shape[0]
        cdef float[:] spikes
        cdef float[:] spikes_importance_view = self.spikes_importance
        cdef float[:,:,:,:] spike_state_decoder_view = self.spike_state_decoder
        cdef bint is_first_index_found = False
        cdef size_t spikes_len = len(spike_state_decoder_view[0, 0, 0])

        for i in range(self.nb_episode):
            for j in range(snn_len):
                for k in range(nb_outputs):
                    spikes = spike_state_decoder_view[i, j, k]
                    
                    if self.importance_type == 0: # first index
                        index = 0
                        is_first_index_found = False
                        for l in range(spikes_len):
                            if spikes[l] != 0:
                                is_first_index_found = True
                            if is_first_index_found == True:
                                if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step
                                spikes[l] = spikes[l] + ((spikes[l]/self.spike_max_time_step) * spikes_importance_view[index])
                                index += 1

                    elif self.importance_type == 1: # by index
                        index = 0
                        for l in range(spikes_len):
                            if spikes[l] != 0:
                                if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step
                                spikes[l] = spikes[l] + ((spikes[l]/self.spike_max_time_step) * spikes_importance_view[index])
                                index += 1

                    elif self.importance_type == 2: # all
                        for l in range(spikes_len):
                            if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step
                            spikes[l] = spikes[l] + ((spikes[l]/self.spike_max_time_step) * spikes_importance_view[l])

                    elif self.importance_type == 3: # nothing
                        for l in range(spikes_len):
                            if spikes[l] > self.spike_max_time_step: spikes[l] = self.spike_max_time_step                    


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef np.ndarray augmented_linear_spike_importance(self, size_t spikes_distribution, size_t run_len, bint is_descending=False):

        cdef np.ndarray[np.float32_t, ndim=1] linespace, linespace_normalized, linespace_spikes
        # 1 - Create a linspace from spikes_distribution to 0 or inversement
        if is_descending == False:
            linespace = np.linspace(0, spikes_distribution, run_len).astype(np.float32)
        else:
            linespace = np.linspace(spikes_distribution, 0, run_len).astype(np.float32)

        # 2 - Normalize the linespace sum to make it sum to 1
        linespace_normalized = linespace / linespace.sum()

        # 3 - Scale the linespace on the scale of spikes
        linespace_spikes = spikes_distribution * linespace_normalized

        return linespace_spikes



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init_augmented_run(self):
        # Init new decoder
        self.voltage_decoder = np.zeros((self.nb_episode, self.nb_networks, len(self.output_indexes)), dtype=np.float32)
        self.threshold_decoder = self.threshold[:, self.output_indexes]
        self.tau_decoder = self.tau[:, self.output_indexes]
        if self.is_refractory == True:
            self.refractory_decoder = self.refractory[:, self.output_indexes]
            self.refractory_decoder_view = self.refractory_decoder

        self.voltage_decoder_view = self.voltage_decoder
        self.threshold_decoder_view = self.threshold_decoder
        self.tau_decoder_view = self.tau_decoder


        self.spike_state_decoder = np.zeros((self.nb_episode, self.nb_networks, len(self.output_indexes), self.run_time_margin), dtype=np.float32)
        self.output_start = self.output_indexes[0]
        self.output_end = self.output_indexes[self.output_indexes.shape[0]-1]
        self.input_size = self.input_indexes.shape[0]



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void init_augmented_spikes_decoder(self, size_t spike_max = 10, size_t spike_distribution_importance = 10, str importance_type = "nothing", str linear_spike_importance_type = "descending", str spike_type = "positive"):
        self.spikes_max = spike_max
        self.spike_distribution_importance = spike_distribution_importance
        self.linear_spike_importance_type = True if linear_spike_importance_type == "descending" else False
        self.is_augmented_decoder = True

        if importance_type == "first_index":
            self.importance_type = 0
        elif importance_type == "by_index":
            self.importance_type = 1
        elif importance_type == "all":
            self.importance_type = 2
        elif importance_type == "nothing":
            self.importance_type = 3

        if spike_type == "positive":
            self.spike_format_type = 0
        elif spike_type == "absolute":
            self.spike_format_type = 1
        elif spike_type == "raw":
            self.spike_format_type = 2


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void record(self):
        # cdef size_t nb_snns = len(self.networks_list)
        # cdef size_t i, j
        # # cdef np.ndarray[np.int_t, ndim=3] spike_sumed = np.sum(self.spike_state, axis=3)
        # cdef long[:,:,:] spike_sumed = np.sum(self.spike_state[:,:, self.neuron_to_record_indexes], axis=3)
        # cdef float[:,:,:] voltage_record = self.voltage[:, :, self.neuron_to_record_indexes, -1]

        # for j in range(self.nb_episode):
        #     for i in range(nb_snns):
        #         self.record_spikes[(<SNN>self.networks_list[i]).id][j] += spike_sumed[j, i, :]
        #         self.record_voltages[(<SNN>self.networks_list[i]).id][j] = voltage_record[j, i]

        # # print("SPIKE MATRIX SUMED =\n", np.array(spike_sumed), "shape =", np.shape(spike_sumed))
        # # for i in range(nb_snns):
        # #     print("snn[",i,"].record\n", snn[i].record)           

        cdef size_t nb_snns = len(self.networks_list)
        cdef size_t i, j
        cdef long[:,:,:] spike_sumed
        cdef float[:,:,:] voltage_record
        cdef float[:,:,:] spike_sumed_decoder

        if self.is_record_spike:
            spike_sumed = np.sum(self.spike_state[:,:, self.neuron_to_record_indexes], axis=3)
        if self.is_record_voltage:
            voltage_record = self.voltage[:, :, self.neuron_to_record_indexes, -1]
        if self.is_record_augmented:
            spike_sumed_decoder = np.sum(self.spike_state_decoder, axis=3) 

        for j in range(self.nb_episode):
            for i in range(nb_snns):
                if self.is_record_spike:
                    self.record_spikes[(<SNN>self.networks_list[i]).id][j] += spike_sumed[j, i, :]
                if self.is_record_voltage:
                    self.record_voltages[(<SNN>self.networks_list[i]).id][j] = voltage_record[j, i]
                if self.is_record_augmented:
                    # self.record_augmented[(<SNN>self.networks_list[i]).id][j] = np.rint(spike_sumed_decoder[j, i, :])
                    self.record_augmented[(<SNN>self.networks_list[i]).id][j] = spike_sumed_decoder[j, i, :]




    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_augmented_spikes(self):
        return self.record_augmented

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_spikes(self):
        return self.record_spikes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef dict get_record_voltages(self):
        return self.record_voltages

    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list split(self, list lst, size_t s):
        cdef list chunks = []
        cdef size_t i
        
        for i in range(0, len(lst), s):
            chunk = lst[i:i + (<size_t>s)]
            chunks.append(chunk)
    
        return chunks

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int[:] n_most_frequent_np(self, np.ndarray[np.int_t, ndim=1] arr, size_t n):
        cdef np.ndarray[np.int_t, ndim=1] unique, counts, indices
        unique, counts = np.unique(arr, return_counts=True)
        indices = np.argsort(counts)[::-1][:n]
        return unique[indices].astype(np.int32)

    cdef int bernoulli_trial(self, double p):
        return rand() < p * RAND_MAX