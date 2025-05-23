#!python
# cython: embedsignature=True, binding=True

from snn_cython cimport SNN_cython as SNN

import numpy as np
cimport numpy as np
np.import_array()

cimport cython
ctypedef np.uint8_t uint8

cdef class Runner_RL_cython:
    cdef public float dt # time step
    cdef public size_t run_time_original # simulation time
    cdef public size_t run_time_margin # simulation time + margin
    cdef public size_t run_time_delay_max # simulation time + margin + delay_max
    cdef public size_t nb_episode # number of episode in batch
    cdef public size_t nb_neurons
    cdef public size_t nb_networks
    cdef public size_t population_len
    cdef public size_t time_step
    cdef public size_t first_time_step
    cdef public bint online
    cdef public bint is_delay
    cdef public bint is_refractory
    cdef public bint is_threshold_reset
    cdef public bint is_record_spike
    cdef public bint is_record_voltage
    cdef public bint is_record_augmented

    cdef public np.ndarray combinatorial_encoder_table
    cdef public float[:,:] combinatorial_encoder_table_view
    cdef public bint is_combinatorial_encoder_decoder_init

    # network matrix
    cdef public np.ndarray voltage
    # cdef public np.ndarray voltage_2
    cdef public np.ndarray voltage_reset
    cdef public np.ndarray weight
    cdef public np.ndarray refractory
    cdef public np.ndarray refractory_active
    cdef public np.ndarray delay
    cdef public size_t delay_max
    cdef public np.ndarray threshold
    cdef public np.ndarray tau
    cdef public np.ndarray spike_state
    cdef public np.ndarray spike_state_sub
    cdef public np.ndarray current_kernel
    cdef public np.ndarray constant_current_kernel
    cdef public np.ndarray current_kernel_delay
    cdef public np.ndarray current_kernel_indexes
    cdef public np.ndarray current_kernel_delay_indexes

    cdef public np.ndarray input_indexes
    cdef public np.ndarray hidden_indexes
    cdef public np.ndarray output_indexes

    cdef public float[:,:,:,:] voltage_view
    cdef public float[:,:,:] voltage_sub_view
    cdef public float[:,:,:] voltage_sub_view_next
    cdef public int[:,:,:,:] spike_state_view
    cdef public int[:,:,:] spike_state_view_sub

    cdef public float[:,:,:] weight_view
    cdef public int[:,:] refractory_view
    cdef public int[:,:,:] refractory_active_view
    cdef public float[:,:] voltage_reset_view
    cdef public float[:,:] threshold_view
    cdef public float[:,:] tau_view
    cdef public float[:,:] current_view
    cdef public float[:,:] constant_current_view
    cdef public int[:,:,:] delay_view
    cdef float[:, :, :, :] current_kernel_delay_2_view
    cdef public int[:, :] synapses_actives_indexes_view
    
    # cdef public np.ndarray input_indexes
    # cdef public np.ndarray hidden_indexes
    # cdef public np.ndarray output_indexes


    # private variables
    cdef float[:,:,:,:] input_data # input data
    cdef float input_spike_amplitude # spike amplitude
    cdef list networks_list # list of networks

    cpdef void init(self, list networks_list, size_t run_time = *, size_t run_time_margin = *, float dt = *, size_t nb_episode = *, bint online = *, str neuron_reset = *, list record_layer = *, bint disable_output_threshold = *, str decay_method = *, bint is_delay = *, bint is_refractory = *, size_t delay_max = *, set record_decoding_method =*)
    cpdef void run(self, str encoder_type, np.ndarray inputs_data, size_t spike_rate= *, float spike_amplitude = *, size_t max_nb_spikes = *, int reduce_noise = *, int combinatorial_factor = *, size_t combinaison_size=*, size_t combinaison_size_max=*, float combinatorial_combinaison_noise=*, bint combinatorial_roll =*, float direct_min = *, float direct_max = *)
    cdef void run_C(self)

    # cpdef dict run(self, str type, np.ndarray inputs_data, size_t spike_rate = *, float spike_amplitude = *, size_t max_nb_spikes = *, int reduce_noise = *)

    # Init Input data functions
    cdef int init_data_set(self, str encoder_type, np.ndarray inputs_data, size_t rate, float spike_amplitude, size_t max_nb_spikes, int reduce_noise, int combinatorial_factor, size_t combinaison_size=*, size_t combinaison_size_max=*, float combinatorial_combinaison_noise=*, bint combinatorial_roll = *, float direct_min = *, float direct_max = *)
    cpdef void poisson_encoder(self, np.ndarray inputs_data, size_t rate, float spike_amplitude = *, size_t max_nb_spikes = *)
    cpdef void binomial_encoder(self, np.ndarray inputs_data, float spike_amplitude = *, size_t max_nb_spikes = *, int reduce_noise = *)
    cpdef void exact_encoder(self, np.ndarray inputs_data, float spike_amplitude = *, int max_nb_spikes = *)
    cpdef void rate_encoder(self, np.ndarray inputs_data, float spike_amplitude = *)
    cpdef void raw_encoder(self, np.ndarray inputs_data, float spike_amplitude = *)
    cpdef void direct_encoder(self, np.ndarray inputs_data, float direct_min = *, float direct_max = *)
    cpdef void combinatorial_encoder(self, np.ndarray inputs_data, int combinatorial_factor = *, size_t combinaison_size=*, size_t combinaison_size_max=*, float combinaison_noise=*, bint combinatorial_roll = *, float spike_amplitude = *)
    cdef void combinatorial_encoder_init(self, int combinatorial_factor=*, size_t combinaison_size=*, size_t combinaison_size_max=*, bint combinatorial_roll = *)
    cdef int find_first_one_index(self, float[:] row)
    cdef void add_padding_input_data(self)
    
    # Network Loop functions
    cdef void init_network(self, list snn)
    cdef void init_network_step(self)
    cdef void input_data_to_network(self, size_t x_volt)
    cdef void LIF_update_voltage_refractory(self, size_t current_time)
    cdef void LIF_update_voltage_no_refractory(self, size_t current_time)
    cdef void current_kernel_update_voltage_delay(self, size_t x_volt) # à voire
    cdef void current_kernel_update_voltage_no_delay(self, size_t current_time)

    # Utils functions
    cdef public dict record_spikes
    cdef public dict record_voltages
    cdef public list record_layer
    cdef public bint is_record_initied
    cdef public bint disable_output_threshold
    cdef public bint decay_method
    cdef public np.ndarray neuron_to_record_indexes
    cdef void record(self)
    cpdef list split(self, list lst, size_t s)
    cdef int[:] n_most_frequent_np(self, np.ndarray arr, size_t n)
    cdef int bernoulli_trial(self, double p)

    # New augmented decoder
    cdef public np.ndarray voltage_decoder
    cdef public np.ndarray threshold_decoder
    cdef public np.ndarray tau_decoder
    cdef public np.ndarray refractory_decoder

    cdef public float[:,:,:] voltage_decoder_view
    cdef public float[:,:] threshold_decoder_view
    cdef public float[:,:] tau_decoder_view
    cdef public float[:,:] refractory_decoder_view
    cdef public np.ndarray spike_state_decoder
    cdef public np.ndarray spikes_importance

    cdef public dict record_augmented

    cdef public bint is_augmented_decoder
    cdef public set output_indexes_set
    cdef public size_t output_start
    cdef public size_t output_end
    cdef public size_t input_size
    cdef public size_t spike_max
    cdef public size_t spike_max_time_step
    cdef public size_t spike_distribution_run
    cdef public size_t spike_distribution_importance
    cdef public size_t spike_format_type
    cdef public size_t importance_type
    cdef public bint linear_spike_importance_type
    cdef public bint is_voltage_reset

    cdef public bint is_normalize
    cdef public bint is_interpolate
    cdef public float interpolate_max
    cdef public float interpolate_min

    cpdef void init_augmented_spikes_decoder(self, size_t spike_max=*, size_t spike_distribution_run=*, size_t spike_distribution_importance=*, str importance_type=*, str linear_spike_importance_type=*, str spike_type=*, bint is_normalize = *, bint is_interpolate = *, float interpolate_max = *, float interpolate_min = *, bint is_voltage_reset = *)
    cdef void init_augmented_run(self)
    cdef void augmented_update(self, size_t current_time)
    cdef np.ndarray augmented_linear_spike_importance(self, size_t spikes_distribution, size_t run_len, bint is_descending=*)
    cdef void augmented_importance(self, list snn)

    cpdef dict get_record_augmented_spikes(self)
    cpdef dict get_record_spikes(self)
    cpdef dict get_record_voltages(self)