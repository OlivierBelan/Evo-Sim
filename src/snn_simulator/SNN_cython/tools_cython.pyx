cimport cython
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
# from snn_cython cimport SNN_cython as SNN

cdef double get_time():
    cdef timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec + (ts.tv_nsec / 1000000000.)

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef SNN generate_snn(size_t nb_input = 2, size_t nb_hidden = 10, size_t nb_output=1):
#     snn = SNN()

#     cdef size_t i
#     cdef list input_list = [Neuron(input_current=0, refractory=1.0, threshold=1.0, tau_m=10) for i in range(nb_input)]
#     cdef list hidden_list = [Neuron(input_current=0, refractory=1.0, threshold=1.0, tau_m=10) for i in range(nb_hidden)]
#     cdef list output_list = [Neuron(input_current=0, refractory=1.0, threshold=1.0, tau_m=10) for i in range(nb_output)]

#     snn.add_neurons("input", input_list, True)
#     snn.add_neurons("hidden", hidden_list, True)
#     snn.add_neurons("output", output_list, True)

#     cdef size_t input_list_len = len(input_list)
#     cdef size_t output_list_len = len(output_list)
#     cdef size_t hidden_list_len = len(hidden_list)

#     cdef size_t h_it, i_it, o_it, h2_it

#     for h_it in range(hidden_list_len):

#         for i_it in range(input_list_len):
#             snn.add_synapse_matrix((<Neuron>input_list[i_it]), (<Neuron>hidden_list[h_it]), weight=1.1, delay=0.1)

#         for o_it in range(output_list_len):
#             snn.add_synapse_matrix((<Neuron>hidden_list[h_it]), (<Neuron>output_list[o_it]), weight=1.1, delay=0.1)

#         for h2_it in range(hidden_list_len):
#             if (<Neuron>hidden_list[h_it]).id != (<Neuron>hidden_list[h2_it]).id:
#                 snn.add_synapse_matrix((<Neuron>hidden_list[h2_it]), (<Neuron>hidden_list[h_it]), weight=1.1, delay=0.1)

#     return snn
