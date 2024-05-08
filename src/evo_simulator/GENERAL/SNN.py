import numpy as np
from typing import Dict, Tuple
import numba as nb


class SNN:
    def __init__(self, nb_inputs:int, nb_hiddens:int, nb_outputs:int, nb_hidden_active:int, topology_type:str= "feed_forward", outputs_multiplicator:int = 1):

        # order of neurons: inputs -> outputs -> hiddens

        self.nb_inputs:int = nb_inputs
        self.nb_hiddens:int = nb_hiddens

        self.nb_outputs:int = nb_outputs * outputs_multiplicator
        self.nb_outputs_original:int = nb_outputs
        self.outputs_multiplicator:int = outputs_multiplicator
        nb_outputs = self.nb_outputs

        self.nb_neurons:int = self.nb_inputs + self.nb_hiddens + self.nb_outputs
        self.nb_hidden_active:int = nb_hidden_active
        self.nb_neurons_active:int = self.nb_inputs + self.nb_hidden_active + self.nb_outputs
        
        self.input_first_id:int = 0
        self.input_last_id:int = self.nb_inputs - 1
        self.output_first_id:int = self.nb_inputs
        self.output_last_id:int = self.nb_inputs + self.nb_outputs - 1
        self.hidden_first_id:int = self.nb_inputs + self.nb_outputs
        self.hidden_last_id:int = self.nb_inputs + self.nb_outputs + self.nb_hiddens - 1


        # Parameters
        self.parameters:Dict[str, np.ndarray] = {}

        # Build matrices
        self.neurons_status:np.ndarray = np.zeros(self.nb_neurons, dtype=np.bool8)
        self.neurons_status[:self.nb_neurons_active] = True
        self.synapses_status:np.ndarray = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.bool8)
        self.neuron_deactived_status:np.ndarray = np.zeros(self.nb_hiddens, dtype=np.bool8)


        # Neurons Parameters
        # Voltages
        self.parameters["voltage"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Thresholds
        self.parameters["threshold"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Tau (decay/leak)
        self.parameters["tau"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Input_current
        self.parameters["input_current"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Refractory
        self.parameters["refractory"] = np.zeros(self.nb_neurons, dtype=np.float32)

        # Synapses Parameters
        # Weights
        self.parameters["weight"] = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.float32)

        # Delay
        self.parameters["delay"] = np.zeros((self.nb_neurons, self.nb_neurons), dtype=np.float32)
        

        self.inputs:Dict[str, np.ndarray] = {}
        self.hiddens:Dict[str, np.ndarray] = {}
        self.outputs:Dict[str, np.ndarray] = {}
        self.__inputs()
        self.__hiddens()
        self.__outputs()

        # remove self connections
        np.fill_diagonal(self.synapses_status, False)

        # Connect layers
        self.__build_topology(topology_type)

        self.neuron_actives_indexes:np.ndarray = None
        self.synapses_actives_indexes:Tuple[np.ndarray, np.ndarray] = None
        self.synapses_unactives_indexes:Tuple[np.ndarray, np.ndarray] = None
        self.synapses_unactives_weight_indexes:Tuple[np.ndarray, np.ndarray] = None
        self.update_indexes()

    def update_indexes(self):
        self.neuron_actives_indexes, self.hiddens["neurons_indexes_active"], self.synapses_actives_indexes, self.synapses_unactives_indexes, self.synapses_unactives_weight_indexes = self.update_indexes_jit(
            self.neurons_status,
            self.synapses_status,
            self.hiddens["neurons_indexes"],
            self.hiddens["neurons_status"],
            self.parameters["weight"],
        )
        self.nb_hidden_active = self.hiddens["neurons_indexes_active"].shape[0]
        self.nb_neurons_active = self.nb_inputs + self.nb_hidden_active + self.nb_outputs

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def update_indexes_jit(
        neurons_status:np.ndarray,
        synapses_status:np.ndarray,
        hidden_neurons_indexes:np.ndarray,
        hidden_neurons_status:np.ndarray,
        weight_matrix:np.ndarray,
        ):
        # 1- Find the active neurons' indices (all and hidden)
        neuron_actives_indexes:np.ndarray = np.where(neurons_status)[0].astype(np.int32)
        hidden_neurons_indexes_active:np.ndarray = hidden_neurons_indexes[np.where(hidden_neurons_status)[0]].astype(np.int32)
        
        # 2 - Find the active and unactive synapses' indices for the given active neurons
        neuron_actives_len:int = neuron_actives_indexes.shape[0]
        neuron_actives_synapses_status:np.ndarray = np.empty((neuron_actives_len, neuron_actives_len), dtype=synapses_status.dtype)
        for i in nb.prange(neuron_actives_len):
            for j in nb.prange(neuron_actives_len):
                neuron_actives_synapses_status[i, j] = synapses_status[neuron_actives_indexes[i], neuron_actives_indexes[j]]
       
        # Find the active and inactive synapses' indices for the given active neurons
        synapses_actives_local_indexes:Tuple[np.ndarray, np.ndarray] = np.where(neuron_actives_synapses_status)
        synapses_unactives_local_indexes:Tuple[np.ndarray, np.ndarray] = np.where(~neuron_actives_synapses_status)

        # Map the local indices back to the original indices in the 'synapses_status' matrix
        synapses_actives_indexes: Tuple[np.ndarray, np.ndarray] = (neuron_actives_indexes[synapses_actives_local_indexes[0]], neuron_actives_indexes[synapses_actives_local_indexes[1]])
        synapses_unactives_indexes: Tuple[np.ndarray, np.ndarray] = (neuron_actives_indexes[synapses_unactives_local_indexes[0]], neuron_actives_indexes[synapses_unactives_local_indexes[1]])

        # 3 - Find the active and unactive synapses' indices given the active weight (weight != 0)
        synapses_unactives_len:int = synapses_unactives_indexes[0].shape[0]
        synapses_unactives_weight_indexes:Tuple[np.ndarray, np.ndarray] = (np.empty(synapses_unactives_len, dtype=np.int32), np.empty(synapses_unactives_len, dtype=np.int32))

        weight, row, col, current_size = 0.0, 0, 0, 0
        for i in nb.prange(synapses_unactives_len):
            row:int = synapses_unactives_indexes[0][i]
            col:int = synapses_unactives_indexes[1][i]
            weight:float = weight_matrix[row, col]
            if weight > 0.001 or weight < -0.001:
                synapses_unactives_weight_indexes[0][current_size] = row
                synapses_unactives_weight_indexes[1][current_size] = col
                current_size += 1
        
        # Trim the arrays to the actual size
        synapses_unactives_weight_indexes:Tuple[np.ndarray, np.ndarray] = (synapses_unactives_weight_indexes[0][:current_size], 
                                                synapses_unactives_weight_indexes[1][:current_size])
        
        return neuron_actives_indexes, hidden_neurons_indexes_active, synapses_actives_indexes, synapses_unactives_indexes, synapses_unactives_weight_indexes


    def __build_topology(self, topology_type:str):
        if topology_type == "feed_forward":
            self.__feed_forward()
        elif topology_type == "fully_connected":
            self.__fully_connected()
        elif topology_type == "recurrent_hidden":
            self.__recurrent_hidden()
        elif topology_type == "no_connection":
            return
        else:
            raise Exception("Topology '", topology_type,"' type not found, the available types are: 'feed_forward', 'fully_connected', 'recurrent_hidden'")

    def __feed_forward(self):
        if self.nb_hidden_active > 0:
            self.connect_layers(self.inputs, self.hiddens) # Connect inputs to hiddens
            self.connect_layers(self.hiddens, self.outputs) # Connect hiddens to outputs
        else:
            self.connect_layers(self.inputs, self.outputs) # Connect inputs to outputs

    def __fully_connected(self):
        self.connect_layers(self.inputs, self.outputs) # Connect inputs to outputs
        if self.nb_hidden_active > 0:
            self.connect_layers(self.inputs, self.hiddens) # Connect inputs to hiddens
            self.connect_layers(self.hiddens, self.outputs) # Connect hiddens to outputs
            self.connect_layers(self.hiddens, self.hiddens) # Connect hiddens to hiddens

    def __recurrent_hidden(self):
        if self.nb_hidden_active > 0:
            self.connect_layers(self.inputs, self.hiddens) # Connect inputs to hiddens
            self.connect_layers(self.hiddens, self.outputs) # Connect hiddens to outputs
            self.connect_layers(self.hiddens, self.hiddens) # Connect hiddens to hiddens
        else:
            raise Exception("No hidden layer")

    def __set_synapses_indexes(self):
        np.fill_diagonal(self.synapses_status, True)
        self.inputs["synapses_indexes"] = np.where(self.synapses_status[:self.nb_inputs] == False)
        self.inputs["synapses_indexes"] = tuple(np.array([self.inputs["synapses_indexes"][0], self.inputs["synapses_indexes"][1]]))
        self.outputs["synapses_indexes"] = np.where(self.synapses_status[self.nb_inputs:self.nb_inputs+self.nb_outputs] == False)
        self.outputs["synapses_indexes"] = tuple(np.array([self.outputs["synapses_indexes"][0] + self.output_first_id, self.outputs["synapses_indexes"][1]]))
        self.hiddens["synapses_indexes"] = np.where(self.synapses_status[self.nb_inputs+self.nb_outputs:] == False)
        self.hiddens["synapses_indexes"] = tuple(np.array([self.hiddens["synapses_indexes"][0] + self.hidden_first_id, self.hiddens["synapses_indexes"][1]]))
        np.fill_diagonal(self.synapses_status, False)

    def __set_synapses_actives_indexes(self):
        self.inputs["synapses_actives_indexes"] = np.where(self.synapses_status[:self.nb_inputs] == True)
        self.inputs["synapses_actives_indexes"] = tuple(np.array([self.inputs["synapses_actives_indexes"][0], self.inputs["synapses_actives_indexes"][1]]))
        self.outputs["synapses_actives_indexes"] = np.where(self.synapses_status[self.nb_inputs:self.nb_inputs+self.nb_outputs] == True)
        self.outputs["synapses_actives_indexes"] = tuple(np.array([self.outputs["synapses_actives_indexes"][0] + self.output_first_id, self.outputs["synapses_actives_indexes"][1]]))
        self.hiddens["synapses_actives_indexes"] = np.where(self.synapses_status[self.nb_inputs+self.nb_outputs:] == True)
        self.hiddens["synapses_actives_indexes"] = tuple(np.array([self.hiddens["synapses_actives_indexes"][0] + self.hidden_first_id, self.hiddens["synapses_actives_indexes"][1]]))

    def connect_layers(self, source:Dict[str, np.ndarray], target:Dict[str, np.ndarray]):
        
        # Get active neurons indexes
        source_active_neurons_indexes = source["neurons_indexes"][source["neurons_status"] == True]
        target_active_neurons_indexes = target["neurons_indexes"][target["neurons_status"] == True]

        # Create meshgrid of active neuron indexes
        source_mesh, target_mesh = np.meshgrid(source_active_neurons_indexes, target_active_neurons_indexes)

        # Reshape meshgrid to 1D arrays
        source_synapses_indexes = source_mesh.ravel()
        target_synapses_indexes = target_mesh.ravel()

        # Connect neurons
        self.synapses_status[source_synapses_indexes, target_synapses_indexes] = True

        # deactivate self-connection
        np.fill_diagonal(self.synapses_status, False)

    def __inputs(self):
        self.inputs["weight"] = self.parameters["weight"][:self.nb_inputs]
        self.inputs["voltage"] = self.parameters["voltage"][:self.nb_inputs]
        self.inputs["threshold"] = self.parameters["threshold"][:self.nb_inputs]
        self.inputs["tau"] = self.parameters["tau"][:self.nb_inputs]
        self.inputs["refractory"] = self.parameters["refractory"][:self.nb_inputs]

        self.inputs["input_current"] = self.parameters["input_current"][:self.nb_inputs]
        self.inputs["delay"] = self.parameters["delay"][:self.nb_inputs]

        self.inputs["neurons_status"] = self.neurons_status[:self.nb_inputs]
        self.inputs["synapses_status"] = self.synapses_status[:self.nb_inputs]
        self.inputs["neurons_indexes"] = np.arange(self.nb_inputs, dtype=np.int32)
    
    def __hiddens(self):
        self.hiddens["weight"] = self.parameters["weight"][self.nb_inputs+self.nb_outputs:]
        self.hiddens["voltage"] = self.parameters["voltage"][self.nb_inputs+self.nb_outputs:]
        self.hiddens["threshold"] = self.parameters["threshold"][self.nb_inputs+self.nb_outputs:]
        self.hiddens["tau"] = self.parameters["tau"][self.nb_inputs+self.nb_outputs:]
        self.hiddens["refractory"] = self.parameters["refractory"][self.nb_inputs+self.nb_outputs:]

        self.inputs["input_current"] = self.parameters["input_current"][self.nb_inputs+self.nb_outputs:]
        self.inputs["delay"] = self.parameters["delay"][self.nb_inputs+self.nb_outputs:]

        self.hiddens["neurons_status"] = self.neurons_status[self.nb_inputs+self.nb_outputs:]
        self.hiddens["synapses_status"] = self.synapses_status[self.nb_inputs+self.nb_outputs:]
        self.hiddens["neurons_indexes"] = np.arange(self.nb_inputs+self.nb_outputs, self.nb_inputs+self.nb_outputs+self.nb_hiddens)
        self.hiddens["neurons_indexes_active"] = self.hiddens["neurons_indexes"][np.where(self.hiddens["neurons_status"] == True)[0]]

    def __outputs(self):
        self.outputs["weight"] = self.parameters["weight"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["voltage"] = self.parameters["voltage"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["threshold"] = self.parameters["threshold"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["tau"] = self.parameters["tau"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["refractory"] = self.parameters["refractory"][self.nb_inputs:self.nb_inputs+self.nb_outputs]

        self.inputs["input_current"] = self.parameters["input_current"][self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.inputs["delay"] = self.parameters["delay"][self.nb_inputs:self.nb_inputs+self.nb_outputs]

        self.outputs["neurons_status"] = self.neurons_status[self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["synapses_status"] = self.synapses_status[self.nb_inputs:self.nb_inputs+self.nb_outputs]
        self.outputs["neurons_indexes"] = np.arange(self.nb_inputs, self.nb_inputs+self.nb_outputs,  dtype=np.int32)
        self.outputs["neurons_indexes_formated"] = np.array(np.split(self.outputs["neurons_indexes"], self.nb_outputs_original))
