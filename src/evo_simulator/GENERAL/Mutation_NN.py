import numpy as np
import numba as nb
from typing import Any, Dict, Tuple, List
from evo_simulator.GENERAL.NN import NN
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
import evo_simulator.TOOLS as TOOLS

class Mutation():    

    def __init__(self, config_path_file:str, attributes_manager:Attribute_Paramaters):
        self.attributes:Mutation_Attributes = Mutation_Attributes(attributes_manager)
        self.topologies:Mutation_Topologies = Mutation_Topologies(attributes_manager)
        self.config_mutation:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["Mutation"])
        self.__init_config()

    def __init_config(self) -> None:
        self.prob_creation_mutation:float = float(self.config_mutation["Mutation"]["prob_creation_mutation"])

        # Mutation probabilities
        self.prob_mutation:float = float(self.config_mutation["Mutation"]["prob_mutation"])

        # Topology mutations
        self.single_structural_mutation:bool = True if self.config_mutation["Mutation"]["single_structural_mutation"] == "True" else False

        self.prob_add_neuron:float = float(self.config_mutation["Mutation"]["prob_add_neuron"])
        self.prob_delete_neuron:float = float(self.config_mutation["Mutation"]["prob_delete_neuron"])

        self.prob_activate_neuron:float = float(self.config_mutation["Mutation"]["prob_activate_neuron"])
        self.prob_deactivate_neuron:float = float(self.config_mutation["Mutation"]["prob_deactivate_neuron"])

        self.prob_add_synapse:float = float(self.config_mutation["Mutation"]["prob_add_synapse"])
        self.prob_delete_synapse:float = float(self.config_mutation["Mutation"]["prob_delete_synapse"])

        self.prob_activate_synapse:float = float(self.config_mutation["Mutation"]["prob_activate_synapse"])
        self.prob_deactivate_synapse:float = float(self.config_mutation["Mutation"]["prob_deactivate_synapse"])

        # Parameters mutations
        self.prob_mutate_neuron_params:float = float(self.config_mutation["Mutation"]["prob_mutate_neuron_params"])
        self.prob_mutate_synapse_params:float = float(self.config_mutation["Mutation"]["prob_mutate_synapse_params"])
    

class Mutation_Attributes():
    def __init__(self, attributes_manager:Attribute_Paramaters):
        self.attributes_manager:Attribute_Paramaters = attributes_manager
        self.mu_parameters:Dict[str, np.ndarray] = self.attributes_manager.mu_parameters
        self.sigma_parameters:Dict[str, np.ndarray] = self.attributes_manager.sigma_parameters
        self.min_parameters:Dict[str, np.ndarray] = self.attributes_manager.min_parameters
        self.max_parameters:Dict[str, np.ndarray] = self.attributes_manager.max_parameters

        self.mu_bias = 0.0
        self.sigma_bias = 1.0


    def first_mutation_attributes_mu_sigma(self, population:Dict[int, Genome_NN], neurons_parameters_names:List[str], synapses_parameters_names:List[str]):
        self.neurons_mu_sigma(population, neurons_parameters_names)
        self.synapses_mu_sigma(population, synapses_parameters_names)


    def neurons_mu_sigma(self, population:Dict[int, Genome_NN], parameters_names:List[str]):
        '''
        Mutate neurons attributes
        mu/sigma_dict: Dict of parameters to mutate SNN(voltage, reset_voltage, threshold, tau, input_current, refractory) or ANN(bias)
        mu/sigma_dict (value): has to be a matrix of shape (nb_neurons, nb_neurons) or a vector of len == nb_neuron_max or a scalars
        '''
        # Safer for the ram memory (not sure if its faster, seems to be faster)
        for genome in population.values():
            neuron_actives_indexes:np.ndarray = genome.nn.neuron_actives_indexes
            for param in parameters_names:
                parameter:np.ndarray = genome.nn.parameters[param][neuron_actives_indexes]
                mu_param:np.ndarray = self.mu_parameters[param] if self.mu_parameters[param].size == 1 else self.mu_parameters[param][neuron_actives_indexes]
                sigma_param:np.ndarray = self.sigma_parameters[param] if self.sigma_parameters[param].size == 1 else self.sigma_parameters[param][neuron_actives_indexes]
                min_param:np.ndarray = self.min_parameters[param] if self.min_parameters[param].size == 1 else self.min_parameters[param][neuron_actives_indexes]
                max_param:np.ndarray = self.max_parameters[param] if self.max_parameters[param].size == 1 else self.max_parameters[param][neuron_actives_indexes]
                genome.nn.parameters[param][neuron_actives_indexes] = self.epsilon_mu_sigma_jit(parameter, mu_param, sigma_param, min_param, max_param, self.mu_bias, self.sigma_bias)


    def neurons_sigma(self, population:Dict[int, Genome_NN], parameters_names:List[str]):
        '''
        Mutate neurons attributes
        mu/sigma_dict: Dict of parameters to mutate SNN(voltage, reset_voltage, threshold, tau, input_current, refractory) or ANN(bias)
        mu/sigma_dict (value): has to be a matrix of shape (nb_neurons, nb_neurons) or a vector of len == nb_neuron_max or a scalars
        '''
        # Safer for the ram memory (not sure if its faster, seems to be faster)
        for genome in population.values():
            neuron_actives_indexes:np.ndarray = genome.nn.neuron_actives_indexes
            for param in parameters_names:
                # if param == "coeff": 
                #     print("before mutation coeff param", genome.nn.parameters[param][neuron_actives_indexes])
                parameter:np.ndarray = genome.nn.parameters[param][neuron_actives_indexes]
                sigma_param:np.ndarray = self.sigma_parameters[param] if self.sigma_parameters[param].size == 1 else self.sigma_parameters[param][neuron_actives_indexes]
                min_param:np.ndarray = self.min_parameters[param] if self.min_parameters[param].size == 1 else self.min_parameters[param][neuron_actives_indexes]
                max_param:np.ndarray = self.max_parameters[param] if self.max_parameters[param].size == 1 else self.max_parameters[param][neuron_actives_indexes]
                genome.nn.parameters[param][neuron_actives_indexes] += self.epsilon_sigma_jit(parameter, sigma_param, min_param, max_param, self.mu_bias, self.sigma_bias)
                genome.nn.parameters[param][neuron_actives_indexes] = np.clip(genome.nn.parameters[param][neuron_actives_indexes], min_param, max_param)
            # if param == "coeff":
            #     print("after mutation coeff param", genome.nn.parameters[param][neuron_actives_indexes])


    def synapses_mu_sigma(self, population:Dict[int, Genome_NN], parameters_names:List[str]):
        '''
        Mutate synapses attributes
        mu/sigma_dict: Dict of parameters to mutate (voltage, reset_voltage, threshold, tau, input_current, refractory)
        mu/sigma_dict (value): has to be a matrix of shape (nb_neurons, nb_neurons) or a vector of len == nb_neuron_max or a scalars
        '''
        # Safer for the ram memory (not sure if its faster, seems to be faster)
        for genome in population.values():
            synapses_actives_indexes:Tuple[np.ndarray, np.ndarray] = genome.nn.synapses_actives_indexes
            for param in parameters_names:
                parameter:np.ndarray = genome.nn.parameters[param][synapses_actives_indexes]
                mu_param:np.ndarray = self.mu_parameters[param] if self.mu_parameters[param].size == 1 else self.mu_parameters[param][synapses_actives_indexes]                
                sigma_param:np.ndarray = self.sigma_parameters[param] if self.sigma_parameters[param].size == 1 else self.sigma_parameters[param][synapses_actives_indexes]
                min_param:np.ndarray = self.min_parameters[param] if self.min_parameters[param].size == 1 else self.min_parameters[param][synapses_actives_indexes]
                max_param:np.ndarray = self.max_parameters[param] if self.max_parameters[param].size == 1 else self.max_parameters[param][synapses_actives_indexes]
                genome.nn.parameters[param][synapses_actives_indexes] = self.epsilon_mu_sigma_jit(parameter, mu_param, sigma_param, min_param, max_param, self.mu_bias, self.sigma_bias)


    def synapses_sigma(self, population:Dict[int, Genome_NN], parameters_names:List[str]):
        '''
        Mutate synapses attributes
        mu/sigma_dict: Dict of parameters to mutate (voltage, reset_voltage, threshold, tau, input_current, refractory)
        mu/sigma_dict (value): has to be a matrix of shape (nb_neurons, nb_neurons) or a vector of len == nb_neuron_max or a scalars
        '''
        # Safer for the ram memory (not sure if its faster, seems to be faster)
        for genome in population.values():
            synapses_actives_indexes:Tuple[np.ndarray, np.ndarray] = genome.nn.synapses_actives_indexes
            for param in parameters_names:
                parameter:np.ndarray = genome.nn.parameters[param][synapses_actives_indexes]
                sigma_param:np.ndarray = self.sigma_parameters[param] if self.sigma_parameters[param].size == 1 else self.sigma_parameters[param][synapses_actives_indexes]
                min_param:np.ndarray = self.min_parameters[param] if self.min_parameters[param].size == 1 else self.min_parameters[param][synapses_actives_indexes]
                max_param:np.ndarray = self.max_parameters[param] if self.max_parameters[param].size == 1 else self.max_parameters[param][synapses_actives_indexes]
                genome.nn.parameters[param][synapses_actives_indexes] += self.epsilon_sigma_jit(parameter, sigma_param, min_param, max_param, self.mu_bias, self.sigma_bias)
                genome.nn.parameters[param][synapses_actives_indexes] = np.clip(genome.nn.parameters[param][synapses_actives_indexes], min_param, max_param)


    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def epsilon_mu_sigma_jit(parameter:np.ndarray, mu_parameter:np.ndarray, sigma_paramater:np.ndarray, min:np.ndarray, max:np.ndarray, mu_bias:float, sigma_coef:float) -> np.ndarray:
        '''
        Jit function for the epsilon computation
        '''
        # 1- set Epislon with the Gaussian (from randn) distribution (Mu -> center of the distribution, Sigma -> width of the distribution)
        mu:np.ndarray = mu_parameter + mu_bias
        sigma:np.ndarray = sigma_paramater * sigma_coef
        epsilon:np.ndarray = np.random.randn(1, parameter.size) * sigma + mu
        # 2- clip Epsilon and apply it to the neurons parameters
        return np.clip(epsilon.astype(np.float32)[0], min, max)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def epsilon_sigma_jit(parameter:np.ndarray, sigma_paramater:np.ndarray, min:np.ndarray, max:np.ndarray, mu_bias:float, sigma_coef:float) -> None:
        '''
        Jit function for the epsilon computation
        '''
        # 1- set Epislon with the Gaussian (from randn) distribution (Mu -> center of the distribution, Sigma -> width of the distribution)
        mu:np.ndarray = mu_bias
        sigma:np.ndarray = sigma_paramater * sigma_coef
        epsilon:np.ndarray = np.random.randn(1, parameter.size) * sigma + mu
        # 2- clip Epsilon and apply it to the neurons parameters
        return np.clip(epsilon.astype(np.float32)[0], min, max)



class Mutation_Topologies():

    def __init__(self, attributes_manager:Attribute_Paramaters):
        # General Parameters
        self.neuron_parameters_name:List[str] = attributes_manager.parameters_neuron_names
        self.synapse_parameters_name:List[str] = attributes_manager.parameters_synapse_names

        self.attributes_manager:Attribute_Paramaters = attributes_manager
        self.mu_parameters:Dict[str, np.ndarray] = self.attributes_manager.mu_parameters
        self.sigma_parameters:Dict[str, np.ndarray] = self.attributes_manager.sigma_parameters
        self.min_parameters:Dict[str, np.ndarray] = self.attributes_manager.min_parameters
        self.max_parameters:Dict[str, np.ndarray] = self.attributes_manager.max_parameters



    def add_neuron_betwen_two_neurons(self, nn:NN, new_neurons:np.ndarray, neurons_in:np.ndarray, neurons_out:np.ndarray, new_weights:np.ndarray = 1, new_delays:np.ndarray = 0.1, set_auto_attributes:bool=True):
        '''
        Add a neuron between two neurons
        '''
        if type(neurons_in) == int: neurons_in = np.array([neurons_in])
        if type(neurons_out) == int: neurons_out = np.array([neurons_out])
        if type(new_neurons) == int: new_neurons = np.array([new_neurons])
        if new_neurons.shape[0] != neurons_in.shape[0] and new_neurons.shape[0] != neurons_out.shape[0]: raise ValueError("new_neurons must have the same shape as neurons_in and neurons_out")
        
        # 1 - add the new neuron 
        self.add_neurons(nn, new_neurons, update_indexes=False)
        # 2 - add the new synapses
        if nn.network_type == "SNN":
            delay:int = nn.parameters["delay"][neurons_in, neurons_out] if "delay" in nn.parameters else 0
            self.add_synapses(nn, neurons_in, new_neurons, weights=nn.parameters["weight"][neurons_in, neurons_out], delays=delay, set_auto_attributes=set_auto_attributes, update_indexes=False)
            self.add_synapses(nn, new_neurons, neurons_out,  weights=new_weights, delays=new_delays, set_auto_attributes=set_auto_attributes, update_indexes=False)
        elif nn.network_type == "ANN":
            self.add_synapses(nn, neurons_in, new_neurons, weights=nn.parameters["weight"][neurons_in, neurons_out], delays=0, set_auto_attributes=set_auto_attributes, update_indexes=False)
            self.add_synapses(nn, new_neurons, neurons_out,  weights=new_weights, delays=0, set_auto_attributes=set_auto_attributes, update_indexes=False)

        # 3 - deactivate the old synapse
        self.deactivate_synapses(nn, neurons_in, neurons_out, update_indexes=False)

        nn.update_indexes()


    def set_neuron_attributes(self, nn:NN, neurons_ids:np.ndarray):
        if type(neurons_ids) == int: neurons_ids = np.array([neurons_ids])
        # for param, sigma in self.sigma_neuron.items():
            # nn.parameters[param][neurons_ids] = np.clip(((np.random.randn(1, neurons_ids.size) * sigma) + self.mu_neuron[param]), self.min_neuron[param], self.max_neuron[param])
            # nn.parameters[param][neurons_ids] = self.set_neuron_attributes_jit(neurons_ids.size, sigma, self.mu_neuron[param], self.min_neuron[param], self.max_neuron[param])

        for param in self.neuron_parameters_name:
            nn.parameters[param][neurons_ids] = self.set_neuron_attributes_jit(neurons_ids.size, self.sigma_parameters[param], self.mu_parameters[param], self.min_parameters[param], self.max_parameters[param])
        
    def set_synapse_attributes(self, nn:NN, neurons_in:np.ndarray, neurons_out:np.ndarray):
        if type(neurons_in) == int and type(neurons_out) == int: 
            neurons_in = np.array([neurons_in])
            neurons_out = np.array([neurons_out])
        # for param, sigma in self.sigma_synapse.items():
            # nn.parameters[param][neurons_in, neurons_out] = np.clip(((np.random.randn(1, neurons_in.size) * sigma) + self.mu_synapse[param]), self.min_synapse[param], self.max_synapse[param])
            # nn.parameters[param][neurons_in, neurons_out] = self.set_synapse_attributes_jit(neurons_in.size, sigma, self.mu_synapse[param], self.min_synapse[param], self.max_synapse[param])

        for param in self.synapse_parameters_name:
            nn.parameters[param][neurons_in, neurons_out] = self.set_synapse_attributes_jit(neurons_in.size, self.sigma_parameters[param], self.mu_parameters[param], self.min_parameters[param], self.max_parameters[param])
    
    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def set_neuron_attributes_jit(size:int, sigma:np.ndarray, mu_neuron:np.ndarray, min_neuron:np.ndarray, max_neuron:np.ndarray):
        return np.clip(((np.random.randn(1, size) * sigma) + mu_neuron).astype(np.float32), min_neuron, max_neuron)
        

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def set_synapse_attributes_jit(size:int, sigma:np.ndarray, mu_synapse:np.ndarray, min_synapse:np.ndarray, max_synapse:np.ndarray):
        return np.clip(((np.random.randn(1, size) * sigma) + mu_synapse).astype(np.float32), min_synapse, max_synapse)


    def add_neurons(self, nn:NN, neurons_ids:np.ndarray, update_indexes:bool=True):
        nn.neurons_status[neurons_ids] = True
        self.set_neuron_attributes(nn, neurons_ids)
        if update_indexes == True:
            nn.update_indexes()

    def del_neurons(self, nn:NN, neurons_ids:np.ndarray):
        # filter hidden neurons
        if type(neurons_ids) == int:
            neurons_ids = np.array([neurons_ids])
        hidden_neuron_id_only:np.ndarray = neurons_ids[np.where(np.isin(neurons_ids, nn.hiddens["neurons_indexes"]))[0]]

        if hidden_neuron_id_only.size == 0:
            raise Exception("Ids: {} are not hidden neurons, only hidden neuron are removable".format(neurons_ids))

        if nn.network_type == "SNN":
            # reset all neurons parameters
            nn.neurons_status[hidden_neuron_id_only] = False
            nn.parameters["voltage"][hidden_neuron_id_only] = 0.0
            nn.parameters["threshold"][hidden_neuron_id_only] = 0.0
            nn.parameters["tau"][hidden_neuron_id_only] = 0.0
            nn.parameters["input_current"][hidden_neuron_id_only] = 0.0
            if "refractory" in nn.parameters:
                nn.parameters["refractory"][hidden_neuron_id_only] = 0.0


            # reset all synapses parameters
            nn.synapses_status[hidden_neuron_id_only, :] = False
            nn.synapses_status[:, hidden_neuron_id_only] = False
            nn.parameters["weight"][hidden_neuron_id_only, :] = 0.0
            nn.parameters["weight"][:, hidden_neuron_id_only] = 0.0
            if "delay" in nn.parameters:
                nn.parameters["delay"][hidden_neuron_id_only, :] = 0.0
                nn.parameters["delay"][:, hidden_neuron_id_only] = 0.0

        elif nn.network_type == "ANN":
            # reset all neurons parameters
            nn.neurons_status[hidden_neuron_id_only] = False
            nn.parameters["bias"][hidden_neuron_id_only] = 0.0

            # reset all synapses parameters
            nn.synapses_status[hidden_neuron_id_only, :] = False
            nn.synapses_status[:, hidden_neuron_id_only] = False
            nn.parameters["weight"][hidden_neuron_id_only, :] = 0.0
            nn.parameters["weight"][:, hidden_neuron_id_only] = 0.0

        nn.update_indexes()


    def activate_neurons(self, nn:NN, neurons_ids:np.ndarray):
        nn.neurons_status[neurons_ids] = True
        nn.update_indexes()
  
    def deactivate_neurons(self, nn:NN, neurons_ids:np.ndarray):
        # filter hidden neurons
        if type(neurons_ids) == int:
            neurons_ids = np.array([neurons_ids])
        hidden_neuron_id_only:np.ndarray = neurons_ids[np.where(np.isin(neurons_ids, nn.hiddens["neurons_indexes"]))[0]]

        if hidden_neuron_id_only.size == 0:
            raise Exception("Ids: {} are not hidden neurons, only hidden neuron can be deactivate".format(neurons_ids))

        # reset all neurons parameters
        nn.neurons_status[hidden_neuron_id_only] = False

        # reset all synapses parameters
        nn.synapses_status[hidden_neuron_id_only, :] = False
        nn.synapses_status[:, hidden_neuron_id_only] = False
        nn.update_indexes()
  
        
    def add_synapses(self, nn:NN, neurons_in:np.ndarray, neurons_out:np.ndarray, weights:np.ndarray=1.0, delays:np.ndarray=0.1, set_auto_attributes:bool = True, update_indexes:bool = True):
        if type(neurons_in) == int and type(neurons_out) == int:
            neurons_in = np.array([neurons_in])
            neurons_out = np.array([neurons_out])

        # remove self connection
        neurons_in = neurons_in[neurons_in != neurons_out]
        neurons_out = neurons_out[neurons_in != neurons_out]
        # neurons_in, neurons_out = self.remove_self_connection_jit(neurons_in, neurons_out)


        # add synapses
        nn.synapses_status[neurons_in, neurons_out] = True

        # set synapses attributes
        if set_auto_attributes == True:
            self.set_synapse_attributes(nn, neurons_in, neurons_out)
        else:
            if nn.network_type == "SNN":
                nn.parameters["weight"][neurons_in, neurons_out] = weights
                if "delay" in nn.parameters:
                    nn.parameters["delay"][neurons_in, neurons_out] = delays
            elif nn.network_type == "ANN":
                nn.parameters["weight"][neurons_in, neurons_out] = weights

        if update_indexes == True:
            nn.update_indexes()

    def del_synapses(self, nn:NN, neurons_in:np.ndarray, neurons_out:np.ndarray):
        nn.synapses_status[neurons_in, neurons_out] = False
        if nn.network_type == "SNN":
            nn.parameters["weight"][neurons_in, neurons_out] = 0.0
            if "delay" in nn.parameters:
                nn.parameters["delay"][neurons_in, neurons_out] = 0.0
        elif nn.network_type == "ANN":
            nn.parameters["weight"][neurons_in, neurons_out] = 0.0
        
        nn.update_indexes()


    def activate_synapses(self, nn:NN, neurons_in:np.ndarray, neurons_out:np.ndarray):
        if type(neurons_in) == int and type(neurons_out) == int:
            neurons_in = np.array([neurons_in])
            neurons_out = np.array([neurons_out])

        # remove self connection
        neurons_in = neurons_in[neurons_in != neurons_out]
        neurons_out = neurons_out[neurons_in != neurons_out]
        # neurons_in, neurons_out = self.remove_self_connection_jit(neurons_in, neurons_out)

        # add synapses
        nn.synapses_status[neurons_in, neurons_out] = True

        nn.update_indexes()

    def deactivate_synapses(self, nn:NN, neurons_in:np.ndarray, neurons_out:np.ndarray, update_indexes:bool = True):
        nn.synapses_status[neurons_in, neurons_out] = False
        if update_indexes == True:
            nn.update_indexes()

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def remove_self_connection_jit(neurons_in:np.ndarray, neurons_out:np.ndarray):
        return neurons_in[neurons_in != neurons_out], neurons_out[neurons_in != neurons_out]