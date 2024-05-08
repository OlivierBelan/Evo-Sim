
import torch
import torch.nn as nn
import numpy as np
from evo_simulator.GENERAL.Population import Population_NN
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
from typing import Dict, List, Union, Tuple
import time
# import warnings
# warnings.filterwarnings("ignore")


class NN_Custom_torch(nn.Module):
    def __init__(self, genome:Genome_NN):
        super(NN_Custom_torch, self).__init__()

        self.genome_architecture_layers:List[List[str]] = genome.nn.architecture_layers
        self.genome_architecture_neurons:Dict[str, Dict[str, np.ndarray]] = genome.nn.architecture_neurons

        self.connection:Dict[str, nn.Linear] = {}
        self.layers:Dict[str, Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]]] = {}
        self.hidden_layers:Dict[str, Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]]] = {}


        self.layers_name_forward:List[str] = ["I"] + genome.hiddens_layer_names + ["O"]
        self.hidden_layer_names:List[str] = genome.hiddens_layer_names

        self.is_inter_hidden_feedback:bool = genome.is_inter_hidden_feedback
        self.is_layer_normalization:bool = genome.is_layer_normalization

        for layer_name in self.layers_name_forward:
            self.layers[layer_name] = {}
            self.layers[layer_name]["output"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["output_prev"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["bias"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["layer_norm"]:nn.LayerNorm = nn.LayerNorm(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["forward"]:Dict[str, nn.Linear] = {}
            self.layers[layer_name]["feedback"]:Dict[str, nn.Linear] = {}
            self.layers[layer_name]["is_feedback"]:bool = False
            self.layers[layer_name]["is_inter_hidden_layer"]:bool = False
            self.layers[layer_name]["size"]:int = self.genome_architecture_neurons[layer_name]["size"]
            if layer_name not in ["I", "O"]:
                self.hidden_layers[layer_name] = self.layers[layer_name]

        for source_name, target_name in self.genome_architecture_layers:
            connection_name:str = source_name + "->" + target_name
            self.connection[connection_name] = nn.Linear(self.genome_architecture_neurons[source_name]["size"], self.genome_architecture_neurons[target_name]["size"], bias=False, device=device)
            for param in self.connection[connection_name].parameters():
                param.requires_grad = False

            # 1 - Feedback case
            if source_name == target_name: # e.g I->I or H1->H1 or O->O
                self.layers[target_name]["feedback"][source_name] = self.connection[connection_name]
                self.layers[target_name]["is_feedback"]:bool = True
            
            elif target_name in genome.hiddens_layer_names and source_name != "I": # input feedback to hidden is not allowed, it's a forward connection
                if source_name in genome.hiddens_layer_names and self.is_inter_hidden_feedback == False:
                    pass
                else:
                    self.layers[target_name]["feedback"][source_name] = self.connection[connection_name]
                    self.layers[target_name]["is_feedback"]:bool = True
            
            elif target_name == "I":
                self.layers[target_name]["feedback"][source_name] = self.connection[connection_name]
                self.layers[target_name]["is_feedback"]:bool = True

            
            # 2 - Forward case            
            elif target_name == "O":
                self.layers[target_name]["forward"][source_name] = self.connection[connection_name]
        
            if target_name in genome.hiddens_layer_names and source_name != "O" and source_name != target_name:
                self.layers[target_name]["forward"][source_name] = self.connection[connection_name]
        
        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)
        
        # print("\nhidden_layers:\n")
        # for layer_name, info in self.hidden_layers.items():
        #     print(layer_name, "->", info, "\n")
        # exit()
        
    def forward(self, input_raw:torch.Tensor):
        if self.is_layer_normalization == True:
            return self.forward_layer_norm(input_raw)
        else:
            return self.forward_no_layer_norm(input_raw)

    def forward_no_layer_norm(self, input_raw:torch.Tensor):
        # 0 - Feedback
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            # 0.1 - Reset
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(device)

            # 0.2 - Feedback
            if layer["is_feedback"] == True:
                for feedback_layer_name, layer_function in layer["feedback"].items():
                    output_layer = output_layer + layer_function(self.layers[feedback_layer_name]["output_prev"])

            # 0.3 - Set
            layer["output"] = output_layer
            # check_values(layer["output"].detach().numpy())
                 
        # 1 - Forward

        # 1.1 - Input
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw # -> seems to be better without the bias
        # check_values(self.layers["I"]["output"].detach().numpy())

        
        # 1.2 - Hidden -> input+hidden_feedback to hidden or hidden_feedback to hidden
        for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
            
            # 1.2.0 - input to hidden otherwise hidden_feedback to hidden
            if "I" in hidden_layer["forward"]:
                layer_function:nn.Linear = hidden_layer["forward"]["I"]
                hidden_layer["output"] = torch.relu(layer_function(self.layers["I"]["output"]) + hidden_layer["output"] + hidden_layer["bias"])
                # check_values(hidden_layer["output"].detach().numpy())

            elif hidden_layer["is_feedback"] == True:
                hidden_layer["output"] = torch.relu(hidden_layer["output"] + hidden_layer["bias"])
                # check_values(hidden_layer["output"].detach().numpy())

        # 1.2.1 - possible hidden to hidden
        is_inter_hidden_layer:bool = False
        for hidden_layer_1 in self.hidden_layers.values(): # H_0...H_n-1
        
            for hidden_name_2, hidden_layer_2 in self.hidden_layers.items(): # H_0...H_n-1
                if hidden_name_2 in hidden_layer_1["forward"]:
                    if hidden_layer_1["is_inter_hidden_layer"] == False:
                        hidden_layer_1["intermetiate_output"] = torch.zeros(hidden_layer_1["size"]).to(device)
                        hidden_layer_1["is_inter_hidden_layer"] = True
                        is_inter_hidden_layer = True

                    layer_function:nn.Linear = hidden_layer_1["forward"][hidden_name_2]
                    output_layer:torch.Tensor = hidden_layer_2["output"]
                    hidden_layer_1["intermetiate_output"] = hidden_layer_1["intermetiate_output"] + layer_function(output_layer)
        
        # 1.2.2 - update hidden if inter_hidden_layer
        if is_inter_hidden_layer == True:
            for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
                if hidden_layer["is_inter_hidden_layer"] == True:
                    hidden_layer["output"] = torch.relu(hidden_layer["intermetiate_output"] + hidden_layer["bias"])
                    # check_values(hidden_layer["output"].detach().numpy())
                    hidden_layer["is_inter_hidden_layer"] = False

                    
        
        # 1.3 - Output        
        output_layer:Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]] = self.layers["O"]
        output_layer_output:torch.Tensor = output_layer["output"]

        for layer_name, layer_function in output_layer["forward"].items():
            output_layer_output = output_layer_output + layer_function(self.layers[layer_name]["output"])

        self.layers["O"]["output"] = torch.tanh(output_layer_output + output_layer["bias"])
        # check_values(self.layers["O"]["output"].detach().numpy())
            
        
        # 2 - Update output_prev -> reflexion: pour le feedback dois-t-on utiliser la fonction d'activation ou non ? 
        # est-ce que le fonction d'activation à un impact positif ou négatif sur la performance ?
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]


        return self.layers["O"]["output"].cpu()

    def forward_layer_norm(self, input_raw:torch.Tensor):
        # 0 - Feedback
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            # 0.1 - Reset
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(device)

            # 0.2 - Feedback
            if layer["is_feedback"] == True:
                for feedback_layer_name, layer_function in layer["feedback"].items():
                    output_layer = output_layer + layer_function(self.layers[feedback_layer_name]["output_prev"])

            # 0.3 - Set
            layer["output"] = layer["layer_norm"](output_layer)
            # check_values(layer["output"].detach().numpy())
                 
        # 1 - Forward

        # 1.1 - Input
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw # -> seems to be better without the bias
        # check_values(self.layers["I"]["output"].detach().numpy())

        
        # 1.2 - Hidden -> input+hidden_feedback to hidden or hidden_feedback to hidden
        for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
            
            # 1.2.0 - input to hidden otherwise hidden_feedback to hidden
            if "I" in hidden_layer["forward"]:
                layer_function:nn.Linear = hidden_layer["forward"]["I"]
                hidden_layer["output"] = torch.relu(hidden_layer["layer_norm"](layer_function(self.layers["I"]["output"]) + hidden_layer["output"] + hidden_layer["bias"]))
                # check_values(hidden_layer["output"].detach().numpy())

            elif hidden_layer["is_feedback"] == True:
                hidden_layer["output"] = torch.relu(hidden_layer["layer_norm"](hidden_layer["output"] + hidden_layer["bias"]))
                # check_values(hidden_layer["output"].detach().numpy())

        # 1.2.1 - possible hidden to hidden
        is_inter_hidden_layer:bool = False
        for hidden_layer_1 in self.hidden_layers.values(): # H_0...H_n-1
        
            for hidden_name_2, hidden_layer_2 in self.hidden_layers.items(): # H_0...H_n-1
                if hidden_name_2 in hidden_layer_1["forward"]:
                    if hidden_layer_1["is_inter_hidden_layer"] == False:
                        hidden_layer_1["intermetiate_output"] = torch.zeros(hidden_layer_1["size"]).to(device)
                        hidden_layer_1["is_inter_hidden_layer"] = True
                        is_inter_hidden_layer = True

                    layer_function:nn.Linear = hidden_layer_1["forward"][hidden_name_2]
                    output_layer:torch.Tensor = hidden_layer_2["output"]
                    hidden_layer_1["intermetiate_output"] = hidden_layer_1["intermetiate_output"] + layer_function(output_layer)
        
        # 1.2.2 - update hidden if inter_hidden_layer
        if is_inter_hidden_layer == True:
            for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
                if hidden_layer["is_inter_hidden_layer"] == True:
                    hidden_layer["output"] = torch.relu(hidden_layer["layer_norm"](hidden_layer["intermetiate_output"] + hidden_layer["bias"]))
                    # check_values(hidden_layer["output"].detach().numpy())
                    hidden_layer["is_inter_hidden_layer"] = False

                    
        
        # 1.3 - Output        
        output_layer:Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]] = self.layers["O"]
        output_layer_output:torch.Tensor = output_layer["output"]

        for layer_name, layer_function in output_layer["forward"].items():
            output_layer_output = output_layer_output + layer_function(self.layers[layer_name]["output"])

        self.layers["O"]["output"] = torch.tanh(self.layers["O"]["layer_norm"](output_layer_output + output_layer["bias"]))
        # check_values(self.layers["O"]["output"].detach().numpy())
            
        
        # 2 - Update output_prev -> reflexion: pour le feedback dois-t-on utiliser la fonction d'activation ou non ? 
        # est-ce que le fonction d'activation à un impact positif ou négatif sur la performance ?
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]


        return self.layers["O"]["output"].cpu()

    def forward_old(self, input_raw:torch.Tensor):

        # 0 - Feedback
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            # 0.1 - Reset
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(device)

            # 0.2 - Feedback
            if layer["is_feedback"] == True:
                for feedback_layer_name, layer_function in layer["feedback"].items():
                    output_layer = output_layer + layer_function(self.layers[feedback_layer_name]["output_prev"])

            # 0.3 - Set
            layer["output"] = layer["layer_norm"](output_layer)
            # check_values(layer["output"].detach().numpy())
                 
        # 1 - Forward

        # 1.1 - Input
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw # -> seems to be better without the bias
        # self.layers["I"]["output"] = torch.relu(self.layers["I"]["output"] + input_raw) # -> seems to be better without the bias

        # self.layers["I"]["output"] = self.layers["I"]["output"] + self.layers["I"]["bias"] + input_raw
        # self.layers["I"]["output"] = torch.relu(self.layers["I"]["output"] + self.layers["I"]["bias"] + input_raw)

        # check_values(self.layers["I"]["output"].detach().numpy())

        
        # 1.2 - Hidden -> input+hidden_feedback to hidden or hidden_feedback to hidden
        for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
            
            # 1.2.0 - input to hidden otherwise hidden_feedback to hidden
            if "I" in hidden_layer["forward"]:
                layer_function:nn.Linear = hidden_layer["forward"]["I"]
                hidden_layer["output"] = torch.relu(hidden_layer["layer_norm"](layer_function(self.layers["I"]["output"]) + hidden_layer["output"] + hidden_layer["bias"]))
                # hidden_layer["output"] = torch.relu(layer_function(self.layers["I"]["output"]) + hidden_layer["output"] + hidden_layer["bias"])
                # check_values(hidden_layer["output"].detach().numpy())

            elif hidden_layer["is_feedback"] == True:
                hidden_layer["output"] = torch.relu(hidden_layer["layer_norm"](hidden_layer["output"] + hidden_layer["bias"]))
                # hidden_layer["output"] = torch.relu(hidden_layer["output"] + hidden_layer["bias"])
                # check_values(hidden_layer["output"].detach().numpy())

        # 1.2.1 - possible hidden to hidden
        is_inter_hidden_layer:bool = False
        for hidden_layer_1 in self.hidden_layers.values(): # H_0...H_n-1
        
            for hidden_name_2, hidden_layer_2 in self.hidden_layers.items(): # H_0...H_n-1
                if hidden_name_2 in hidden_layer_1["forward"]:
                    if hidden_layer_1["is_inter_hidden_layer"] == False:
                        hidden_layer_1["intermetiate_output"] = torch.zeros(hidden_layer_1["size"]).to(device)
                        hidden_layer_1["is_inter_hidden_layer"] = True
                        is_inter_hidden_layer = True

                    layer_function:nn.Linear = hidden_layer_1["forward"][hidden_name_2]
                    output_layer:torch.Tensor = hidden_layer_2["output"]
                    hidden_layer_1["intermetiate_output"] = hidden_layer_1["intermetiate_output"] + layer_function(output_layer)
        
        # 1.2.2 - update hidden if inter_hidden_layer
        if is_inter_hidden_layer == True:
            for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
                if hidden_layer["is_inter_hidden_layer"] == True:
                    hidden_layer["output"] = torch.relu(hidden_layer["layer_norm"](hidden_layer["intermetiate_output"] + hidden_layer["bias"]))
                    # hidden_layer["output"] = torch.relu(hidden_layer["intermetiate_output"] + hidden_layer["bias"])
                    # check_values(hidden_layer["output"].detach().numpy())
                    hidden_layer["is_inter_hidden_layer"] = False

                    
        
        # 1.3 - Output        
        output_layer:Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]] = self.layers["O"]
        output_layer_output:torch.Tensor = output_layer["output"]

        for layer_name, layer_function in output_layer["forward"].items():
            output_layer_output = output_layer_output + layer_function(self.layers[layer_name]["output"])

        self.layers["O"]["output"] = torch.tanh(self.layers["O"]["layer_norm"](output_layer_output + output_layer["bias"]))
        # self.layers["O"]["output"] = torch.tanh(output_layer_output + output_layer["bias"])
        # check_values(self.layers["O"]["output"].detach().numpy())
            
        
        # 2 - Update output_prev -> reflexion: pour le feedback dois-t-on utiliser la fonction d'activation ou non ? 
        # est-ce que le fonction d'activation à un impact positif ou négatif sur la performance ?
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]


        return self.layers["O"]["output"].cpu()

    def get_weight_layers(self, weight:np.ndarray, synapses_indexes:Tuple[np.ndarray, np.ndarray], source_indexes:np.ndarray, target_indexes:np.ndarray) -> np.ndarray:
        input_synpases_indexes = np.where(np.isin(synapses_indexes[0], source_indexes))[0]
        target_synapse_indexes = np.where(np.isin(synapses_indexes[1][input_synpases_indexes], target_indexes))[0]
        weight_sub = weight[synapses_indexes[0][input_synpases_indexes[target_synapse_indexes]], synapses_indexes[1][input_synpases_indexes[target_synapse_indexes]]]
        return weight_sub

    def set_parameters(self, genome:Genome_NN, is_bias:bool = False) -> None:
        g_nn:NN = genome.nn
        weights:np.ndarray = g_nn.parameters["weight"]
        synapses_indexes:Tuple[np.ndarray, np.ndarray] = g_nn.synapses_indexes

        for source_name, target_name in self.genome_architecture_layers:
            weights_connection_nn:np.ndarray = self.get_weight_layers(weights, synapses_indexes, self.genome_architecture_neurons[source_name]["neurons_indexes"], self.genome_architecture_neurons[target_name]["neurons_indexes"])
            weights_connection_torch:torch.Tensor = self.connection[source_name + "->" + target_name].weight.data
            self.connection[source_name + "->" + target_name].weight.data = torch.tensor(weights_connection_nn).reshape(weights_connection_torch.shape).to(device)
        
        if is_bias == True:
            biases:np.ndarray = g_nn.parameters["bias"]
            biases_index:int = 0
            for layer in self.layers.values():
                layer["bias"] = torch.tensor(biases[biases_index:biases_index+layer["size"]]).to(device)
                biases_index += layer["size"]

    def set_parameters_2(self, genome:Genome_NN, weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]], is_bias:bool = False) -> None:
        g_nn:NN = genome.nn
        weights:np.ndarray = g_nn.parameters["weight"].copy()
        synapses_unactive_indexes:Tuple[np.ndarray, np.ndarray] = g_nn.synapses_unactives_weight_indexes
        weights[synapses_unactive_indexes[0], synapses_unactive_indexes[1]] = 0.0 # set unactive synapses to 0.0 (for the NEAT algorithm)
        for source_name, target_name in self.genome_architecture_layers:
            connection_key:str = source_name + "->" + target_name
            self.connection[connection_key].weight.data = torch.tensor(weights[weights_layers_indexes[connection_key]]).reshape(self.connection[connection_key].weight.data.shape).to(device)
        if is_bias == True:
            biases:np.ndarray = g_nn.parameters["bias"]
            biases_index:int = 0
            for layer in self.layers.values():
                layer["bias"] = torch.tensor(biases[biases_index:biases_index+layer["size"]]).to(device)
                biases_index += layer["size"]

device = "cpu"

class ANN_Runner():
    def __init__(self, config_path_file:str):
        self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = None

    def get_weight_layers_indexes(self, synapses_indexes:Tuple[np.ndarray, np.ndarray], source_indexes:np.ndarray, target_indexes:np.ndarray) -> np.ndarray:
        input_synpases_indexes = np.where(np.isin(synapses_indexes[0], source_indexes))[0]
        target_synapse_indexes = np.where(np.isin(synapses_indexes[1][input_synpases_indexes], target_indexes))[0]
        return (synapses_indexes[0][input_synpases_indexes[target_synapse_indexes]], synapses_indexes[1][input_synpases_indexes[target_synapse_indexes]])


    def set_net_torch_population(self, population:Dict[int, Genome_NN], is_bias:bool) -> None:
        for genome in population.values():

            # 1 - New version
            genome.net_torch:NN_Custom_torch = NN_Custom_torch(genome).to(device)
            if self.weights_layers_indexes == None:
                self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for source_name, target_name in genome.nn.architecture_layers:
                    self.weights_layers_indexes[source_name + "->" + target_name] = self.get_weight_layers_indexes(genome.nn.synapses_indexes, genome.nn.architecture_neurons[source_name]["neurons_indexes"], genome.nn.architecture_neurons[target_name]["neurons_indexes"])
            genome.net_torch.set_parameters_2(genome, self.weights_layers_indexes, is_bias)

            # # 2 - New old version
            # genome.net_torch:NN_Custom_torch_2 = NN_Custom_torch_2(genome).to(device)
            # genome.net_torch.set_parameters(genome, is_bias)

            # # 3 - Old version
            # genome.net_torch:NN_Custom_torch = NN_Custom_torch(13, 20, 3).to(device)
            # genome.net_torch.set_parameters(genome, is_bias)


    def unset_net_torch_population(self, population:Dict[int, Genome_NN]) -> None:
        for genome in population.values():
            genome.net_torch = None
    

    def run_SL(self, population:Dict[int, Genome_NN], inputs:np.ndarray) -> Dict[int, np.ndarray]:
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        results:Dict[int, np.ndarray] = {}
        for genome in population.values():
            results[genome.id] = genome.net_torch(inputs).detach().numpy()

        return results
    
    def run_RL(self, population:Dict[int, Genome_NN], inputs:Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        results:Dict[int, np.ndarray] = {}
        for genome_id, genome in population.items():
            results[genome_id] = genome.net_torch(torch.tensor(inputs[genome_id], dtype=torch.float32).to(device)).detach().numpy()

        return results
    
# def check_values(values):
#     if np.any(np.isnan(values)) or np.any(np.isinf(values)) or np.any(np.abs(values) > 100_000_000):
#         print("values:", values)
#         raise ValueError("Invalid value detected for actuator")
