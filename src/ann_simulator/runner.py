
import torch
import torch.nn as nn
import numpy as np
from evo_simulator import TOOLS
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
from typing import Dict, List, Union, Tuple, Any, Callable
import time
import re

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
                # print("self-feedback")
            
            elif target_name in genome.hiddens_layer_names and source_name != "I": # input feedback to hidden is not allowed, it's a forward connection
                if source_name in genome.hiddens_layer_names and self.is_inter_hidden_feedback == False: # Hidden->to_Hidden
                    pass
                else: # O->to_Hidden
                    self.layers[target_name]["feedback"][source_name] = self.connection[connection_name]
                    self.layers[target_name]["is_feedback"]:bool = True
                    # print("O->to_Hidden")
            
            elif target_name == "I": # Any layer -> I
                self.layers[target_name]["feedback"][source_name] = self.connection[connection_name]
                self.layers[target_name]["is_feedback"]:bool = True
                # print("Any layer -> I")

            
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
                 
        # 1 - Forward

        # 1.1 - Input
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw # -> seems to be better without the bias

        
        # 1.2 - Hidden : input+hidden_feedback -> hidden or hidden_feedback -> hidden
        for hidden_layer in self.hidden_layers.values(): # H_0...H_n-1
            
            # 1.2.0 - input to hidden otherwise hidden_feedback to hidden
            if "I" in hidden_layer["forward"]:
                layer_function:nn.Linear = hidden_layer["forward"]["I"]
                hidden_layer["output"] = torch.relu(layer_function(self.layers["I"]["output"]) + hidden_layer["output"] + hidden_layer["bias"]) # Input+Hidden_feedback to Hidden

            elif hidden_layer["is_feedback"] == True:
                hidden_layer["output"] = torch.relu(hidden_layer["output"] + hidden_layer["bias"]) # Hidden_feedback to Hidden

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
                    hidden_layer["is_inter_hidden_layer"] = False

                    
        
        # 1.3 - Output        
        output_layer:Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]] = self.layers["O"]
        output_layer_output:torch.Tensor = output_layer["output"]

        for layer_name, layer_function in output_layer["forward"].items():
            output_layer_output = output_layer_output + layer_function(self.layers[layer_name]["output"])

        self.layers["O"]["output"] = torch.tanh(output_layer_output + output_layer["bias"])
        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)
        # exit()
            
        
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


class NN_Custom_torch_2(nn.Module):
    def __init__(self, genome:Genome_NN, forward_config:Dict[str, Dict[str, Any]], forward_order:List[str]):
        super(NN_Custom_torch_2, self).__init__()

        self.genome_architecture_layers:List[List[str]] = genome.nn.architecture_layers
        self.genome_architecture_neurons:Dict[str, Dict[str, np.ndarray]] = genome.nn.architecture_neurons
        self.forward_config:Dict[str, Dict[str, Any]] = forward_config
        self.forward_order:List[str] = forward_order

        self.connection:Dict[str, nn.Linear] = {}
        self.layers:Dict[str, Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]]] = {}
        self.hidden_layers:Dict[str, Dict[str, Union[torch.Tensor, Dict[str, nn.Linear]]]] = {}


        self.layers_name_forward:List[str] = ["I"] + genome.hiddens_layer_names + ["O"]
        self.hidden_layer_names:List[str] = genome.hiddens_layer_names


        for layer_name in self.layers_name_forward:
            self.layers[layer_name] = {}
            self.layers[layer_name]["output"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["output_prev"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["bias"]:torch.Tensor = torch.zeros(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["norm"]:nn.LayerNorm = nn.LayerNorm(self.genome_architecture_neurons[layer_name]["size"]).to(device)
            self.layers[layer_name]["size"]:int = self.genome_architecture_neurons[layer_name]["size"]

            if layer_name not in ["I", "O"]:
                self.hidden_layers[layer_name] = self.layers[layer_name]

        for source_name, target_name in self.genome_architecture_layers:
            connection_name:str = source_name + "->" + target_name
            self.connection[connection_name] = nn.Linear(self.genome_architecture_neurons[source_name]["size"], self.genome_architecture_neurons[target_name]["size"], bias=False, device=device)
            for param in self.connection[connection_name].parameters():
                param.requires_grad = False

        
        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)

        # print("\nhidden_layers:\n")
        # for layer_name, info in self.hidden_layers.items():
        #     print(layer_name, "->", info, "\n")

        # print("\nforward_config:", self.forward_config)
        # print("\nforward_order:", self.forward_order)
        # exit()


    def forward_debug(self, input_raw:torch.Tensor):
        
        # 0.0 - Reset output tensor layers        
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(device)
            layer["output"] = output_layer

        # 0.1 - Add input raw to input layer
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw

        # 1 - Forward
        for order in self.forward_order:

            input_activation:Callable = self.forward_config[order]["layer_input"]["activation"]
            input_norm:bool = self.forward_config[order]["layer_input"]["norm"]
            for input_layer_config in self.forward_config[order]["layer_input"]["layer"]:
                input_layer_name, input_tensor_name = input_layer_config
                input_tensor:torch.Tensor = self.layers[input_layer_name][input_tensor_name]
                print("order:", order)
                print("\ninput_layer_name:", input_layer_name)
                print("input_tensor_name:", input_tensor_name)
                print("input_tensor:", input_tensor)
                print("input_activation:", input_activation)
                print("input_norm:", input_norm)
                for output_layer_name, output_layer_config in self.forward_config[order]["layer_output"].items():
                    output_activation:Callable = output_layer_config["activation"]
                    output_norm:bool = output_layer_config["norm"]
                    output_tensor:torch.Tensor = self.layers[output_layer_name]["output"]
                    connection_weight:nn.Linear = self.connection[input_layer_name + "->" + output_layer_name]
                    print("\noutput_layer_name:", output_layer_name)
                    print("output_activation:", output_activation)
                    print("output_norm:", output_norm)
                    print("output_tensor before activation:\n", output_tensor)
                    if input_norm == True or output_norm == True:
                        norm_function:nn.LayerNorm = self.layers[output_layer_name]["norm"]
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(norm_function(connection_weight(input_tensor))))
                    else:
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(connection_weight(input_tensor)))
                    print("\noutput_tensor after activation:\n", self.layers[output_layer_name]["output"]," \n")

        # 2 - Output Layer
        output_activation:Callable = self.forward_config["O"]["activation"]
        output_norm:bool = self.forward_config["O"]["norm"]
        output_tensor:torch.Tensor = self.layers["O"]["output"]
        print("\nO before activation\n:", self.layers["O"]["output"])
        if output_norm == True:
            norm_function:nn.LayerNorm = self.layers["O"]["norm"]
            self.layers["O"]["output"] = output_activation(norm_function(output_tensor))
        else:
            self.layers["O"]["output"] = output_activation(output_tensor)
        print("\nO after activation\n:", self.layers["O"]["output"])

        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)

        exit()
        # 3 - Update output_prev
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]

        return self.layers["O"]["output"].cpu()
                    

    def forward(self, input_raw:torch.Tensor):
        
        # 0.0 - Reset output tensor layers
        for layer in self.layers.values(): # I -> H_0...H_n-1 -> O
            output_layer:torch.Tensor = torch.zeros(layer["size"]).to(device)
            layer["output"] = output_layer

        # 0.1 - Add input raw to input layer
        self.layers["I"]["output"] = self.layers["I"]["output"] + input_raw

        # 1 - Forward
        for order in self.forward_order:

            input_activation:Callable = self.forward_config[order]["layer_input"]["activation"]
            input_norm:bool = self.forward_config[order]["layer_input"]["norm"]
            for input_layer_config in self.forward_config[order]["layer_input"]["layer"]:
                input_layer_name, input_tensor_name = input_layer_config
                input_tensor:torch.Tensor = self.layers[input_layer_name][input_tensor_name]

                for output_layer_name, output_layer_config in self.forward_config[order]["layer_output"].items():
                    output_activation:Callable = output_layer_config["activation"]
                    output_norm:bool = output_layer_config["norm"]
                    output_tensor:torch.Tensor = self.layers[output_layer_name]["output"]
                    connection_weight:nn.Linear = self.connection[input_layer_name + "->" + output_layer_name]

                    if input_norm == True or output_norm == True:
                        norm_function:nn.LayerNorm = self.layers[output_layer_name]["norm"]
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(norm_function(connection_weight(input_tensor))))
                    else:
                        self.layers[output_layer_name]["output"] = output_tensor + output_activation(input_activation(connection_weight(input_tensor)))

        # 2 - Output Layer
        output_activation:Callable = self.forward_config["O"]["activation"]
        output_norm:bool = self.forward_config["O"]["norm"]
        output_tensor:torch.Tensor = self.layers["O"]["output"]

        if output_norm == True:
            norm_function:nn.LayerNorm = self.layers["O"]["norm"]
            self.layers["O"]["output"] = output_activation(norm_function(output_tensor))
        else:
            self.layers["O"]["output"] = output_activation(output_tensor)

        # 3 - Update output_prev
        for layer in self.layers.values():
            layer["output_prev"] = layer["output"]
        
        # print("\n ALL connection:\n", self.connection)
        # for layer_name, info in self.layers.items():
        #     print("\n", layer_name, "->", info)

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


device = "cpu" # cuda or cpu

class ANN_Runner():
    def __init__(self, config_path:str):
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN"])
        self.weights_layers_indexes:Dict[str, Tuple[np.ndarray, np.ndarray]] = None
        self.forward_config, self.forward_order = self.set_forward_config(self.config["Genome_NN"]["forward"])


    def get_weight_layers_indexes(self, synapses_indexes:Tuple[np.ndarray, np.ndarray], source_indexes:np.ndarray, target_indexes:np.ndarray) -> np.ndarray:
        input_synpases_indexes = np.where(np.isin(synapses_indexes[0], source_indexes))[0]
        target_synapse_indexes = np.where(np.isin(synapses_indexes[1][input_synpases_indexes], target_indexes))[0]
        return (synapses_indexes[0][input_synpases_indexes[target_synapse_indexes]], synapses_indexes[1][input_synpases_indexes[target_synapse_indexes]])


    def set_net_torch_population(self, population:Dict[int, Genome_NN], is_bias:bool) -> None:
        for genome in population.values():

            # 1 - New version
            # genome.net_torch:NN_Custom_torch = NN_Custom_torch(genome).to(device)
            genome.net_torch:NN_Custom_torch_2 = NN_Custom_torch_2(genome, self.forward_config, self.forward_order).to(device)
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
    

    def set_forward_config(self, forward_from_config:str) -> Dict[str, Dict[str, Any]]:
        forward_order = []
        forward_config:Dict[str, Dict[str, Any]] = {}
        for layer in re.split(r',(?![^()]*\))', forward_from_config): # thanks gpt for the regex...
            forward_order.append(layer.replace(" ", ""))
        index_O = None
        for index, forward in enumerate(forward_order):
            forward_config.update(self.get_layers(forward))
            if "->" not in forward and "O_" in forward:
                index_O = index
        if index_O != None:
            forward_order.pop(index_O)
        if "O" not in forward_config:
            forward_config["O"] = {"activation": self.get_activation(["raw"]), "norm": False}
        return forward_config, forward_order

    def get_layers(self, forward:str):
        layers:List[str] = []
        param:List[str] = []
        forward_config:Dict[str, Dict[str, Any]] = {}

        # 0. Set O layer config
        if ("O" == forward or "O_" in forward) and "->" not in forward:
            forward_config["O"] = {"activation": self.get_activation(forward.split("_")[1:]), "norm": "norm" in forward}
            return forward_config

        forward_config[forward] = {"layer_input": {}, "layer_output": {}}
        layers_and_param:List[str] = forward.split("->")
        
        # 1. Set Layers INTPUTs config
        layers_and_param_in:str = layers_and_param[0]
        if "(" not in layers_and_param_in:
            layers.append(layers_and_param_in) # get layers
            param.append(None) # in this case, there is no param
        else:
            layers = layers_and_param_in.split("(")[1].split(")")[0].split(",") # get layers
            param = layers_and_param_in.split("(")[1].split(")")[1].split("_") # get param (e.g "relu_norm" -> ["relu", "norm"])
            param = [x for x in param if x] # remove empty string

        layer_conf = []
        for layer in layers:
            if "_prev" in layer or "_prev_" in layer: # get if use previous output (e.g "H1_prev" -> ["H1", "output_prev"])
                layer_conf.append([layer.replace("_prev_", '').replace("_prev", ''), "output_prev"])
            else: # get if use current output (e.g "H1" -> ["H1", "output"])
                layer_conf.append([layer, "output"])
        layers = layer_conf


        forward_config[forward]["layer_input"]["layer"] = layers
        forward_config[forward]["layer_input"]["activation"] = self.get_activation(param)
        forward_config[forward]["layer_input"]["norm"] = "norm" in param


        # 2. Set Layers OUTPUTs config
        layers_and_param_out:str = layers_and_param[1]
        if "(" not in layers_and_param_out:
            layer_name = layers_and_param_out.split("_")[0]
            param = layers_and_param_out.split("_")[1:]
            activation = self.get_activation(param)
            norm = "norm" in param
            forward_config[forward]["layer_output"][layer_name] = {"activation": activation, "norm": norm}
        else:
            elements = layers_and_param_out.replace("(", "").replace(")", "").split(",")
            for element in elements:
                layer_name = element.split("_")[0]
                param = element.split("_")[1:]
                activation = self.get_activation(param)
                norm = "norm" in param
                forward_config[forward]["layer_output"][layer_name] = {"activation": activation, "norm": norm}
        return forward_config

    def get_activation(self, activation_list:List[str]) -> Callable:
        for activation in activation_list:
            if activation == "relu":
                return torch.relu
            elif activation == "tanh":
                return torch.tanh
            elif activation == "sigmoid":
                return torch.sigmoid
            elif activation == "raw":
                return lambda x: x
        return lambda x: x

# def check_values(values):
#     if np.any(np.isnan(values)) or np.any(np.isinf(values)) or np.any(np.abs(values) > 100_000_000):
#         print("values:", values)
#         raise ValueError("Invalid value detected for actuator")
