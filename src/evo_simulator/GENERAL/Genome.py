from typing import Dict, Any, List
from evo_simulator.GENERAL.SNN import SNN
from evo_simulator.GENERAL.NN import NN
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
import numpy as np
import evo_simulator.TOOLS as TOOLS

class Genome:
    def __init__(self, id:int, config_genome:Dict[str, Any], attributes_manager:Attribute_Paramaters):
        self.id:int = id
        self.fitness:Fitness = Fitness()
        self.info:Dict[str, Any] = {"is_elite":False}
        self.config_genome:Dict[str, Any] = config_genome
        self.attributes_manager:Attribute_Paramaters = attributes_manager


class Genome_Classic(Genome):
    def __init__(self, id:int, config:Dict[str, Any], attributes_manager:Attribute_Paramaters) -> None:
        Genome.__init__(self, id, config, attributes_manager)
        self.parameter:np.ndarray = np.zeros(int(config["parameter_size"]))


class Genome_Decoder(Genome):
    def __init__(self, id:int, config:Dict[str, Any], attributes_manager:Attribute_Paramaters) -> None:
        Genome.__init__(self, id, config, attributes_manager)
        self.decoder_neuron:np.ndarray = np.zeros(int(config["decoder_neuron_size"]))
        self.decoder_synapse:np.ndarray = np.zeros(int(config["decoder_synapse_size"]))



class Genome_NN(Genome):
    def __init__(self, id:int, config_genome:Dict[str, Any], attributes_manager:Attribute_Paramaters, hiddens_active:float=None) -> None:
        Genome.__init__(self, id, config_genome, attributes_manager)

        self.__init_config(config_genome)
        if hiddens_active is not None:
            self.hiddens_active = hiddens_active
        self.nn:NN = NN(
                    nb_inputs=self.inputs, 
                    nb_outputs=self.outputs,
                    nb_hiddens=self.hiddens,
                    nb_hiddens_active=self.hiddens_active, 
                    hiddens_config=self.hiddens_config,
                    hiddens_layer_names=self.hiddens_layer_names,
                    architecture=self.architecture,
                    is_self_neuron_connection=self.is_self_neuron_connection,
                    inputs_multiplicator=self.inputs_multiplicator,
                    outputs_multiplicator=self.outputs_multiplicator, 
                    network_type=self.network_type, 
                    attributes_manager=self.attributes_manager
                    )
        self.net_torch = None
    
    def __init_config(self, config_genome) -> None:
        self.config_genome:Dict[str, Any] = config_genome

        self.inputs:int = int(self.config_genome["inputs"])
        self.inputs_multiplicator:int = max(1, int(self.config_genome["inputs_multiplicator"])) # number of neurons used to represent one input

        self.hiddens:int = TOOLS.hiddens_nb_from_config(self.config_genome["hiddens"])
        self.hiddens_active:int = int(self.config_genome["hiddens_active"])
        self.hiddens_config:List[int] = TOOLS.hiddens_from_config(self.config_genome["hiddens"])


        self.outputs:int = int(self.config_genome["outputs"])
        self.outputs_multiplicator:int = max(1, int(self.config_genome["outputs_multiplicator"])) # number of neurons used to represent one output

        self.architecture, self.hiddens_layer_names = TOOLS.architecture_from_config(self.config_genome["architecture"], len(self.hiddens_config))
        self.architecture_print_orignal:str = self.config_genome["inputs"] + "x" + self.config_genome["hiddens"] + "x" + self.config_genome["outputs"]
        self.architecture_print_multiplied:str = str(self.inputs*self.inputs_multiplicator) + "x" + self.config_genome["hiddens"] + "x" + str(self.outputs*self.outputs_multiplicator)

        self.is_self_neuron_connection:bool = True if self.config_genome["is_self_neuron_connection"] == "True" else False

        self.network_type:str = self.config_genome["network_type"]
        if self.network_type == "ANN":
            self.is_inter_hidden_feedback:bool = True if self.config_genome["is_inter_hidden_feedback"] == "True" else False
            self.is_layer_normalization:bool = True if self.config_genome["is_layer_normalization"] == "True" else False