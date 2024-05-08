from problem.RL.ENVIRONNEMENT import Environment
from evo_simulator import TOOLS
from typing import  Dict, Any
import numpy as np

class Config_Problem():
    def __init__(self, name:str, nb_inputs:int, nb_outputs:int, config_path:str, obs_max_init_value:float = 5.0, obs_min_init_value:float = -5.0, termination_fitness:float = None, auto_obersvation:bool = False, auto_obersvation_ratio:float=0.01):
        self.name:str = name
        self.nb_inputs:int = nb_inputs
        self.nb_outputs:int = nb_outputs
        self.config_path:str = config_path
        self.termination_fitness:float = termination_fitness
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN", "NEURO_EVOLUTION"])
        self.observation_max_global:np.ndarray = np.full(nb_inputs, obs_max_init_value, dtype=np.float64) # 5 is arbitrary
        self.observation_min_global:np.ndarray = np.full(nb_inputs, obs_min_init_value, dtype=np.float64) # -5 is arbitrary
        self.auto_obersvation:bool = auto_obersvation
        self.auto_obersvation_ratio:float = auto_obersvation_ratio

        nb_inputs_config:int = int(self.config["Genome_NN"]["inputs"])
        nb_outputs_config:int = int(self.config["Genome_NN"]["outputs"])
        if nb_inputs_config != nb_inputs:
            raise Exception("The number of inputs in the config file (", nb_inputs_config ,") is different from the number of inputs in the problem (", nb_inputs,")")
        if nb_outputs_config != nb_outputs:
            raise Exception("The number of outputs in the config file (", nb_outputs_config ,") is different from the number of outputs in the problem (", nb_outputs,")")
        
    def get_env(self) -> Environment:
        raise NotImplementedError("get_env() not implemented")
    
    def encoding_observation_to_input_network(self, observation: np.ndarray, observation_max:np.ndarray, observation_min:np.ndarray) -> np.ndarray:
        raise NotImplementedError("encoding_observation_to_input_network() not implemented")
    
    def decoding_output_network_to_action(self, action: np.ndarray) -> int:
        raise NotImplementedError("decoding_output_network_to_action() not implemented")

    def fitness_step(self, genome, episode:int, info:Dict[str, Any]) -> None:
        raise NotImplementedError("fitness_step() not implemented")

    def fitness_end(self, genome, episode:int, info:Dict[str, Any]) -> None:
        raise NotImplementedError("fitness_end() not implemented")
    
    def update_observation_min_max(self, obs_max:np.ndarray, obs_min:np.ndarray) -> None:
        self.observation_max_global = obs_max
        self.observation_min_global = obs_min
