import numpy as np
from typing import Any, Dict, List
import evo_simulator.TOOLS as TOOLS

class Attribute_Paramaters:
    def __init__(self, config_path_file:str):
        self.__config_all:Dict[str, Dict[str, Any]] = TOOLS.config_function_all(config_path_file)

        self.parameters_all_names:List[str] = []
        self.parameters_neuron_names:List[str] = []
        self.parameters_synapse_names:List[str] = []
        
        self.parameters_all_names_original:List[str] = []
        self.parameters_neuron_names_original:List[str] = []
        self.parameters_synapse_names_original:List[str] = []

        self.get_parameters_names(self.__config_all)
        # print("self.parameters_names", self.parameters_names,'\n')

        # General (contain all parameters information)
        self.mu_parameters:Dict[str, np.ndarray] = {}
        self.mu_max_parameters:Dict[str, np.ndarray] = {}
        self.mu_min_parameters:Dict[str, np.ndarray] = {}

        self.sigma_parameters:Dict[str, np.ndarray] = {}
        self.sigma_max_parameters:Dict[str, np.ndarray] = {}
        self.sigma_min_parameters:Dict[str, np.ndarray] = {}
        self.sigma_decay_parameters:Dict[str, np.ndarray] = {}

        self.max_parameters:Dict[str, np.ndarray] = {}
        self.min_parameters:Dict[str, np.ndarray] = {}

        # Neuron (Cointain only neuron parameters information)
        self.mu_neuron_parameters:Dict[str, np.ndarray] = {}
        self.mu_neuron_max_parameters:Dict[str, np.ndarray] = {}
        self.mu_neuron_min_parameters:Dict[str, np.ndarray] = {}

        self.sigma_neuron_parameters:Dict[str, np.ndarray] = {}
        self.sigma_neuron_max_parameters:Dict[str, np.ndarray] = {}
        self.sigma_neuron_min_parameters:Dict[str, np.ndarray] = {}
        self.sigma_neuron_decay_parameters:Dict[str, np.ndarray] = {}

        self.max_neuron_parameters:Dict[str, np.ndarray] = {}
        self.min_neuron_parameters:Dict[str, np.ndarray] = {}

        # Synapse (Cointain only synpase parameters information)
        self.mu_synapse_parameters:Dict[str, np.ndarray] = {}
        self.mu_synapse_max_parameters:Dict[str, np.ndarray] = {}
        self.mu_synapse_min_parameters:Dict[str, np.ndarray] = {}

        self.sigma_synapse_parameters:Dict[str, np.ndarray] = {}
        self.sigma_synapse_max_parameters:Dict[str, np.ndarray] = {}
        self.sigma_synapse_min_parameters:Dict[str, np.ndarray] = {}
        self.sigma_synapse_decay_parameters:Dict[str, np.ndarray] = {}

        self.max_synapse_parameters:Dict[str, np.ndarray] = {}
        self.min_synapse_parameters:Dict[str, np.ndarray] = {}
        

        # self.mu_bias_parameters:Dict|[str, np.ndarray] = {} # (addition)
        # self.sigma_bias_parameters:Dict|[str, np.ndarray] = {} # (multiplication)
        for i in range(len(self.parameters_all_names)):
            self.__set_config_all_parameters(self.parameters_all_names_original[i], self.parameters_all_names[i])
        for i in range(len(self.parameters_neuron_names)):
            self.__set_config_neuron_parameters(self.parameters_neuron_names_original[i], self.parameters_neuron_names[i])
        for i in range(len(self.parameters_synapse_names)):
            self.__set_config_synapse_parameters(self.parameters_synapse_names_original[i], self.parameters_synapse_names[i])
        
        # print("mu_parameters", self.mu_parameters,'\n')
        # print("mu_max_parameters", self.mu_max_parameters,'\n')
        # print("mu_min_parameters", self.mu_min_parameters,'\n')

        # print("sigma_parameters", self.sigma_parameters,'\n')
        # print("sigma_parameters", self.sigma_parameters,'\n')
        # print("sigma_max_parameters", self.sigma_max_parameters,'\n')
        # print("sigma_min_parameters", self.sigma_min_parameters,'\n')
        # print("sigma_decay_parameters", self.sigma_decay_parameters,'\n')

        # print("self.max_parameters", self.max_parameters,'\n')
        # print("self.min_parameters", self.min_parameters,'\n')

        # print("mu_neuron_parameters", self.mu_neuron_parameters,'\n')
        # print("mu_neuron_max_parameters", self.mu_neuron_max_parameters,'\n')
        # print("mu_neuron_min_parameters", self.mu_neuron_min_parameters,'\n')

        # print("sigma_neuron_parameters", self.sigma_neuron_parameters,'\n')
        # print("sigma_neuron_parameters", self.sigma_neuron_parameters,'\n')
        # print("sigma_neuron_max_parameters", self.sigma_neuron_max_parameters,'\n')
        # print("sigma_neuron_min_parameters", self.sigma_neuron_min_parameters,'\n')
        # print("sigma_neuron_decay_parameters", self.sigma_neuron_decay_parameters,'\n')

        # print("self.max_neuron_parameters", self.max_neuron_parameters,'\n')
        # print("self.min_neuron_parameters", self.min_neuron_parameters,'\n')

        # print("mu_synapse_parameters", self.mu_synapse_parameters,'\n')
        # print("mu_synapse_max_parameters", self.mu_synapse_max_parameters,'\n')
        # print("mu_synapse_min_parameters", self.mu_synapse_min_parameters,'\n')

        # print("sigma_synapse_parameters", self.sigma_synapse_parameters,'\n')
        # print("sigma_synapse_parameters", self.sigma_synapse_parameters,'\n')
        # print("sigma_synapse_max_parameters", self.sigma_synapse_max_parameters,'\n')
        # print("sigma_synapse_min_parameters", self.sigma_synapse_min_parameters,'\n')
        # print("sigma_synapse_decay_parameters", self.sigma_synapse_decay_parameters,'\n')

        # print("self.max_synapse_parameters", self.max_synapse_parameters,'\n')
        # print("self.min_synapse_parameters", self.min_synapse_parameters,'\n')
        # exit()

    def get_parameters_names(self, config:Dict[str, Any]):
        self.parameters_all_names:List[str] = []
        self.parameters_neuron_names:List[str] = []
        self.parameters_synapse_names:List[str] = []
        
        self.parameters_all_names_original:List[str] = []
        self.parameters_neuron_names_original:List[str] = []
        self.parameters_synapse_names_original:List[str] = []

        for names in config.keys():
            if "_parameter" in names:
                if "_neuron_parameter" in names:
                    # Original
                    self.parameters_neuron_names_original.append(names)
                    self.parameters_all_names_original.append(names)
                    # Replace 
                    self.parameters_neuron_names.append(names.replace("_neuron_parameter", ""))
                    self.parameters_all_names.append(names.replace("_neuron_parameter", ""))

                elif "_synapse_parameter" in names:
                    # Replace
                    self.parameters_synapse_names_original.append(names)
                    self.parameters_all_names_original.append(names)
                    
                    # Original
                    self.parameters_synapse_names.append(names.replace("_synapse_parameter", ""))
                    self.parameters_all_names.append(names.replace("_synapse_parameter", ""))
                else:
                    # Original
                    self.parameters_all_names_original.append(names)

                    # Replace
                    self.parameters_all_names.append(names.replace("_parameter", ""))

    def __set_config_all_parameters(self, config_name_orignal:str, config_name:str):
        config_full_name:str = config_name_orignal
        if self.__config_all.get(config_full_name):
            self.mu_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu"])], dtype=np.float32)
            self.mu_max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu_max"])], dtype=np.float32)
            self.mu_min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu_min"])], dtype=np.float32)

            self.sigma_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma"])], dtype=np.float32)
            self.sigma_max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_max"])], dtype=np.float32)
            self.sigma_min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_min"])], dtype=np.float32)
            self.sigma_decay_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_decay"])], dtype=np.float32)
            
            self.max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["max"])], dtype=np.float32)
            self.min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["min"])], dtype=np.float32)

    def __set_config_neuron_parameters(self, config_name_orignal:str, config_name:str):
        config_full_name:str = config_name_orignal
        if self.__config_all.get(config_full_name):
            self.mu_neuron_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu"])], dtype=np.float32)
            self.mu_neuron_max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu_max"])], dtype=np.float32)
            self.mu_neuron_min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu_min"])], dtype=np.float32)

            self.sigma_neuron_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma"])], dtype=np.float32)
            self.sigma_neuron_max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_max"])], dtype=np.float32)
            self.sigma_neuron_min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_min"])], dtype=np.float32)
            self.sigma_neuron_decay_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_decay"])], dtype=np.float32)
            
            self.max_neuron_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["max"])], dtype=np.float32)
            self.min_neuron_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["min"])], dtype=np.float32)

    def __set_config_synapse_parameters(self, config_name_orignal:str, config_name:str):
        config_full_name:str = config_name_orignal
        if self.__config_all.get(config_full_name):
            self.mu_synapse_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu"])], dtype=np.float32)
            self.mu_synapse_max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu_max"])], dtype=np.float32)
            self.mu_synapse_min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["mu_min"])], dtype=np.float32)

            self.sigma_synapse_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma"])], dtype=np.float32)
            self.sigma_synapse_max_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_max"])], dtype=np.float32)
            self.sigma_synapse_min_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_min"])], dtype=np.float32)
            self.sigma_synapse_decay_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["sigma_decay"])], dtype=np.float32)
            
            self.max_synapse_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["max"])], dtype=np.float32)
            self.min_synapse_parameters[config_name.lower()] = np.array([float(self.__config_all[config_full_name]["min"])], dtype=np.float32)
