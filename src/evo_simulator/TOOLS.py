from typing import List, Union, Any, ValuesView, Generator, Dict, Tuple
import numpy as np
import random
from configparser import ConfigParser
import numba as nb

def random_fast_choice(options:List, probs:List) -> Any:
    if (len(options) != len(probs)):
        raise Exception("random_fast_choice: options and probs must have the same length")
    x:float = random.random()
    cum:float = 0.0
    for i, p in enumerate(probs):            
        cum += p            
        if x < cum:
            return options[i]
    return options[-1]

def random_fast_choice_view(options:ValuesView, probs:Generator) -> Any:
    x:float = random.random()
    cum:float = 0.0
    for o, p in zip(options, probs):
        cum += p            
        if x < cum:
            return o
    return o

def normalize_array(array:Union[np.ndarray, float], max:float, min:float) -> Union[np.ndarray, float]:
    if isinstance(array, list):
        array = np.array(array)
    if np.all((array == min)) and (max - min) == 0 or (max - min) == 0:
        return (array - min) + 1
    return (array - min) / (max - min)

def config_function(config_path_file:str, items:List[str]) -> Dict[str, Dict[str, Any]]:
    config_dict:Dict[str, Dict[str, Any]] = {}
    config_parser = ConfigParser()
    with open(config_path_file) as f:
        config_parser.read_file(f)
    
    for section in items:
        if config_parser.has_section(section) == False:
            raise Exception(f"{section} section not found in config file")
        config_dict[section] = dict(config_parser.items(section))
    
    return config_dict

def is_config_section(config_path_file:str, section:str) -> bool:
    config_parser = ConfigParser()
    with open(config_path_file) as f:
        config_parser.read_file(f)
    return config_parser.has_section(section)

def config_function_all(config_path_file:str) -> Dict[str, Dict[str, Any]]:
    config_dict:Dict[str, Dict[str, Any]] = {}
    config_parser = ConfigParser()
    with open(config_path_file) as f:
        config_parser.read_file(f)
    
    for section in config_parser.keys():
        config_dict[section] = dict(config_parser.items(section))
    
    return config_dict

@staticmethod
@nb.njit(cache=True, fastmath=True, nogil=True)
def cosine_distance_array_jit(array_1:np.ndarray, array_2:np.ndarray) -> float:
    return 1 - np.dot(array_1, array_2) / (np.linalg.norm(array_1) * np.linalg.norm(array_2))

@staticmethod
@nb.njit(cache=True, fastmath=True, nogil=True)
def euclidean_distance_array_jit(array_1:np.ndarray, array_2:np.ndarray) -> float:
    return np.linalg.norm(array_1-array_2)


def split(seq:list, num: int) -> List[List]:
    if num <= 0:
        raise ValueError("Number of chunks should be greater than 0")

    k, m = divmod(len(seq), num)
    out = []

    i = 0
    for _ in range(num):
        next_i = i + k + (1 if _ < m else 0)  # This distributes the extra m elements to the first m chunks
        out.append(seq[i:next_i])
        i = next_i

    return out

def hiddens_nb_from_config(hiddens:str) -> int:
    return sum(list(map(int, hiddens.split("x"))))
def hiddens_from_config(hiddens:str) -> List[int]:
    return list(map(int, hiddens.split("x")))

def architecture_from_config(architecture:str, nb_layers:int) -> Tuple[List[List[str]], List[str]]:
    string_list = architecture.split(", ")
    archictecture_config = []
    is_inputs = False
    is_outputs = False
    hiddens_layer_names:List[str] = []

    for item in string_list:
        connection = item.split("->")
        if len(connection) != 2:
            raise Exception("architecture_from_config: each connection must be a tuple of two elements")
        if connection[0] == "I":
            is_inputs = True
        if connection[1] == "O":
            is_outputs = True
        if connection[0] not in ["I", "O"] and connection[0] not in hiddens_layer_names:
            hiddens_layer_names.append(connection[0])
        if connection[1] not in ["I", "O"] and connection[1] not in hiddens_layer_names:
            hiddens_layer_names.append(connection[1])
        if connection not in archictecture_config:
            archictecture_config.append(connection)

    if is_inputs == False or is_outputs == False: raise Exception("architecture_from_config: architecture must contain at least one input ( I ) and one output (O)")
    if len(hiddens_layer_names) != nb_layers: raise Exception("archi must contain the same number of hidden layers in hiddens config -> architecture_hidden : architecture ( " +str(len(hiddens_layer_names))+" ) != hidden_config (layers) nb ( "+str(nb_layers)+" )" )
    return archictecture_config, hiddens_layer_names


@staticmethod
@nb.njit(cache=True, fastmath=True, nogil=True)
def epsilon_mu_sigma_jit(parameter:np.ndarray, mu_parameter:np.ndarray, sigma_paramater:np.ndarray, min:np.ndarray, max:np.ndarray, mu_bias:float=0, sigma_coef:float=1.0) -> np.ndarray:
    '''
    Jit function for the epsilon computation
    '''
    # 1- set Epislon with the Gaussian (from randn) distribution (Mu -> center of the distribution, Sigma -> width of the distribution)
    mu:np.ndarray = mu_parameter + mu_bias
    sigma:np.ndarray = sigma_paramater * sigma_coef
    epsilon:np.ndarray = np.random.randn(1, parameter.size) * sigma + mu
    # 2- clip Epsilon and apply it to the neurons parameters
    return np.clip(epsilon.astype(np.float32)[0], min, max)
