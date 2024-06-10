import os
import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
os.environ["RAY_DEDUP_LOGS"] = "0"
from typing import List, Tuple, Dict, Any, Callable
import argparse
import random


from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution
from problem.SL.SUPERVISED import Supervised_Manager

# Algorithms
from evo_simulator.ALGORITHMS.NEAT.Neat import NEAT
from evo_simulator.ALGORITHMS.GA.GA import GA
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES import CMA_ES
from evo_simulator.ALGORITHMS.NES.NES import NES
from evo_simulator.ALGORITHMS.NES.OpenES import OpenES
from evo_simulator.ALGORITHMS.MAP_ELITE.MAP_ELITE import MAP_ELITE
from evo_simulator.ALGORITHMS.NSLC.NSLC import NSLC
from evo_simulator.ALGORITHMS.HyperNetwork.HyperNetwork import HyperNetwork
from evo_simulator.ALGORITHMS.ES_HyperNetwork.ES_HyperNetwork import ES_HyperNetwork

from evo_simulator.ALGORITHMS.EvoSAX.EvoSax_algo import EvoSax_algo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import hyper_substrat_config

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def get_mnist_data_set(sample_size:int=60_000):
    batch_size = sample_size
    data_path='./data_set/mnist'

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    # mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader_mnist = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader_mnist = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    features, labels = next(iter(train_loader_mnist))
    return "MNIST", features.flatten(1).numpy().astype(np.float64), labels.numpy().astype(np.float64)


def min_max_normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def z_score_standardize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev

def wine_data_set():
    # 1 - Load data
    data = np.genfromtxt("./data_set/wine.csv", delimiter=',', skip_header=False)
    labels = data[:, 0]
    features = data[:, 1:]

    # 2 - Normalize data
    features = min_max_normalize(features)

    # 3 - Shuffle data
    data_list = list(zip(labels, features))
    random.shuffle(data_list)
    labels, features = zip(*data_list)

    # 4 - Convert to numpy array
    labels = np.array(labels, dtype=np.float32) - 1
    features = np.array(features, dtype=np.float32)

    return "WINE", features, labels

def breast_cancer_data_set():
    # 1 - Load data
    data = np.genfromtxt("./data_set/wdbc.data", delimiter=',', dtype=object)

    # 2 - Replace labels by numbers
    labels = data[:, 1]
    labels[labels == b'M'] = 1
    labels[labels == b'B'] = 2
    data[:, 1] = labels
    data = data.astype(np.float32)

    # 3 - Separate labels and features
    labels = data[:, 1]
    features = data[:, 2:]

    # 4 - Normalize data
    features = min_max_normalize(features)

    # 5 - Shuffle data
    data_list = list(zip(labels, features))
    random.shuffle(data_list)
    labels, features = zip(*data_list)

    # 6 - Convert to numpy array
    labels = np.array(labels, dtype=np.float32) - 1
    features = np.array(features, dtype=np.float32)

    return "BREAST_CANCER", features, labels

def xor_data_set():
    features = np.array([[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]], dtype=np.float64)
    label = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    # data_list = list(zip(output_expected, data_set))
    # random.shuffle(data_list)
    return "XOR", features, label


# Algo Mono-Objective
def ga_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return "GA", GA, neat_config_path, extra_info

def neat_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return "NEAT", NEAT, neat_config_path, extra_info

def hypernetwork_func(config_path, cppn_builder:Callable, cpu:int) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    hypernetwork_config_path = os.path.join(local_dir, config_path)

    extra_info:Dict[str, Any] = {
        "cppn_builder": cppn_builder,
        "substrat_function": hyper_substrat_config.generate_multi_layer_circle_points, # circle substrat param
        # "substrat_function": hyper_substrat_config.generate_vertical_line_points, # vertical substrat param
        "cpu": cpu,
        }

    # 2 - Algorithms
    return "HyperNetwork", HyperNetwork, hypernetwork_config_path, extra_info

def es_hypernetwork_func(config_path, cppn_builder:Callable, cpu:int) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    es_hypernetwork_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {
        "cppn_builder": cppn_builder,
        "substrat_function": hyper_substrat_config.generate_multi_layer_circle_points, # circle substrat param
        # "substrat_function": hyper_substrat_config.generate_vertical_line_points, # vertical substrat param
        "cpu": cpu,
        }

    # 2 - Algorithms
    return "ES_HyperNetwork", ES_HyperNetwork, es_hypernetwork_config_path, extra_info

def cma_es_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return "CMA_ES", CMA_ES, neat_config_path, extra_info

def nes_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return "NES", NES, neat_config_path, extra_info

def openES_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    # 2 - Algorithms
    return "OpenES", OpenES, neat_config_path, extra_info



def evosax_func(name:str, config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return name, EvoSax_algo, config_path, extra_info



def parse_arg():
    def to_bool(s) -> bool:
        if s == "True":
            return True
        else:
            return False
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cpu', type=int, help='Number of cpu', default=1)
    parser.add_argument('--nn', type=str, help='Type of neural network', default="SNN")
    parser.add_argument('--algo', type=str, help='Algorithm name', default="NES-evosax")
    parser.add_argument('--problem', type=str, help='Problem name')
    parser.add_argument('--nb_runs', type=int, help='Number of runs', default=3)
    parser.add_argument('--nb_generations', type=int, help='Number of generations', default=50)
    parser.add_argument('--record', type=to_bool, help='Record data', default="False")

    return parser.parse_args()


def get_algorithm(nn:str, algo:str, cpu:int) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 0 - Config path
    if nn.upper() == "SNN":
        start_config_path = "./config/config_snn/SL/"
    elif nn.upper() == "ANN":
        start_config_path = "./config/config_ann/SL/"
    else:
        raise Exception("Neural network:" + nn + " not found")
    
    # 1.0 - Algorithms
    if   algo == "NEAT":   return neat_func(start_config_path + "NEAT_CONFIG_SL.cfg")
    elif algo == "GA":     return ga_func(start_config_path + "GA_CONFIG_SL.cfg")
    elif algo == "CMA_ES": return cma_es_func(start_config_path + "CMA_ES_CONFIG_SL.cfg")
    elif algo == "NES":    return nes_func(start_config_path + "NES_CONFIG_SL.cfg")
    elif algo == "OpenES": return openES_func(start_config_path + "OpenES_CONFIG_SL.cfg")

    elif algo == "HyperNetwork": return hypernetwork_func(start_config_path + "HyperNetwork_CONFIG_SL.cfg", neat_func( start_config_path + "NEAT_CONFIG_SL.cfg"), cpu)
    # elif algo == "HyperNetwork": return hypernetwork_func(start_config_path + "HyperNetwork_CONFIG_SL.cfg", neat_func( "./config/config_ann/SL/" + "NEAT_CONFIG_SL.cfg"), cpu)
    # elif algo == "HyperNetwork": return hypernetwork_func(start_config_path + "HyperNetwork_CONFIG_SL.cfg", evosax_func("NES-evosax",    start_config_path + "NES-evosax_CONFIG_SL.cfg"), cpu)
    # elif algo == "HyperNetwork": return hypernetwork_func(start_config_path + "HyperNetwork_CONFIG_SL.cfg", evosax_func("NES-evosax",   "./config/config_ann/SL/" + "NES-evosax_CONFIG_SL.cfg"), cpu)

    # if algo == "ES_HyperNetwork": return es_hypernetwork_func(start_config_path + "ES_HyperNetwork_CONFIG_SL.cfg", neat_func( start_config_path + "NEAT_CONFIG_SL.cfg"), cpu) # Coming soon

    elif algo == "DE-evosax":     return evosax_func("DE-evosax",     start_config_path + "DE-evosax_CONFIG_SL.cfg")
    elif algo == "ARS-evosax":    return evosax_func("ARS-evosax",    start_config_path + "ARS-evosax_CONFIG_SL.cfg")
    elif algo == "NES-evosax":    return evosax_func("NES-evosax",    start_config_path + "NES-evosax_CONFIG_SL.cfg")
    elif algo == "PEPG-evosax":   return evosax_func("PEPG-evosax",   start_config_path + "PEPG-evosax_CONFIG_SL.cfg")
    elif algo == "OpenES-evosax": return evosax_func("OpenES-evosax", start_config_path + "OpenES-evosax_CONFIG_SL.cfg")

    else:
        raise Exception("Algorithm:" + algo + " not found")
    

def get_data_set(problem:str) -> Tuple[str, np.ndarray, np.ndarray]:
    if problem == "WINE":            return wine_data_set()           # input size = 13,  output size = 3
    elif problem == "BREAST_CANCER": return breast_cancer_data_set()  # input size = 30,  output size = 2
    elif problem == "XOR":           return xor_data_set()            # input size = 2,   output size = 2
    elif problem == "MNIST":         return get_mnist_data_set()      # input size = 784, output size = 10
    else:
        raise Exception("Problem:" + problem + " not found")

def neuro_evo_func():
    args = parse_arg()

    # 1 - Get Algorithm
    name, algorithm, config_path, algo_extra_info = get_algorithm(args.nn, args.algo, args.cpu)

    # 2 - Get Data Set
    problem_name, features, labels = get_data_set(args.problem)
    print("\nLEN DATA SET = ", problem_name, "size", len(features), "LEN input = ", len(features[0]), "labels= ", np.unique(labels), "\n")

    # 3 - Run
    neuro_evo:Neuro_Evolution = Neuro_Evolution(nb_generations=args.nb_generations, nb_runs=args.nb_runs, is_record=args.record, config_path=config_path, cpu=args.cpu)
    neuro_evo.init_algorithm(name, algorithm, config_path, algo_extra_info)
    neuro_evo.init_problem_SL(Supervised_Manager, config_path, problem_name, features, labels)
    neuro_evo.run()



def main():
    parse_arg()
    neuro_evo_func()

if __name__ == "__main__":
    main()
