import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import argparse

import numpy as np
from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution

from problem.RL.REINFORCEMENT import Reinforcement_Manager

# Algorithms Mono-Objective
from evo_simulator.ALGORITHMS.NEAT.Neat import NEAT
from evo_simulator.ALGORITHMS.GA.GA import GA
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES import CMA_ES
from evo_simulator.ALGORITHMS.NES.NES import NES
from evo_simulator.ALGORITHMS.NES.OpenES import OpenES
from evo_simulator.ALGORITHMS.MAP_ELITE.MAP_ELITE import MAP_ELITE
from evo_simulator.ALGORITHMS.NSLC.NSLC import NSLC
from evo_simulator.ALGORITHMS.HyperNetwork.HyperNetwork import HyperNetwork
import hyper_substrat_config

# JAX Algorithms
from ALGORITHMS.EvoSAX.EvoSax_algo import EvoSax_algo



# Problems
# Discrete
from RL_problems_config.Mountain_car import Mountain_Car
from RL_problems_config.Cart_Pole import Cart_Pole_Config
from RL_problems_config.Arcrobot import Acrobot_Config
from RL_problems_config.Lunar_Lander import Lunar_Lander

# Continuous
from RL_problems_config.Bipedal_Walker import Bipedal_Walker
from RL_problems_config.Mountain_car_Continuous import Mountain_Car_Continuous
from RL_problems_config.Pendulum import Pendulum
from RL_problems_config.Lunar_Lander_Continuous import Lunar_Lander_Continuous


# Robot
from RL_problems_config.HalfCheetah import HalfCheetah
from RL_problems_config.Walker2D import Walker2D
from RL_problems_config.Swimmer import Swimmer
from RL_problems_config.Hopper import Hopper
from RL_problems_config.Ant import Ant

# QD Gym
from RL_problems_config.QDHalfCheetah import QDHalfCheetah
from RL_problems_config.QDAnt import QDAnt
from RL_problems_config.QDHopper import QDHopper
from RL_problems_config.QDWalker2D import QDWalker2D
from RL_problems_config.QDHumanoid import QDHumanoid

from typing import List, Dict, Tuple, Any, Callable
np.set_printoptions(threshold=sys.maxsize)



# Algo Mono-Objective
def ga_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    ga_config_path = os.path.join(local_dir, config_path)
    
    return "GA", GA, ga_config_path

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

def cma_es_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return "CMAES", CMA_ES, config_path, extra_info

def nes_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    # 2 - Algorithms
    return "NES", NES, neat_config_path, extra_info

def openES_func(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    # 2 - Algorithms
    return "OpenES", OpenES, neat_config_path, extra_info


# Algo Multi-Objective
def map_elite(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}

    return "MAPELITE", MAP_ELITE, neat_config_path, extra_info
   
def nslc(config_path) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    extra_info:Dict[str, Any] = {}
    
    return "NSLC", NSLC, config_path, extra_info

# Algo evosax
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
    parser.add_argument('--nb_episodes', type=int, help='Number of episodes', default=1)
    parser.add_argument('--record', type=to_bool, help='Record data', default="False")

    return parser.parse_args()



def get_algorithm(nn:str, algo:str) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 0 - Config path
    if nn.upper() == "SNN":
        start_config_path = "./config/config_snn/RL/"
    elif nn.upper() == "ANN":
        start_config_path = "./config/config_ann/RL/"
    else:
        raise Exception("Neural network:" + nn + " not found")
    
    # 1.0 - Algorithms Mono-Objective
    if algo == "NEAT":     return neat_func(start_config_path + "NEAT_CONFIG_RL.cfg")
    elif algo == "GA":     return ga_func(start_config_path + "GA_CONFIG_RL.cfg")
    elif algo == "CMA_ES": return cma_es_func(start_config_path + "CMA_ES_CONFIG_RL.cfg")
    elif algo == "NES":    return nes_func(start_config_path + "NES_CONFIG_RL.cfg")
    elif algo == "OpenES": return openES_func(start_config_path + "OpenES_CONFIG_RL.cfg")
    
    # 1.2 - Algorithms Multi-Objective
    elif algo == "MAP_ELITE": return map_elite(start_config_path + "MAP_ELITE_CONFIG_RL.cfg")
    elif algo == "NSLC":      return nslc(start_config_path + "NSLC_CONFIG_RL.cfg")
    
    # 1.3 - Algorithms from evoSAX (https://github.com/RobertTLange/evosax)
    elif algo == "DE-evosax":    return evosax_func("DE-evosax", start_config_path + "DE-evosax_CONFIG_RL.cfg")
    elif algo == "ARS-evosax":   return evosax_func("ARS-evosax", start_config_path + "ARS-evosax_CONFIG_RL.cfg")
    elif algo == "NES-evosax":   return evosax_func("NES-evosax", start_config_path + "NES-evosax_CONFIG_RL.cfg")
    elif algo == "PEPG-evosax":  return evosax_func("PEPG-evosax", start_config_path + "PEPG-evosax_CONFIG_RL.cfg")
    elif algo == "OpenES-evosax":return evosax_func("OPENES-evosax", start_config_path + "OpenES-evosax_CONFIG_RL.cfg")
    
    else:
        raise Exception("Algorithm" + algo + " not found")

def get_problem(problem:str, config_path:str) -> Tuple[Neuro_Evolution, str, Dict[str, Any]]:
    # 2 - Environnement
    # 2.1 - Discrete
    if problem == "Mountain_Car":          return Mountain_Car("MountainCar", config_path, nb_input=2, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Cart_Pole":           return Cart_Pole_Config("CartPole", config_path, nb_input=4, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Acrobot":             return Acrobot_Config("Acrobot", config_path, nb_input=6, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
    elif problem == "Lunar_Lander":        return Lunar_Lander("LunarLander", config_path, nb_input=8, nb_output=4, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
    
    # 2.2 - Continuous
    elif problem == "Mountain_Car_Continuous":  return Mountain_Car_Continuous("MountainCarContinous", config_path, nb_input=2, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Pendulum":                 return Pendulum("Pendulum", config_path, nb_input=3, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Lunar_Lander_Continuous":  return Lunar_Lander_Continuous("LunarLanderContinuous", config_path, nb_input=8, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
    elif problem == "Bipedal_Walker":           return Bipedal_Walker("BipedWalker", config_path, nb_input=24, nb_output=4, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False, hardcore=False)
    
    # 2.3 - Continuous Robot
    elif problem == "Swimmer":                 return Swimmer("Swimmer", config_path, nb_input=8, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Hopper":                  return Hopper("Hopper", config_path, nb_input=11, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "HalfCheetah":             return HalfCheetah("HalfCheetah", config_path, nb_input=17, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Walker2D":                return Walker2D("Walker2D", config_path, nb_input=17, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "Ant":                     return Ant("Ant", config_path, nb_input=27, nb_output=8, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)

    # 2.4 - QD Gym
    elif problem == "QDHalfCheetah":           return QDHalfCheetah("QDHalfCheetah", config_path, nb_input=26, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "QDAnt":                   return QDAnt("QDAnt", config_path, nb_input=28, nb_output=8, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "QDHopper":                return QDHopper("QDHopper", config_path, nb_input=15, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "QDWalker":                return QDWalker2D("QDWalker", config_path, nb_input=22, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
    elif problem == "QDHumanoid":              return QDHumanoid("QDHumanoid", config_path, nb_input=44, nb_output=17, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)

    else:
        raise Exception("Problem" + problem + " not found")

def neuro_evo_matrix_func():
    args = parse_arg()

    # 1 - Algorithm
    name, algorithm, config_path, algo_extra_info = get_algorithm(args.nn, args.algo)

    # 2 - Problem
    environnement = get_problem(args.problem, config_path)


    # 3 - Seeds
    seeds = []
    max_seeds:int = 1_000_000
    for _ in range(args.nb_runs): seeds.append(np.random.choice(np.arange(max_seeds), size=args.nb_episodes, replace=False))
    seeds = np.array(seeds)
    print("seeds: ", seeds)

    # 4 - Run
    neuro:Neuro_Evolution = Neuro_Evolution(nb_generations=args.nb_generations, nb_runs=args.nb_runs, is_record=args.record, is_Archive=False, config_path=config_path, cpu=args.cpu)
    neuro.init_algorithm(name, algorithm, config_path, algo_extra_info)


    # If you want to run QD Gym uncomment the following line and comment the following line (neuro.run_rastrigin)
    neuro.init_problem_RL(Reinforcement_Manager, config_path, environnement, nb_episode=args.nb_episodes, seeds=seeds, render=False)
    neuro.run()

    # If you want to run rasstrigin uncomment the following line and comment the following line (neuro.init_problem_RL & neuro.run)
    # neuro.run_rastrigin()

def main():
    neuro_evo_matrix_func()

if __name__ == "__main__":
    main()
