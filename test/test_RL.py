import sys
sys.path.append('../')
sys.path.append('../src/')
sys.path.append('../src/snn_simulator/')
sys.path.append('../src/evo_simulator/')
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import numpy as np
from evo_simulator.GENERAL.Neuro_Evolution import Neuro_Evolution

from problem.RL.REINFORCEMENT import Reinforcement_Manager


from evo_simulator.ALGORITHMS.Algorithm import Algorithm

# Algorithms Mono-Objective
from evo_simulator.ALGORITHMS.NEAT.Neat import NEAT
from evo_simulator.ALGORITHMS.GA.GA import GA
from evo_simulator.ALGORITHMS.CMA_ES.CMA_ES import CMA_ES
from evo_simulator.ALGORITHMS.NES.NES import NES
from evo_simulator.ALGORITHMS.NES.OpenES import OpenES
from evo_simulator.ALGORITHMS.MAP_ELITE.MAP_ELITE import MAP_ELITE
from evo_simulator.ALGORITHMS.NSLC.NSLC import NSLC
# from evo_simulator.ALGORITHMS.HyperNEAT.HyperNEAT import HyperNEAT
# import hyper_substrat_config

# JAX Algorithms
from ALGORITHMS.JAX_algo.jax_algo import Jax_algo



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


from typing import List, Dict, Tuple
np.set_printoptions(threshold=sys.maxsize)



# Algo Mono-Objective
def ga_func(config_path) -> Tuple[str, Algorithm, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    ga_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "GA", GA, ga_config_path

def neat_func(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "NEAT", NEAT, neat_config_path

def cma_es_func(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "CMAES", CMA_ES, config_path

def nes_func(name, config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    # 2 - Algorithms
    return name, NES, neat_config_path

def openES_func(name, config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    # 2 - Algorithms
    return name, OpenES, neat_config_path


# Algo Multi-Objective
def map_elite(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    neat_config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "MAPELITE", MAP_ELITE, neat_config_path
   
def nslc(config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, start_config_path + config_path)
    
    return "NSLC", NSLC, config_path

# Algo evosax
def algo_jax_func(name:str, config_path) -> Tuple[Neuro_Evolution, str]:
    # 1 - Config path file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, start_config_path + config_path)

    return name, Jax_algo, config_path

start_config_path = "./config/config_snn/RL/"
# start_config_path = "./config/config_ann/RL/"

def neuro_evo_matrix_func(args:List[str]):
    if len(args) == 0:raise Exception("Error: No arguments")

    aglos_dict:Dict[str, Tuple[Neuro_Evolution, str]] = {
        # 1.0 - Algorithms Mono-Objective
        "NEAT":           neat_func("NEAT_CONFIG_RL.cfg"),
        "GA":             ga_func("GA_CONFIG_RL.cfg"),
        "CMA_ES":         cma_es_func("CMA_ES_CONFIG_RL.cfg"),
        "NES":            nes_func("NES","NES_CONFIG_RL.cfg"),
        "OpenES":         openES_func("OpenES", "OpenES_CONFIG_RL.cfg"),
        # "HyperNEAT":      hyperneat_func("HyperNEAT_CONFIG_RL.cfg"), # "HyperNEAT_CONFIG_RL.cfg

        # 1.2 - Algorithms Multi-Objective
        "MAP_ELITE":      map_elite("MAP_ELITE_CONFIG_RL.cfg"),
        "NSLC":           nslc("NSLC_CONFIG_RL.cfg"),

        # 1.3 - Algorithms from evoSAX (https://github.com/RobertTLange/evosax)
        "DE-evosax":      algo_jax_func("DE-evosax","DE-evosax_CONFIG_RL.cfg"),
        "ARS-evosax":     algo_jax_func("ARS-evosax", "ARS-evosax_CONFIG_RL.cfg"),
        "NES-evosax":     algo_jax_func("NES-evosax", "NES-evosax_CONFIG_RL.cfg"),
        "PEPG-evosax":    algo_jax_func("PEPG-evosax", "PEPG-evosax_CONFIG_RL.cfg"),
        "OpenES-evosax":  algo_jax_func("OPENES-evosax", "OPENES-evosax_CONFIG_RL.cfg"),

        
    }
    algos = [aglos_dict[arg] for arg in args if arg in aglos_dict]
    nb_runs:int = 3
    nb_episode:int = 1
    nb_generation:int = 100
    max_seeds:int = 100_000
    seeds = []
    for _ in range(nb_runs): seeds.append(np.random.choice(np.arange(max_seeds), size=nb_episode, replace=False))
    seeds = np.array(seeds)
    print("seeds: ", seeds)

    for name, algorithm, config_path in algos:
        # 2 - Environnement

        # 2.1 - Discrete
        # environnement:Mountain_Car = Mountain_Car("MountainCar", config_path, nb_input=2, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Cart_Pole_Config = Cart_Pole_Config("CartPole", config_path, nb_input=4, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Acrobot_Config = Acrobot_Config("Acrobot", config_path, nb_input=6, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
        # environnement:Lunar_Lander = Lunar_Lander("LunarLander", config_path, nb_input=8, nb_output=4, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)

        # 2.2 - Continuous
        environnement:Mountain_Car_Continuous = Mountain_Car_Continuous("MountainCarContinous", config_path, nb_input=2, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Pendulum = Pendulum("Pendulum", config_path, nb_input=3, nb_output=1, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Lunar_Lander = Lunar_Lander_Continuous("LunarLanderContinuous", config_path, nb_input=8, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False)
        # environnement:Bipedal_Walker = Bipedal_Walker("BipedWalker", config_path, nb_input=24, nb_output=4, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=False, hardcore=False)


        # 2.3 - Continuous Robot
        # environnement:Swimmer = Swimmer("Swimmer", config_path, nb_input=8, nb_output=2, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Hopper = Hopper("Hopper", config_path, nb_input=11, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:HalfCheetah = HalfCheetah("HalfCheetah", config_path, nb_input=17, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Walker2D = Walker2D("Walker2D", config_path, nb_input=17, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:Ant = Ant("Ant", config_path, nb_input=27, nb_output=8, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)

        # 2.4 - QD Gym
        # environnement:QDHalfCheetah = QDHalfCheetah("QDHalfCheetah", config_path, nb_input=26, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDAnt = QDAnt("QDAnt", config_path, nb_input=28, nb_output=8, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDHopper = QDHopper("QDHopper", config_path, nb_input=15, nb_output=3, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDWalker2D = QDWalker2D("QDWalker", config_path, nb_input=22, nb_output=6, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)
        # environnement:QDHumanoid = QDHumanoid("QDHumanoid", config_path, nb_input=44, nb_output=17, obs_max_init_value=5, obs_min_init_value=-5, auto_obersvation=True)

        # 3 - Reinforcement Manager -> Run
        neuro:Neuro_Evolution = Neuro_Evolution(nb_generations=nb_generation, nb_runs=nb_runs, is_record=True, is_Archive=False, config_path=config_path, cpu=1)
        neuro.init_algorithm(name, algorithm, config_path)
        neuro.init_problem_RL(Reinforcement_Manager, config_path, environnement, nb_episode=nb_episode, seeds=seeds, render=False)
        neuro.run()

def main():
    neuro_evo_matrix_func(sys.argv[1:])

if __name__ == "__main__":
    main()
