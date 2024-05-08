
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.Genome import Genome_NN
from problem.RL.ENVIRONNEMENT import Environment
from problem.RL.GYM_ENV import Environment_Gym
from typing import List, Dict, Any, Tuple
from RL_problems_config.Config_Problem import Config_Problem
# import gym
import gymnasium as gym

import numpy as np
import numba as nb

# @nb.njit(cache=True, fastmath=False, nogil=True)
def normalize_number(nb:float, max_value:float, min_value:float) -> float:
    nb = np.clip(nb, min_value, max_value)
    return (float(nb) - float(min_value)) / (float(max_value) - float(min_value))

class QDWalker2D(Config_Problem):
    def __init__(self, name:str, config_path:str, nb_input:int=22, nb_output:int=6, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True, auto_obersvation_ratio:float=0.01) -> None:
        Config_Problem.__init__(self, name, nb_input, nb_output, config_path, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation, auto_obersvation_ratio)

    def get_env(self, render:bool=False) -> Environment_Gym:
        self.gym_env = gym.make("QDWalker2DBulletEnv-v0", render=render)
        return Environment_Gym(
            self.name,
            self.gym_env,
            self.encoding_observation_to_input_network,
            self.decoding_output_network_to_action, 
            self.fitness_step, 
            self.fitness_end,
            self.update_observation_min_max,
            is_Pybullet_Gym=True
        )

    def encoding_observation_to_input_network(self, observation: np.ndarray, observation_max:np.ndarray, observation_min:np.ndarray) -> np.ndarray:
        observation_len:int = len(observation)
        for i in nb.prange(observation_len):
            if self.auto_obersvation == True:
                observation[i] = normalize_number(observation[i], self.observation_max_global[i], self.observation_min_global[i])
            else:
                observation[i] = normalize_number(observation[i], observation_max[i], observation_min[i])
                self.observation_max_global[i] = observation_max[i] # just to for the print in the end (used vs found)
                self.observation_min_global[i] = observation_min[i]
            if observation[i] > 1 or observation[i] < 0:
                raise Exception("observation[i] > 1 or observation[i] < 0")
        return observation

    def transformer_valeur(self, value, min_input=0, max_input=1, min_output=-1, max_output=1):
        """Transforme une valeur proportionnellement d'une plage de départ vers une plage de sortie."""
        return min_output + (value - min_input) / (max_input - min_input) * (max_output - min_output)

    def decoding_output_network_to_action(self, action:np.ndarray, network_type:str) -> np.ndarray:        
        if network_type == "SNN":
            return np.interp(action, [0, 1], [-1, +1])
        else:
            return action # car l'action est déjà entre -1 et 1 (tanh function output)

    @staticmethod
    def fitness_step(genome:Genome_NN, episode:int, info:Dict[str, Any]) -> None:
        if info["terminated"] == True or info["truncated"] == True: return

        # 1 - update the fitness with the reward
        fitness_obj:Fitness = genome.fitness
        if fitness_obj.extra_info.get(episode) == None:
            fitness_obj.extra_info[episode] = info["reward"]
        else:
            fitness_obj.extra_info[episode] += info["reward"]
        genome.info["description"] = info["extra_info_env"]["bc"]
        
    def fitness_end(self, genomes:Dict[int, Genome_NN], episodes:List[int]) -> None:
        # 0 - fitness qui check le success

        # bis 1 - In case the observation is not known
        observation_global_max_history:List[np.ndarray] = []
        observation_global_min_history:List[np.ndarray] = []

        episode_score:List[float] = []

        for genome in genomes.values():
            fitness_obj:Fitness = genome.fitness
            fitness_obj.score:int = 0

            episode_score:List[float] = []
            for episode in episodes:
                fitness_obj.score += fitness_obj.extra_info[episode]
                episode_score.append(fitness_obj.extra_info[episode])
            genome.info["best_episode_raw_score"] = max(episode_score)
            genome.info["mean_episode_raw_score"] = np.mean(episode_score).astype(float)

            # bis 2 - In case the observation is not known            
            observation_global_max_history.append(fitness_obj.extra_info["observation_max_history"])
            observation_global_min_history.append(fitness_obj.extra_info["observation_min_history"])

        # bis 3 - In case the observation is not known
        # print("obs_max use :  ", np.round(self.observation_max_global, 4).tolist(), " obs_min use:   ", np.round(self.observation_min_global, 4).tolist())
        self.observation_max_global = self.observation_max_global.copy()
        self.observation_min_global = self.observation_min_global.copy()
        self.obersvation_stats(np.array(observation_global_max_history), np.array(observation_global_min_history), self.observation_max_global, self.observation_min_global, self.auto_obersvation_ratio)
        # print("obs_max found: ", np.round(self.observation_max_global, 4).tolist(), " obs_min found: ", np.round(self.observation_min_global, 4).tolist())

        return self.observation_max_global, self.observation_min_global


    # @staticmethod
    def obersvation_stats(self, observation_max_history:np.ndarray, observation_min_history:np.ndarray, obs_max:np.ndarray, obs_min:np.ndarray, percent_use:float) -> None:
        observation_use:int = np.ceil(observation_max_history.shape[0] * percent_use).astype(int)
        for i in range(observation_max_history.shape[1]):
            obs_max[i] = np.mean(np.sort(observation_max_history[:, i])[-observation_use:])
            obs_min[i] = np.mean(np.sort(observation_min_history[:, i])[:observation_use])
            if obs_max[i] == obs_min[i]:
                obs_max[i] += 0.1
                obs_min[i] -= 0.1

