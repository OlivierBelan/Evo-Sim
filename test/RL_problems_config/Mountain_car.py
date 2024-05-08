
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.Genome import Genome_NN
from problem.RL.ENVIRONNEMENT import Environment
from problem.RL.GYM_ENV import Environment_Gym
from typing import List, Dict, Any, Tuple
from RL_problems_config.Config_Problem import Config_Problem
import gymnasium as gym

import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True, nogil=True)
def normalize_number(nb:float, max_value:float, min_value:float) -> float:
    res = (float(nb) - float(min_value)) / (float(max_value) - float(min_value))
    if res < 0.0: return 0.0
    if res > 1.0: return 1.0
    return res
    # return (float(nb) - float(min_value)) / (float(max_value) - float(min_value))

class Mountain_Car(Config_Problem):
    def __init__(self, name:str, config_path:str, nb_input:int=2, nb_output:int=3, obs_max_init_value:float=5, obs_min_init_value:float=-5, termination_finess:float=None, auto_obersvation:bool=True) -> None:
        Config_Problem.__init__(self, name, nb_input, nb_output, config_path, obs_max_init_value, obs_min_init_value, termination_finess, auto_obersvation)

    def get_env(self, render:bool = False) -> Environment_Gym:
        if render == True:
            self.gym_env:gym.Env = gym.make('MountainCar-v0', render_mode="human")
        else:
            self.gym_env:gym.Env = gym.make('MountainCar-v0')

        return Environment_Gym(
            self.name,
            self.gym_env,
            self.encoding_observation_to_spike_Mountain_car,
            self.decoding_action_from_spike_Mountain_car, 
            self.fitness_step_Mountain_car_RAW_200, 
            # fitness_end_Mountain_car_RAW_200
            self.fitness_end_Mountain_car_MANUAL,
            self.update_observation_min_max
        )

    def encoding_observation_to_spike_Mountain_car(self, observation: np.ndarray, observation_max:np.ndarray, observation_min:np.ndarray) -> np.ndarray:
        observation_len:int = len(observation)
        for i in nb.prange(observation_len):
            # if self.auto_obersvation == True:
            #     observation[i] = normalize_number(observation[i], self.observation_max_global[i], self.observation_min_global[i])
            # else:
            observation[i] = normalize_number(observation[i], observation_max[i], observation_min[i])
                # self.observation_max_global[i] = observation_max[i] # just to for the print in the end (used vs found)
                # self.observation_min_global[i] = observation_min[i]
        if np.where(observation < 0.0)[0].shape[0] != 0: 
            print("observation < 0.0", observation)
            exit()
        return observation

    @staticmethod
    def decoding_action_from_spike_Mountain_car(action: np.ndarray, network_type:str) -> int:
        return np.argmax(action)

    @staticmethod
    def fitness_step_Mountain_car_RAW_200(genome:Genome_NN, episode:int, info:Dict[str, Any]) -> None:
        if info["terminated"] == True or info["truncated"] == True: return
        action_left, action_nothing, action_right = 0, 1, 2

        # 1 - update the fitness with the reward
        fitness_obj:Fitness = genome.fitness
        if fitness_obj.extra_info.get(episode) == None:
            fitness_obj.extra_info[episode] = info["reward"]
        else:
            fitness_obj.extra_info[episode] += info["reward"]

        # 2 - fitness qui check la vitesse, la direction et l'action
        if fitness_obj.extra_info.get("fitness_vitesse_direction_action") == None:
            fitness_obj.extra_info["fitness_vitesse_direction_action"] = 0.0
        velocity:float = info["observation"][1]
        action:int = info["action"]
        if ((velocity > 0.0 and action == action_left) or (velocity < 0.0 and action == action_right)): # can be improved with previous action/position but ok for now
            fitness_obj.extra_info["fitness_vitesse_direction_action"] += 1.0
            

        # 3 - fitness qui check s'il n'y a pas d'acceleration
        if fitness_obj.extra_info.get("fitness_no_acceleration") == None:
            fitness_obj.extra_info["fitness_no_acceleration"] = 0.0
        if action == action_nothing:
            fitness_obj.extra_info["fitness_no_acceleration"] += 1.0
            
        
        # 4 - fitness qui check la distance entre la position et la ligne d'arrivée
        if fitness_obj.extra_info.get("fitness_distance") == None:
            fitness_obj.extra_info["fitness_distance"] = 0.0
        position:float = info["observation"][0]
        distance:float = normalize_number(position, 0.5, info["observation_space_low"][0])
        distance:float = distance if distance <= 1.0 else 1.0
        if fitness_obj.extra_info["fitness_distance"] < distance:
            fitness_obj.extra_info["fitness_distance"] = distance
        
    @staticmethod
    def fitness_end_Mountain_car_RAW_200(genomes:Dict[int, Genome_NN], episodes:List[int]):
        best_performances:List[float] = []

        for genome in genomes.values():
            fitness_obj:Fitness = genome.fitness
            fitness_obj.score:int = 0
            for episode in episodes:
                if fitness_obj.extra_info[episode] > -199.0:
                    fitness_obj.score += 1
                    best_performances.append(fitness_obj.extra_info[episode])
                    
        # print("best_performances: ", sorted(best_performances, reverse=True)[:10])

    def fitness_end_Mountain_car_MANUAL(self, genomes:Dict[int, Genome_NN], episodes:List[int]) -> None:
        # 1 - fitness qui check le success
        # 2 - fitness qui check le nombre de pas
        # 3 - fitness qui check la distance entre la position et la ligne d'arrivée
        # 4 - fitness qui check la vitesse, la direction et l'action
        
        # 0 - In case the observation is not known
        observation_global_max_history:List[np.ndarray] = []
        observation_global_min_history:List[np.ndarray] = []

        nb_success:int = 0
        nb_steps:float = 0
        best_performances_overall:List[float] = []
        episode_score:List[float] = []

        for genome in genomes.values():
            fitness_obj:Fitness = genome.fitness
            fitness_obj.score:int = 0

            nb_success:int = 0
            nb_steps:float = 0
            episode_score:List[float] = []
            for episode in episodes:
                if fitness_obj.extra_info[episode] >= -199.0:
                    nb_success += 1
                    best_performances_overall.append(fitness_obj.extra_info[episode])
                nb_steps += fitness_obj.extra_info[episode]
                episode_score.append(fitness_obj.extra_info[episode])
            genome.info["best_episode_raw_score"] = max(episode_score)
            genome.info["mean_episode_raw_score"] = np.mean(episode_score).astype(float)
            nb_steps /= len(episodes)

            # normalize the fitness
            success_fitness:float = nb_success / len(episodes)
            steps_fitness:float = 1 - (nb_steps / -199.0)
            fitness_vitesse_direction_action:float = 1 - (fitness_obj.extra_info["fitness_vitesse_direction_action"] / len(episodes)) / 199.0
            fitness_no_acceleration:float = 1 - (fitness_obj.extra_info["fitness_no_acceleration"] / len(episodes)) / 199.0
            fitness_distance:float = fitness_obj.extra_info["fitness_distance"]
            # Build the total fitness
            total_fitness:float = 5*success_fitness + 2*steps_fitness + fitness_vitesse_direction_action + fitness_no_acceleration + fitness_distance
            fitness_obj.score = total_fitness

            # 1 - In case the observation is not known            
            observation_global_max_history.append(fitness_obj.extra_info["observation_max_history"])
            observation_global_min_history.append(fitness_obj.extra_info["observation_min_history"])

        # 2 - In case the observation is not known
        self.observation_max_global = self.observation_max_global.copy()
        self.observation_min_global = self.observation_min_global.copy()
        self.obersvation_stats_Mountain_car_2(np.array(observation_global_max_history), np.array(observation_global_min_history), self.observation_max_global, self.observation_min_global, 0.01)


        if best_performances_overall == []: return
        best_current = max(best_performances_overall)
        if self.best_ever_score == None:self.best_ever_score = best_current
        if best_current > self.best_ever_score:self.best_ever_score = best_current
        # print("Best Ever Score", self.best_ever_score, "best_performances_overall: ", sorted(best_performances_overall, reverse=True)[:10])
        return self.observation_max_global, self.observation_min_global

    @staticmethod
    def fitness_update_with_reward_Mountain_car_AUTO(genomes:Dict[int, Genome_NN], episodes:List[int]) -> None:
        pass

    # @staticmethod
    def obersvation_stats_Mountain_car_2(self, observation_max_history:np.ndarray, observation_min_history:np.ndarray, obs_max:np.ndarray, obs_min:np.ndarray, percent_use:float) -> None:
        observation_use:int = np.ceil(observation_max_history.shape[0] * percent_use).astype(int)
        # observation_use:int = 1
        for i in range(observation_max_history.shape[1]):
            obs_max[i] = np.mean(np.sort(observation_max_history[:, i])[-observation_use:])
            obs_min[i] = np.mean(np.sort(observation_min_history[:, i])[:observation_use])
        # print("obs_max: ", obs_max, " obs_min: ", obs_min)

    def obersvation_stats_Mountain_car(self, observation_history:np.ndarray, obs_max:np.ndarray, obs_min:np.ndarray, percent_use:float) -> None:
        observation_use:int = np.ceil(observation_history.shape[0] * percent_use).astype(int)
        observation_use:int = 1
        # print("observation_use: ", observation_use)
        for i in range(observation_history.shape[1]):
            # print("np.sort(observation_history[:, i])[-observation_use:]", np.sort(observation_history[:, i])[-observation_use:])
            # print("np.sort(observation_history[:, i])[-1]", np.sort(observation_history[:, i])[-3:])
            # print("np.sort(observation_history[:, i])[0]", np.sort(observation_history[:, i])[:3])
            # exit()
            obs_max[i] = np.mean(np.sort(observation_history[:, i])[-observation_use:])
            obs_min[i] = np.mean(np.sort(observation_history[:, i])[:observation_use])
        # print("obs_max: ", obs_max, " obs_min: ", obs_min)
        # exit()
