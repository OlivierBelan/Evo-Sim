import gymnasium as gym
from problem.RL import QDgym
from typing import List, Callable, Any
from evo_simulator.GENERAL.Fitness import Fitness
from evo_simulator.GENERAL.Genome import Genome_NN
from problem.RL.ENVIRONNEMENT import Environment
# import copy
import numpy as np
import numba as nb


class Environment_Gym(Environment):
    def __init__(self, env_name, gym_env:gym.Env, encoding_observation_to_network_input_function:Callable, decoding_output_network_to_action_function:Callable, fitness_step_function:Callable, fitness_end_function:Callable, update_observation_min_max_function:Callable, seed:int = None, is_Pybullet_Gym:bool = False):
        Environment.__init__(self, env_name, encoding_observation_to_network_input_function, decoding_output_network_to_action_function, fitness_step_function, fitness_end_function, update_observation_min_max_function)
        
        self.gym_env:gym.Env = gym_env
        self.seed = seed
        self.is_Pybullet_Gym = is_Pybullet_Gym

    def update(self, action_encoded:Any, genome:Genome_NN, episode:int) -> bool:
        if self.terminated == True or self.truncated == True: 
            return False

        # 1 - Decode the action and update the environment with the decoded action
        action_decoded = self.decoding_output_network_to_action_function(action_encoded, genome.network_type)
        self.observation, self.reward, self.terminated, self.truncated, self.extra_info = self.gym_env.step(action_decoded)

        # 2 - Update the fitness of the current step
        self.fitness_step_function(
            genome, 
            episode, 
            {
            "observation":self.observation, 
            "reward": self.reward, 
            "action": action_decoded, 
            "terminated": self.terminated, 
            "truncated": self.truncated,
            "extra_info_env": self.extra_info, 
            # "observation_max_history": self.observation_max_history,
            # "observation_min_history": self.observation_min_history,
            "observation_space_high": self.gym_env.observation_space.high,
            "observation_space_low": self.gym_env.observation_space.low,
            # "observation_history": self.observation_history,
            # "reward_history": self.reward_history, 
            # "action_history":self.action_history, 
            })

        # 3 - Save the observation, reward and action in the history variables
        # self.observation_history.append(self.observation)
        # self.reward_history.append(self.reward)
        # self.action_history.append(action_decoded)

        if self.observation_max_history is None:
            self.observation_max_history = self.observation.copy()
            self.observation_min_history = self.observation.copy()
        else:
            self.update_observation_history(self.observation_max_history, self.observation_min_history, self.observation)

        # 4 - get add history information to the fitness object
        fitness_obj:Fitness = genome.fitness
        if fitness_obj.extra_info.get("observation_max_history") is None:
            fitness_obj.extra_info["observation_max_history"] = self.observation_max_history
            fitness_obj.extra_info["observation_min_history"] = self.observation_min_history
            fitness_obj.extra_info["observation_space_high"] = self.gym_env.observation_space.high,
            fitness_obj.extra_info["observation_space_low"] = self.gym_env.observation_space.low
            # fitness_obj.extra_info["observation_history"] = self.observation_history
            # fitness_obj.extra_info["reward_history"] = self.reward_history
            # fitness_obj.extra_info["action_history"] = self.action_history

        return True

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def update_observation_history(observation_max_history:np.ndarray, observation_min_history:np.ndarray, observation:np.ndarray) -> None:
        for i in nb.prange(len(observation)):
            if observation[i] > observation_max_history[i]:
                observation_max_history[i] = observation[i]
            elif observation[i] < observation_min_history[i]:
                observation_min_history[i] = observation[i]

    def reset(self, seed:int = None):
        seed = int(seed) if seed is not None else self.seed
        if self.is_Pybullet_Gym == True:
            self.gym_env.seed(seed)
            self.gym_env.action_space.seed(seed)
            self.gym_env.observation_space.seed(seed)
            self.observation = self.gym_env.reset()
        else:
            if self.seed is not None:
                self.observation, self.extra_info = self.gym_env.reset(seed=int(self.seed))
            else:
                self.observation, self.extra_info = self.gym_env.reset()
        self.reward:float = None
        self.terminated:bool = False
        self.truncated:bool = False
        self.observation_history:List[np.ndarray] = []
        self.reward_history:List[float] = []
        self.action_history:List[np.ndarray] = []

    def encoding_observation_to_network_input(self) -> np.ndarray:
        if self.encoding_observation_to_network_input_function is not None:
            return self.encoding_observation_to_network_input_function(self.observation.copy(), self.gym_env.observation_space.high, self.gym_env.observation_space.low)
        else:
            raise NotImplementedError("The encoding_observation_to_spike function is not implemented")
            
    def close(self):
        self.gym_env.close()
