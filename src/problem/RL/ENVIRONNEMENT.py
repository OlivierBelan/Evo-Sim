
from typing import List, Dict, Any, Callable, Tuple
from evo_simulator.GENERAL.Genome import Genome_NN
import numpy as np

class Environment:
    def __init__(self, env_name:str, encoding_observation_to_network_input_function:Callable, decoding_output_network_to_action_function:Callable, fitness_step_function:Callable, fitness_end_function:Callable = None, update_observation_min_max_function:Callable = None):
        self.env_name:str = env_name
        self.encoding_observation_to_network_input_function:Callable = encoding_observation_to_network_input_function
        self.decoding_output_network_to_action_function:Callable = decoding_output_network_to_action_function
        self.fitness_step_function:Callable = fitness_step_function
        self.fitness_end_function:Callable = fitness_end_function
        self.update_observation_min_max_function:Callable = update_observation_min_max_function
        self.check_functions()

        self.id:int = None
        self.observation:Any = None
        self.reward:float = None
        self.terminated:bool = False 
        self.truncated: bool = False
        self.extra_info:Dict[str, Any] = {}
        self.seed:int = None

        self.observation_history:List[np.ndarray] = []
        self.observation_max_history:np.ndarray = None
        self.observation_min_history:np.ndarray = None
        self.reward_history:List[float] = []
        self.action_history:List[np.ndarray] = []

    def check_functions(self) -> None:
        if self.encoding_observation_to_network_input_function == None: raise Exception("The encoding_observation_to_network_input_function function is not implemented")
        if self.decoding_output_network_to_action_function == None: raise Exception("The decoding_output_network_to_action_function function is not implemented")
        if self.fitness_step_function == None: raise Exception("The fitness_step_function function is not implemented")
        if self.fitness_end_function == None: raise Exception("The fitness_end_function function is not implemented")

    def update(self, action, genome:Genome_NN, episode:int) -> bool:
        '''
            Update the environment with the action and update the fitness object with the reward
            Return True if the episode is active, else False if the episode is terminated or truncated
        '''
        raise NotImplementedError

    def reset(self, seed:int = None):
        '''
            Reset the environment and set the seed if it is not None
        '''
        raise NotImplementedError

    def encoding_observation_to_network_input(self) -> np.ndarray:
        '''
            Return the spikes encoding of the observation
        '''
        raise NotImplementedError
        
    def fitness_end(self, genomes:Dict[int, Genome_NN], episodes:List[int]):
        '''
            Update the fitness object of each genome with the fitness_end_function
        '''
        if self.fitness_end_function is not None:
            self.fitness_end_function(genomes, episodes)
        else:
            raise NotImplementedError("The fitness_end_function function is not implemented")

    def close(self):
        raise NotImplementedError

    def copy(self) -> "Environment":
        raise NotImplementedError

class Environment_Manager:
    def __init__(self, environment_builer:Callable):
        self.environement_builder:Callable = environment_builer
        self.envs_dict:Dict[int, List[Environment]] = {} # {genome_id: [env1, env2, ...]}
        self.seeds:List[int] = []
        environment_core:Environment = environment_builer.get_env()
        environment_core.reset()
        self.name:str = environment_core.env_name
        self.input_size:int = environment_core.encoding_observation_to_network_input().shape[0]
        self.fitness_end_function:Callable = environment_core.fitness_end_function
        self.update_observation_min_max_function:Callable = environment_core.update_observation_min_max_function
        del environment_core # Free to save a bit of memory (cause the environment_core is not used anymore and pybullet take a lot of memory)
        self.nb_episodes:int = 0
        self.nb_genomes:int = 0
        
    def create_environments(self, genomes_ids:List[int], seeds:List[int]) -> Dict[int, List[Environment]]:

        nb_episode:int = len(seeds)

        if nb_episode == 0 or len(np.unique(seeds)) != len(seeds): raise ValueError("The number of seeds must be equal to the number of episodes and the seeds must be unique")
        self.nb_episodes = nb_episode
        self.nb_genomes = len(genomes_ids)
        self.seeds = seeds

        # 0 - Update the envs_dict with the new current genomes_ids (in order to save memory by not creating new environments)
        self.envs_dict = self.__update_envs_dict_ids_with_genomes_ids(genomes_ids)

        # 1 - Set the environments for each genome
        envs_list:List[Environment] = []
        for id in genomes_ids:
            # 1.2 - Add the genome id to the envs_dict if it does not exist
            if id not in self.envs_dict:
                self.envs_dict[id] = []
            
            # 1.3 - Update the number of environments
            envs_list = self.envs_dict[id]
            while len(envs_list) != nb_episode:

                # 1.3.1 - Build new environments
                if len(envs_list) < nb_episode:
                    envs_list.append(self.create_new_env(id))

                # 1.3.2 - Remove environments
                elif len(envs_list) > nb_episode: 
                    envs_list.pop()
        
        
        # 2 - Reset & Update environment seeds
        self.reset(self.seeds)
        return self.envs_dict


    def encoding_observation_to_snn_input(self) -> List[np.ndarray]:
        inputs_array:np.ndarray = np.zeros((self.nb_episodes, len(self.envs_dict), self.input_size))
        for i in range(self.nb_episodes):
            for j, envs_list in enumerate(self.envs_dict.values()):
                inputs_array[i, j] = envs_list[i].encoding_observation_to_network_input()
        return inputs_array

    def encoding_observation_to_ann_input(self) -> Dict[int, np.ndarray]:
        observations_dict:Dict[int, np.ndarray] = {}
        env_observation:List[np.ndarray] = []

        for env_id, envs_list in self.envs_dict.items():
            env_observation = []
            for j in range(len(envs_list)):
                env_observation.append(envs_list[j].encoding_observation_to_network_input())
            observations_dict[env_id] = np.array(env_observation)
        return observations_dict


    def update_environments(self, genomes:Dict[int, Genome_NN], actions_dict:Dict[int, np.ndarray], episodes:List[int], output_indexes:np.ndarray=None) -> bool:
        is_active:int = 0
        for id, actions in actions_dict.items():
            for i in range(actions.shape[0]):
                if output_indexes is not None: # For SNN only (cause actions dict contain spikes of all neurons, then we need to select only the output neurons spikes)
                    is_active += self.envs_dict[id][i].update(actions[i][output_indexes], genomes[id], episodes[i])
                else:
                    is_active += self.envs_dict[id][i].update(actions[i], genomes[id], episodes[i])
        return is_active > 0 # Return True if at least one environment is active
        
    def fitness_end(self, genomes:Dict[int, Genome_NN], episodes:List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.fitness_end_function is not None:
            return self.fitness_end_function(genomes, episodes)
        else:
            raise NotImplementedError("The fitness_end_function function is not implemented")

    def update_observation_min_max(self, observation_max:np.ndarray, observation_min:np.ndarray) -> None:
        if self.update_observation_min_max_function is not None:
            self.update_observation_min_max_function(observation_max, observation_min)
        else:
            raise NotImplementedError("The update_observation_min_max_function function is not implemented")

    def reset(self, seeds:List[int]) -> None:
        # 1 - Check if the number of seeds is equal to the number of environments
        first_key:int = next(iter(self.envs_dict))
        if len(seeds) != len(self.envs_dict[first_key]) or len(np.unique(seeds)) != len(seeds): raise ValueError("The number of seeds must be equal to the number of environments and the seeds must be unique")

        # 2 - Reset the environments with the seeds
        env:Environment = None
        seed:int = None
        for env_list in self.envs_dict.values():
            for i in range(len(env_list)):
                env:Environment = env_list[i]
                seed:int = seeds[i]
                env.seed = seed
                env.reset(seed)
    
    def __update_envs_dict_ids_with_genomes_ids(self, genomes_ids:List[int]) -> Dict[int, List[Environment]]:
        new_envs_dict:Dict[int, List[Environment]] = {}
        for index, envs_list in enumerate(self.envs_dict.values()):
            if index < len(genomes_ids):
                new_envs_dict[genomes_ids[index]] = envs_list
        return new_envs_dict
    
    def create_new_env(self, id:int) -> Environment:
        new_env:Environment = self.environement_builder.get_env()
        new_env.id = id
        return new_env
    
    def __split_2(self, seq: List, num: int) -> List[List]:
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