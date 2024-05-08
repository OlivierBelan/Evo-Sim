import sys
sys.setrecursionlimit(100000)
from snn_simulator.runner_api_cython import Runner as SNN_Runner
from snn_simulator.runner_api_cython import Runner_Info
from ann_simulator.runner import ANN_Runner
from ann_simulator.runner import NN_Custom_torch

from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator import TOOLS


from problem.Problem import Problem
from problem.RL.ENVIRONNEMENT import Environment_Manager, Environment
from snn_simulator.snn_decoder import Decoder


import time
from typing import List, Dict, Any, Callable
import numpy as np
import sys
import psutil

class Reinforcement_Manager(Runner_Info, Problem):
    def __init__(self, environment_builder:Callable, config_path:str, seeds:np.ndarray=None, nb_episode:int=None, cpu_id:int=None, cpu_affinity:bool=False):

        if cpu_id is not None and sys.platform == "linux" and cpu_affinity == True:
            nb_max_cpu = psutil.cpu_count(logical=True)
            id_cpu = nb_max_cpu - cpu_id - 1
            print("CPU ID:", id_cpu)
            psutil.Process().cpu_affinity([id_cpu])        

        self.reset(environment_builder, config_path, seeds, nb_episode)

    def reset(self, environment_builder:Callable, config_path:str, seeds:np.ndarray=None, nb_episode:int=None) -> None:
        Runner_Info.__init__(self, config_path, nb_episode=nb_episode)
        Problem.__init__(self)
        # Public variables
        self.config_path:str = config_path
        self.seeds_from_param:np.ndarray= seeds        

        # Private variables
        self.environment_builder:Callable = environment_builder
        self.environment_manager:Environment_Manager = Environment_Manager(self.environment_builder) # Will be Initialized in the run_generation function

        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN", "NEURO_EVOLUTION"])
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.nb_original_outputs:int = int(self.config["Genome_NN"]["outputs"])
        self.input_multiplicator:int = int(self.config["Genome_NN"]["inputs_multiplicator"])
        self.is_bias:bool = TOOLS.is_config_section(config_path, "bias_neuron_parameter")
        self.__init_runner()        
    

    def __init_runner(self) -> None:
        # if self.runner != None: return
        if self.network_type == "SNN":
            self.runner = SNN_Runner(self.config_path, self.nb_episode) # Initialize the Runner
            self.snn_decoder = Decoder(self.config_path)
        elif self.network_type == "ANN":
            self.runner = ANN_Runner(self.config_path)
        else:
            raise Exception("The network type is not recognized", self.network_type, "The network type must be SNN or ANN")

    def run(self, population:Population, run_nb:int, generation:int, seeds:np.ndarray = None, indexes:List[int] = None, obs_max:np.ndarray = None, obs_min:np.ndarray = None) -> Population:  
        # 0 - Set population received and update the observation_max and observation_min (if needed)
        if indexes is not None:
            genomes_dict:Dict[int, Genome_NN] = population.population
            genomes:Dict[int, Genome_NN] = {genome_id:genomes_dict[genome_id] for genome_id in indexes}
            population.population = genomes
        else:
            genomes:Dict[int, Genome_NN] = population.population
        if obs_max is not None: self.environment_manager.update_observation_min_max(obs_max, obs_min)
        if seeds is not None: self.seeds_from_param = seeds

 
        # 1 - Check if the number of inputs of the genomes is equal to the number of inputs of the environment
        first_key:int = next(iter(genomes))
        if genomes[first_key].inputs != self.environment_manager.input_size:
            raise Exception(str("The number of inputs of the genome (" + str(genomes[first_key].inputs) + ") is different from the number of inputs of the environment (" + str(self.environment_manager.input_size)+ ")"))

        # 2 - RUN NNs
        if self.network_type == "SNN": self.__run_snns(genomes)
        elif self.network_type == "ANN": self.__run_anns(genomes)
        return population, self.obs_max, self.obs_min


    def __run_snns(self, genomes:Dict[int, Genome_NN]) -> Dict[int, Genome_NN]:
        seeds_batch:List[int] = None
        episodes:List[int] = None
        self.seeds_list:List[int] = self.__get_seeds()
        self.__reset_genomes_fitness(genomes)
        nns_cython_dict = self.runner.init_networks(genomes.values())
        envs_dict:Dict[int, List[Environment]] = self.environment_manager.create_environments(genomes_ids=list(genomes.keys()), seeds=self.seeds_list[:self.nb_episode].tolist())
        # print("Neuron_actives_total:", self.runner.neurons_total_used, "; Synapses_actives_total:", self.runner.synapses_total_used)
        is_actives:bool = True
        first_key:int = next(iter(genomes))
        if "input" not in self.record_layer: output_indexes_record:np.ndarray = genomes[first_key].nn.outputs["neurons_indexes"] - genomes[first_key].nn.nb_inputs
        else: output_indexes_record:np.ndarray = genomes[first_key].nn.outputs["neurons_indexes"]
        output_indexes:np.ndarray = genomes[first_key].nn.outputs["neurons_indexes"]
        # print("output_indexes", output_indexes)
        # print("output_indexes_record", output_indexes_record)
        # exit()
        for i in range(0, self.nb_episode, self.nb_episode):
            seeds_batch = self.seeds_list[i:i+self.nb_episode].tolist()
            episodes:List[int] = list(range(i, i+self.nb_episode))

            # 0 - Init RL episode (create/reset environments and inits nns)
            self.environment_manager.reset(seeds_batch)

            while is_actives == True:
                    
                # 1 - (RL Observation) Init inputs networks &  Check if genome_inputs == observation_inputs
                observation_spikes:np.ndarray = self.environment_manager.encoding_observation_to_snn_input()

                # 1.1 - Shape the observation_spikes to match the input_multiplicator
                observation_spikes = np.repeat(observation_spikes, self.input_multiplicator, axis=2)

                # 2 - (RL Action) Run the networks (cython) & get the actions
                actions_dict = self.runner.run(observation_spikes)[self.record_type]

                # 3 - Decoding Spikes
                actions_dict = self.decoding_spikes(actions_dict, output_indexes, output_indexes_record, genomes)

                # 3 - (RL Update/Step) Update the environments with the actions
                is_actives = self.environment_manager.update_environments(genomes, actions_dict, episodes)
        
        # # 5 - Set fitnesses of genomes after the end of evaluation
        # print("spike max", np.max(self.snn_decoder.output_spike_average), 
        #       "spike min", np.min(self.snn_decoder.output_spike_average), 
        #       "spike mean", round(np.mean(self.snn_decoder.output_spike_average),3))
        # print("spike max summed", np.max(self.snn_decoder.output_spike_average_summed),
        #         "spike min summed", np.min(self.snn_decoder.output_spike_average_summed),
        #         "spike mean summed", round(np.mean(self.snn_decoder.output_spike_average_summed),3))
        # self.snn_decoder.output_spike_average = []
        # self.snn_decoder.output_spike_average_summed = []
        self.obs_max, self.obs_min = self.environment_manager.fitness_end(genomes, list(range(0, self.nb_episode)))

    def __run_anns(self, genomes:Dict[int, Genome_NN]) -> Dict[int, Genome_NN]:
        seeds_batch:List[int] = None
        # episodes:List[int] = None
        self.seeds_list:List[int] = self.__get_seeds()
        self.__reset_genomes_fitness(genomes)
        # envs_dict:Dict[int, List[Environment]] = self.environment_manager.create_environments(genomes_ids=list(genomes.keys()), seeds=self.seeds_list[:self.nb_episode].tolist())
        # is_actives:bool = True

        # # 0 - Set net_torch
        # self.runner.set_net_torch_population(genomes, is_bias=self.is_bias)

        # for i in range(0, self.nb_episode, self.nb_episode):
            # seeds_batch = self.seeds_list[i:i+self.nb_episode].tolist()
            # episodes:List[int] = list(range(i, i+self.nb_episode))
        for i in range(self.nb_episode):

            # 0 - Set net_torch
            self.runner.set_net_torch_population(genomes, is_bias=self.is_bias)

            is_actives:bool = True
            envs_dict:Dict[int, List[Environment]] = self.environment_manager.create_environments(genomes_ids=list(genomes.keys()), seeds=[self.seeds_list[i]])
            seeds_batch = [self.seeds_list[i]]

            # 0 - Init RL episode (create/reset environments and inits nns)
            self.environment_manager.reset(seeds_batch)

            while is_actives == True:
                    
                # 1 - (RL Observation) Init inputs networks &  Check if genome_inputs == observation_inputs
                observation_dict:Dict[int, np.ndarray] = self.environment_manager.encoding_observation_to_ann_input()

                # 2 - (RL Action) Run the networks (cython) & get the actions
                actions_dict:Dict[int, np.ndarray] = self.runner.run_RL(genomes, observation_dict)

                # 3 - (RL Update/Step) Update the environments with the actions
                is_actives = self.environment_manager.update_environments(genomes, actions_dict, seeds_batch)

            # 4 - Unset net_torch
            self.runner.unset_net_torch_population(genomes)


        # # 6 - Unset net_torch
        # self.runner.unset_net_torch_population(genomes)

        # 5 - Set fitnesses of genomes after the end of evaluation
        # self.obs_max, self.obs_min = self.environment_manager.fitness_end(genomes, list(range(0, self.nb_episode)))
        self.obs_max, self.obs_min = self.environment_manager.fitness_end(genomes, self.seeds_list.tolist())



    def decoding_spikes(self, actions_dict:Dict[int, np.ndarray], output_indexes_nn:np.ndarray, output_indexes_record:np.ndarray, genomes:Dict[int, Genome_NN]) -> Dict[int, np.ndarray]:
        if self.decoder == "voltage" and self.disable_output_threshold == True:
            voltage_min:np.ndarray = np.full(len(output_indexes_nn), self.voltage_min, dtype=np.float32)
            voltage_max:np.ndarray = np.full(len(output_indexes_nn), self.voltage_max, dtype=np.float32)

        for id, actions in actions_dict.items():
            # actions_dict[id] = np.array([self.snn_decoder.rate(action[output_indexes_record], self.nb_original_outputs) for action in actions], dtype=np.float32)

            if self.decoder == "max_spikes": 
                actions_dict[id] = np.array([self.snn_decoder.max_spikes(action[output_indexes_record], self.nb_original_outputs) for action in actions], dtype=np.float32)
            
            elif self.decoder == "augmented":
                # if self.output_multiplier > 1:
                #     actions_dict[id] = np.array([self.snn_decoder.augmented(action[output_indexes_record], self.nb_original_outputs, self.spike_max) for action in actions], dtype=np.float32)
                # else:
                actions_dict[id] = np.clip(actions[:, output_indexes_record]/self.spike_max, 0, 1) # does not work if output_multiplicator > 1

            elif self.decoder == "rate":
                actions_dict[id] = np.array([self.snn_decoder.rate(action[output_indexes_record], self.nb_original_outputs, self.ratio_max_output_spike) for action in actions], dtype=np.float32)
            
            elif self.decoder == "voltage":
                voltage_min:np.ndarray = genomes[id].nn.parameters["voltage"][output_indexes_nn] if self.is_voltages_min_decoder == True else self.voltage_min
                voltage_max:np.ndarray = genomes[id].nn.parameters["threshold"][output_indexes_nn] if self.is_threshold_max_decoder == True else self.voltage_max
                actions_dict[id] = self.snn_decoder.voltage(actions[:, output_indexes_record], voltage_min, voltage_max)
            
            elif self.decoder == "coeff":
                actions_dict[id] = np.array([self.snn_decoder.coefficient(action[output_indexes_record], genomes[id].nn.parameters["coeff"][output_indexes_nn], self.nb_original_outputs) for action in actions], dtype=np.float32)

            # actions_dict[id] = np.array([self.snn_decoder.voltage(action[output_indexes], reset, threshold) for action in actions], dtype=np.float32)
            # actions_dict[id] = np.array([self.snn_decoder.coefficient(action[output_indexes], genomes[id].nn.parameters["coeff"][output_indexes_nn], self.nb_original_outputs) for action in actions], dtype=np.float32)
        return actions_dict

    def __reset_genomes_fitness(self, genomes:Dict[int, Genome_NN]) -> None:
        for genome in genomes.values():
            genome.fitness.reset()

    def __get_seeds(self) -> np.ndarray:
        if self.seeds_from_param is not None:
            # if len(self.seeds_from_param) != self.nb_episode: raise Exception("The number of seeds given in the parameter is different from the number of episodes ->", self.nb_episode, "vs", len(self.seeds_from_param))
            return self.seeds_from_param
        else:
            return np.random.choice(np.arange(1e6), size=self.nb_episode, replace=False)
            # return self.__get_random_seeds(self.nb_episode, 10000)

    def __get_random_seeds(self, nb_seeds:int, max_seeds:int) -> np.ndarray:
        return np.random.choice(np.arange(max_seeds), size=nb_seeds, replace=False)
    
    def run_render(self, genome:Genome_NN, env:Environment, seed:int, obs_max:np.ndarray=None, obs_min:np.ndarray=None) -> None:
        if self.network_type == "SNN": self.run_render_SNN(genome, env, seed, obs_max, obs_min)
        elif self.network_type == "ANN": self.run_render_ANN(genome, env, seed, obs_max, obs_min)
        else: raise Exception("The network type is not recognized", self.network_type, "The network type must be SNN or ANN")

    def run_render_ANN(self, genome:Genome_NN, env:Environment, seed:int, obs_max:np.ndarray=None, obs_min:np.ndarray=None) -> None:
        seed = int(seed)
        envs_dict = self.environment_manager.create_environments(genomes_ids=[genome.id], seeds=[seed])
        self.env_render = envs_dict[genome.id][0]
        self.env_render.seed = seed
        self.env_render.reset(seed=seed)
        genome.fitness.reset()

        if obs_max is None or obs_min is None:
            obs_max:np.ndarray = np.full(genome.inputs, 5.0, dtype=np.float64) # 5 is arbitrary
            obs_min:np.ndarray = np.full(genome.inputs, -5.0, dtype=np.float64) # -5 is arbitrary
        self.env_render.update_observation_min_max_function(obs_max, obs_min)

        # 0 - Set net_torch
        self.runner.set_net_torch_population({genome.id:genome}, is_bias=self.is_bias)

        while self.env_render.terminated == False and self.env_render.truncated == False:
            # 1 - Observation
            observation:np.ndarray = self.env_render.encoding_observation_to_network_input()

            # 2 - Action
            action:Dict[int, np.ndarray] = self.runner.run_RL({genome.id:genome}, {genome.id:np.array([observation])})

            # 3 - Update
            self.env_render.update(action[genome.id][0], genome, seed)

            # 4 - Render
            if self.env_render.is_QDgym == False:
                self.env_render.gym_env.render()
            # time.sleep(0.01)
        
        # 5 - Unset net_torch
        self.runner.unset_net_torch_population({genome.id:genome})

        self.env_render.fitness_end_function({genome.id:genome}, [seed])
        print("genome", genome, "id", genome.id)
        print("after genome.score", genome.fitness.score)
        print("genome.best_fitness", genome.info["best_episode_raw_score"])

    def run_render_SNN(self, genome:Genome_NN, env:Environment, seed:int, obs_max:np.ndarray=None, obs_min:np.ndarray=None) -> None:
        seed = int(seed)
        envs_dict = self.environment_manager.create_environments(genomes_ids=[genome.id], seeds=[seed])
        self.env_render = envs_dict[genome.id][0]
        self.env_render.seed = seed
        self.env_render.reset(seed=seed)
        genome.fitness.reset()
        nns_cython_dict = self.runner.init_networks([genome])

        if obs_max is None:
            obs_max:np.ndarray = np.full(genome.inputs, 5.0, dtype=np.float64) # 5 is arbitrary
            obs_min:np.ndarray = np.full(genome.inputs, -5.0, dtype=np.float64) # -5 is arbitrary
        self.env_render.update_observation_min_max_function(obs_max, obs_min)
        # output_indexes:np.ndarray = genome.nn.outputs["neurons_indexes"]

        if "input" not in self.record_layer: output_indexes_record:np.ndarray = genome.nn.outputs["neurons_indexes"] - genome.nn.nb_inputs
        else: output_indexes_record:np.ndarray = genome.nn.outputs["neurons_indexes"]
        output_indexes:np.ndarray = genome.nn.outputs["neurons_indexes"]

        while self.env_render.terminated == False and self.env_render.truncated == False:
            # 1 - Observation
            observation:np.ndarray = self.env_render.encoding_observation_to_network_input()
            observation = np.array([[observation]], dtype=np.float64)

            # 1.1 - Shape the observation_spikes to match the input_multiplicator
            observation = np.repeat(observation, self.input_multiplicator, axis=2)

            # 2 - Action
            actions_dict:Dict[int, np.ndarray] = self.runner.run(observation)
    
            # 2.bis - Decoding Spikes
            actions_dict = self.decoding_spikes(actions_dict, output_indexes, output_indexes_record, {genome.id, genome})
            # for id, actions in actions_dict.items():
            #     actions_dict[id] = np.array([self.snn_decoder.rate(action[output_indexes], self.nb_original_outputs) for action in actions], dtype=np.float32)

            # 3 - Update
            self.env_render.update(actions_dict[genome.id][0], genome, seed)

            # 4 - Render
            if self.env_render.is_QDgym == False:
                self.env_render.gym_env.render()
            # time.sleep(0.01)

        self.env_render.fitness_end_function({genome.id:genome}, [seed])
        print("genome", genome, "id", genome.id)
        print("genome.score", genome.fitness.score)
        print("genome.best_fitness", genome.info["best_episode_raw_score"])

        