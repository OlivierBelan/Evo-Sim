from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Population import Population_NN
from evo_simulator.GENERAL.Index_Manager import get_new_population_id, get_new_genome_id
import evo_simulator.TOOLS as TOOLS

from problem.SL.SUPERVISED import Supervised_Manager as CPPN_Runner
from snn_simulator.runner_api_cython import Runner_Info
from snn_simulator.snn_decoder import Decoder

from typing import Dict, Any, List, Callable, Tuple, Union
import numpy as np
import numba as nb
import ray
import time




class HyperNetwork(Algorithm):
    def __init__(self, config_path_file:str, name:str = "HyperNetwork", extra_info:Dict[str, Any] = None) -> None:
        Algorithm.__init__(self, config_path_file, name, extra_info)
        

        # 1 - Initialize HyperNetwork configs
        self.config_hypernetwork:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["HyperNetwork", "Genome_NN"])
        self.verbose:bool = True if self.config_hypernetwork["HyperNetwork"]["verbose"] == "True" else False
        self.pop_size:int = int(self.config_hypernetwork["HyperNetwork"]["pop_size"])
        self.is_first_generation:bool = True
        self.cpu:int = extra_info["cpu"]

        self.inputs:int = int(self.config_hypernetwork["Genome_NN"]["inputs"])
        self.outputs:int = int(self.config_hypernetwork["Genome_NN"]["outputs"])
        self.hiddens_config:Dict[str, Union[Dict[str, List[int]], int]] = TOOLS.hiddens_from_config(self.config_hypernetwork["Genome_NN"]["hiddens"])


        # 2- Get substrat configuration
        hidden_names:List[str] = self.hiddens_config["layer_names"]
        substrat_function:Callable = self.extra_info["substrat_function"]
        inputs_coordinates, outputs_coordinates, hiddens_coordinates_list = substrat_function(self.inputs, self.outputs, [self.hiddens_config[hidden]["nb_neurons"] for hidden in hidden_names]) 
        architecture_coordinates:Dict[str, np.ndarray] = {"I": inputs_coordinates, "O": outputs_coordinates}
        genome_core:Genome_NN = Genome_NN(0, self.config_hypernetwork["Genome_NN"], self.attributes_manager)
        self.substrat_network_type:str = self.config_hypernetwork["Genome_NN"]["network_type"]
        for i in range(len(hiddens_coordinates_list)): architecture_coordinates[hidden_names[i]] = hiddens_coordinates_list[i]

        self.cppn_data_set, self.substrat_synapses_actives_indexes = self.__get_substrat_connection_and_build_cppn_data_set(genome_core.architecture, architecture_coordinates, genome_core.nn.architecture_neurons)

        # print("architecture\n", genome_core.architecture)
        # print("architecture_coordinates\n", architecture_coordinates)
        # print("architecture_neurons\n", genome_core.nn.architecture_neurons)
        # print("cppn_data_set", self.cppn_data_set, self.cppn_data_set.shape)
        # print("substrat_synapses_actives_indexes\n", self.substrat_synapses_actives_indexes, self.substrat_synapses_actives_indexes.shape)
        # print("genome_core.nn.synapses_actives_indexes\n", genome_core.nn.synapses_actives_indexes)
        # exit()
        
        
        # 3 - Initialize CPPN parameters 
        self.cppn_name, self.cppn_algorithm_builder, self.cppn_config_path, self.cppn_extra_info = self.extra_info["cppn_builder"]
        self.cppn_algorithm:Algorithm = self.cppn_algorithm_builder(self.cppn_config_path, self.cppn_name, self.cppn_extra_info)
        self.cppn_population:Population_NN = Population_NN(get_new_population_id(), self.cppn_config_path)
        self.config_cppn:Dict[str, Dict[str, Any]] = TOOLS.config_function(self.cppn_config_path, ["NEURO_EVOLUTION", "Genome_NN"])
        self.cppn_inputs:int = int(self.config_cppn["Genome_NN"]["inputs"])
        self.cppn_outputs:int = int(self.config_cppn["Genome_NN"]["outputs"])
        self.cppn_network_type:str = self.config_cppn["Genome_NN"]["network_type"]
        cppn_genome_core:Genome_NN = Genome_NN(0, self.config_cppn["Genome_NN"], self.attributes_manager)
        # 3.1 - Initialize CPPN SNN decoder
        if self.cppn_network_type == "SNN":
            self.cppn_snn_decoder = Decoder(self.cppn_config_path)
            self.cppn_runner_info:Runner_Info = Runner_Info(self.cppn_config_path)
            if "input" not in self.cppn_runner_info.record_layer: self.cppn_output_indexes_record:np.ndarray = cppn_genome_core.nn.outputs["neurons_indexes"] - cppn_genome_core.nn.nb_inputs
            else: self.cppn_output_indexes_record:np.ndarray = cppn_genome_core.nn.outputs["neurons_indexes"]
            self.cppn_output_indexes_nn:np.ndarray = cppn_genome_core.nn.outputs["neurons_indexes"]


        if self.cpu == 1:
            self.cppn_runner:CPPN_Runner = CPPN_Runner(config_path=self.cppn_config_path, features=None, labels=None, nb_generations=0)
            print("cppn_runner sequential", self.cppn_runner)
        else:
            cppn_runner_builder_ray:CPPN_Runner = ray.remote(CPPN_Runner)
            self.cppn_data_set_ray_id = ray.put(self.cppn_data_set)
            self.cppn_runners_ray = [cppn_runner_builder_ray.remote(config_path=self.cppn_config_path, features=None, labels=None, nb_generations=0) for _ in range(self.cpu)]
            self.cppn_populations_cpu:List[Population_NN] = [Population_NN(get_new_population_id(), self.cppn_config_path) for _ in range(self.cpu)]
            print("cppn_runners parallel", self.cppn_runners_ray)


        # 4 - Check if all population have the same size
        if self.cppn_algorithm.pop_size != self.pop_size: raise Exception("HyperNetwork: pop_size (" + str(self.pop_size) + ") must be equal to CPPN_algo pop_size (" + str(self.cppn_algorithm.pop_size) +")")
        if self.cppn_inputs != self.cppn_data_set.shape[1]: raise Exception("HyperNetwork: CPPN inputs (" + str(self.cppn_inputs) + ") must be equal to CPPN data set size (" + str(self.cppn_data_set.shape[1]) +")")
        
        # Utils variables
        self.cppn_record:Dict[int, np.ndarray] = {}



    def __get_substrat_connection_and_build_cppn_data_set(self, architectures_layer_connection:List[List[str]], architectures_substrate_coordinates:Dict[str, np.ndarray], nn_architecture_neurons:Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        substrat_connection_data_set:np.ndarray = np.empty((0, architectures_substrate_coordinates["I"].shape[1] + architectures_substrate_coordinates["O"].shape[1]))
        synpases_indexes:np.ndarray = np.empty((2, 0), dtype=np.int32)
        # substrat_layer_connection:Dict[str, np.ndarray] = {}
        for connection in architectures_layer_connection:
            substrat_connection_data_set = np.concatenate((substrat_connection_data_set, self.__combine_coordinate(architectures_substrate_coordinates[connection[0]], architectures_substrate_coordinates[connection[1]])))
            synpases_indexes = np.concatenate((synpases_indexes, self.__combine_connections(nn_architecture_neurons[connection[0]]["neurons_indexes"], nn_architecture_neurons[connection[1]]["neurons_indexes"])), axis=1)
            # substrat_layer_connection[connection[0] + "->" + connection[1]] = self.combine_connections(nn_architecture_neurons[connection[0]]["neurons_indexes"], nn_architecture_neurons[connection[1]]["neurons_indexes"])
        # return substrat_layer_connection, substrat_connection_data_set, synpases_indexes
        return substrat_connection_data_set, synpases_indexes
        

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __combine_coordinate(a:np.ndarray, b:np.ndarray) -> np.ndarray:
        combined_array:np.ndarray = np.zeros((a.shape[0] * b.shape[0], a.shape[1] + b.shape[1]))
        for i in nb.prange(a.shape[0]):
            for j in nb.prange(b.shape[0]):
                if np.array_equal(a[i], b[j]) == False:
                    combined_array[i* b.shape[0] + j] = np.concatenate((a[i], b[j]))
        return combined_array
    def __combine_connections(self, a:np.ndarray, b:np.ndarray) -> np.ndarray:
        a, b = np.meshgrid(a, b)
        return np.array([a.ravel(), b.ravel()])


    def __run_cppn_sequential(self) -> None:

        # 0 - Run CPPN_algorithm
        self.cppn_population = self.cppn_algorithm.run(self.cppn_population)

        # 1 - Run CPPN_runner
        if self.cppn_network_type == "ANN":
            self.cppn_record:Dict[int, np.ndarray] = self.cppn_runner.run_anns(self.cppn_population.population, self.cppn_data_set)
        elif self.cppn_network_type == "SNN":
            self.cppn_record:Dict[int, np.ndarray] = self.cppn_runner.run_snns(self.cppn_population.population, self.cppn_data_set)[self.cppn_runner_info.record_type]
            self.cppn_record = self.decoding_spikes(self.cppn_record, self.cppn_population)
            # print("self.cppn_record", self.cppn_record)
            # exit()

    def __run_cppn_parallel(self) -> None:

        # 0 - Run CPPN_algorithm
        self.cppn_population = self.cppn_algorithm.run(self.cppn_population)
        
        # 1 - Check and Split population for each cpu
        pop_size:int = len(self.cppn_population.population)
        if pop_size < self.cpu: raise ValueError("pop_size ("+ str(pop_size)+") must be equal or greater than cpu number (" + str(self.cpu) + ")")
        self.cppn_record:Dict[int, np.ndarray] = {}

        # 2 - Build cpu populations in a list of populations
        self.__set_populations_cpu(self.cppn_population, self.cppn_populations_cpu)

        # 3 - Run CPPN_algorithm parallel
        if self.cppn_network_type == "ANN":
            records_ids = [self.cppn_runners_ray[i].run_anns.remote(self.cppn_populations_cpu[i].population, self.cppn_data_set_ray_id) for i in range(self.cpu)]
        elif self.cppn_network_type == "SNN":
            records_ids = [self.cppn_runners_ray[i].run_snns.remote(self.cppn_populations_cpu[i].population, self.cppn_data_set_ray_id) for i in range(self.cpu)]
        
        for i in range(self.cpu): self.cppn_populations_cpu[i].population = {} # free memory (technically not real free but still...)
        # self.cppn_population.population = {} # free memory (technically not real free but still...)

        # 4 - Get results from ray
        results = ray.get(records_ids)

        if self.cppn_network_type == "ANN":
            for i in range(self.cpu): self.cppn_record.update(results[i])
        elif self.cppn_network_type == "SNN":
            for i in range(self.cpu): self.cppn_record.update(results[i][self.cppn_runner_info.record_type])
            self.cppn_record = self.decoding_spikes(self.cppn_record, self.cppn_population)
            # print("results", results)
            # print("self.cppn_record", self.cppn_record)
            # exit()

    def __run_cppn(self) -> None:
        if self.cpu == 1:
            self.__run_cppn_sequential()
        else:
            self.__run_cppn_parallel()


    def __set_populations_cpu(self, population:Population_NN, populations_cpu:List[Population_NN]) -> None:
        indexes:List[List[int]] = TOOLS.split(list(population.population.keys()), self.cpu)
        for i in range(self.cpu):
            populations_cpu[i].population = {key:population.population[key] for key in indexes[i]}


    def run(self, global_population:Population_NN) -> Population_NN:

        self.population_manager.population = global_population.population

        # 0 - First generation
        if self.is_first_generation:
            self.__first_generation(self.population_manager)
            return self.population_manager

        # 1 - Syncronize fitness
        self.__syncronize_fitness(self.population_manager)

        # 2 - Run CPPN
        self.__run_cppn()

        # 3 - Builds NNs from CPPNs
        self.__build_substrat_population(self.population_manager)
        
        # 4 - Update population
        global_population.population = self.population_manager.population

        return global_population


    def __first_generation(self, population_manager:Population_NN) -> None:
        self.is_first_generation = False
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_hypernetwork["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=True)
            population[new_genome.id] = new_genome
        self.__run_cppn()
        self.__build_substrat_population(population_manager)

    def __build_substrat_population(self, population_manager:Population_NN) -> None:
        hypernetwork_population:List[Genome_NN] = list(population_manager.population.values())
        substrat_population:Dict[int, Genome_NN] = {}

        for index, (cppn_id, cppn_output) in enumerate(self.cppn_record.items()):
            substrat_genome:Genome_NN = hypernetwork_population[index]
            if self.cpu > 1:
                substrat_genome.nn.parameters["weight"] = substrat_genome.nn.parameters["weight"].copy()
            # print("cppn_output", cppn_output, cppn_output.shape)
            self.__build_nn_from_cppn(
                cppn_output,
                substrat_genome.nn.parameters["weight"],
                self.substrat_synapses_actives_indexes
                )
            # print("substrat_genome.nn.parameters[weight]", substrat_genome.nn.parameters["weight"])
            # print("allo")
            # exit()
            substrat_genome.id = cppn_id
            substrat_genome.nn.update_indexes()
            substrat_population[cppn_id] = substrat_genome
        population_manager.population = substrat_population

            

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __build_nn_from_cppn(
                                cppn_output:np.ndarray, 
                                weights:np.ndarray,
                                synapse_indexes:np.ndarray
                              ) -> None:


        # 2 - set values to NN parameters (bias) and synapses (weights)
        # Order is very important (it depend of the global_parameters.nn_neuron_parameters and global_parameters.nn_synapse_parameters)
        # for i in nb.prange(bias.shape[0]):
        #     bias[i]       = cppn_neurons[i, 0]
        
        for i in nb.prange(cppn_output.shape[0]):
            weights[synapse_indexes[0, i], synapse_indexes[1, i]] = cppn_output[i, 0]

        # print("weights", weights)
        # print("bias", bias)


    def __syncronize_fitness(self, population_manager:Population_NN) -> None:
        substrat_pop:Dict[int, Genome_NN] = population_manager.population
        cppn_pop:Dict[int, Genome_NN] = self.cppn_population.population

        for substrat_id, substrat_genome in substrat_pop.items():
            cppn_pop[substrat_id].fitness.score = substrat_genome.fitness.score
        
        self.cppn_population.update_info()


    def decoding_spikes(self, actions_dict:Dict[int, np.ndarray], population:Population_NN) -> Dict[int, np.ndarray]:
        genomes:Dict[int, Genome_NN] = population.population
        if self.cppn_runner_info.decoder == "augmented": return actions_dict

        # print("actions_dict", actions_dict)
        # exit()
        for id, actions in actions_dict.items():
            if self.cppn_runner_info.decoder == "max_spikes": 
                actions_dict[id] = np.array([self.cppn_snn_decoder.max_spikes(action[self.cppn_output_indexes_record], self.cppn_outputs) for action in actions], dtype=np.float32)
            
            elif self.cppn_runner_info.decoder == "augmented":
                pass
                # actions_dict[id] = np.interp(
                #                                 np.clip(actions[:, output_indexes_record]/self.cppn_runner_info.spike_max, 0, 1),
                #                                 [0, 1], [self.cppn_runner_info.interpolate_min, self.cppn_runner_info.interpolate_min]
                #                             ) # does not work if output_multiplicator > 1
            elif self.cppn_runner_info.decoder == "rate":
                # print("actions", actions)
                # print("self.cppn_output_indexes_record", self.cppn_output_indexes_record)
                # print("self.cppn_outputs", self.cppn_outputs)
                # print("self.cppn_runner_info.ratio_max_output_spike", self.cppn_runner_info.ratio_max_output_spike)
                # print("actions[0][self.cppn_output_indexes_record]", actions[0][self.cppn_output_indexes_record])
                actions_dict[id] = np.array([self.cppn_snn_decoder.rate(action[self.cppn_output_indexes_record], self.cppn_outputs, self.cppn_runner_info.ratio_max_output_spike) for action in actions], dtype=np.float32)
            
            elif self.cppn_runner_info.decoder == "voltage":
                voltage_min:np.ndarray = genomes[id].nn.parameters["voltage"][self.cppn_output_indexes_nn] if self.cppn_runner_info.is_voltages_min_decoder == True else self.cppn_runner_info.voltage_min
                voltage_max:np.ndarray = genomes[id].nn.parameters["threshold"][self.cppn_output_indexes_nn] if self.cppn_runner_info.is_threshold_max_decoder == True else self.cppn_runner_info.voltage_max
                actions_dict[id] = self.cppn_snn_decoder.voltage(actions[:, self.cppn_output_indexes_record], voltage_min, voltage_max)
            
            elif self.cppn_runner_info.decoder == "coeff":
                actions_dict[id] = np.array([self.cppn_snn_decoder.coefficient(action[self.cppn_output_indexes_record], genomes[id].nn.parameters["coeff"][self.cppn_output_indexes_nn], self.cppn_outputs) for action in actions], dtype=np.float32)

        return actions_dict
