from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Genome import Genome, Genome_NN, Genome_Decoder
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Distance import Distance
from evo_simulator.GENERAL.Index_Manager import get_new_population_id, get_new_genome_id
from evo_simulator.ALGORITHMS.NEAT.Neat import NEAT
from ALGORITHMS.Decoder.Decoder_GA import Decoder_GA
from evo_simulator.ALGORITHMS.Decoder.Decoder_CMA import Decoder_CMA

import evo_simulator.TOOLS as TOOLS
# import evo_simulator.GENERAL.Globals as global_parameters

from problem.SL.SUPERVISED import Supervised_Manager as CPPN_Runner

from typing import Dict, Any, List, Callable, Tuple
import time
import numpy as np
import numba as nb



class HyperNEAT_SNN_ANN(Algorithm):
    def __init__(self, config_path_file:str, cppn_synpases_config_path:str, cppn_neurons_config_path:str, substrats:List[np.ndarray], substrats_connection:List[Tuple[int, int]], name:str = "HyperNEAT") -> None:
        Algorithm.__init__(self, config_path_file, name)
        
        self.substrats:List[np.ndarray] = substrats

        # Initialize configs
        self.config_hyperneat:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["HyperNEAT", "Genome_NN"])
        self.verbose:bool = True if self.config_hyperneat["HyperNEAT"]["verbose"] == "True" else False
        self.pop_size:int = int(self.config_hyperneat["HyperNEAT"]["pop_size"])
        self.is_first_generation:bool = True
        self.nb_neurons:int = 0
        self.distance:Distance = Distance(config_path_file)

        # Initialize population 
        self.population_cppn_synpases:Population = Population(get_new_population_id(), cppn_synpases_config_path)
        self.population_cppn_neurons:Population = Population(get_new_population_id(), cppn_neurons_config_path)
        self.genome_core:Genome_NN = Genome_NN(0, self.config_hyperneat["Genome_NN"], self.attributes_manager)
        self.genome_core.nn.synapses_status[:, :] = False
        self.genome_core.nn.update_indexes()

        # Initialize substrat configuration
        self.substrat_synpases_configuration, self.substrat_neurons_configuration = self.init_substrat_configuration(substrats, substrats_connection)

        # Intialize CPPNs (algorithms and runners) configuration (here NEAT)        
        self.cppn_synpases_algorithm:NEAT = NEAT(cppn_synpases_config_path, name="CPPN_synpases_NEAT")
        self.cppn_neurons_algorithm:NEAT = NEAT(cppn_neurons_config_path, name="CPPN_neurons_NEAT")

        self.cppn_synpases_runner:CPPN_Runner = CPPN_Runner(None, cppn_synpases_config_path)
        self.cppn_neurons_runner:CPPN_Runner = CPPN_Runner(None, cppn_neurons_config_path)

        self.cppn_synpases_runner.features_batches = self.substrat_synpases_configuration.shape[0]
        self.cppn_synpases_runner.runner.batch_features = self.substrat_synpases_configuration.shape[0]

        self.cppn_neurons_runner.features_batches = self.substrat_neurons_configuration.shape[0]
        self.cppn_neurons_runner.runner.batch_features = self.substrat_neurons_configuration.shape[0]

        print("self.cppn_neurons_runner.features_batches",self.cppn_neurons_runner.features_batches)
        print("self.substrat_neurons_configuration (shape):", self.substrat_neurons_configuration.shape)
        print("self.substrat_synpases_configuration (shape):", self.substrat_synpases_configuration.shape)


        # Check if all population have the same size
        if self.cppn_synpases_algorithm.pop_size != self.pop_size: raise Exception("HyperNEAT: pop_size (" + str(self.pop_size) + ") must be equal to CPPN pop_size (" + str(self.cppn_synpases_algorithm.pop_size) +")")
        if self.cppn_neurons_algorithm.pop_size != self.pop_size: raise Exception("HyperNEAT: pop_size (" + str(self.pop_size) + ") must be equal to CPPN pop_size (" + str(self.cppn_neurons_algorithm.pop_size) +")")
        if self.cppn_neurons_algorithm.pop_size != self.cppn_synpases_algorithm.pop_size: raise Exception("HyperNEAT: CPPN_synpases pop_size (" + str(self.cppn_synpases_algorithm.pop_size) + ") must be equal to CPPN_neurons pop_size (" + str(self.cppn_neurons_algorithm.pop_size) +")")
        if self.genome_core.nn.nb_inputs != self.substrats[0].shape[0]: raise Exception("HyperNEAT: NN nb_inputs (" + str(self.genome_core.nn.nb_inputs) + ") must be equal to substrat size (" + str(self.substrats[0].shape[0]) +")")
        if self.genome_core.nn.nb_hidden_active != self.substrats[1].shape[0]: raise Exception("HyperNEAT: NN nb_hidden_active (" + str(self.genome_core.nn.nb_hidden_active) + ") must be equal to self.substrats[1].shape[0] size (" + str(self.substrats[1].shape[0]) +")")
        if self.genome_core.nn.nb_outputs != self.substrats[2].shape[0]: raise Exception("HyperNEAT: NN nb_outputs (" + str(self.genome_core.nn.nb_outputs) + ") must be equal to substrat size (" + str(self.substrats[2].shape[0]) +")")

        # Utils variables
        self.cppn_neurons_record:Dict[int, np.ndarray] = {}
        self.cppn_synapses_dict:Dict[int, np.ndarray] = {}

        self.cppn_neurons_output_indexes:np.ndarray = None
        self.cppn_synapses_output_indexes:np.ndarray = None

    def init_substrat_configuration(self, substrats:List[np.ndarray], substrats_connection:List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        substrat_synapses:List[List[float]] = []
        substrat_neurons:List[Tuple[int, int]] = []

        core_layer:Dict[int, Any] = {0:self.genome_core.nn.inputs, 1:self.genome_core.nn.hiddens, 2:self.genome_core.nn.outputs}

        # 1 - Build substrat synapses data & and get synapses indexes (neuron_in to neuron_out)
        for i in range(len(substrats_connection)):
            substrat_synapses.extend(self.combine_arrays(substrats[substrats_connection[i][0]], substrats[substrats_connection[i][1]]).tolist())
            self.genome_core.nn.connect_layers(core_layer[substrats_connection[i][0]], core_layer[substrats_connection[i][1]])
        self.genome_core.nn.update_indexes()
        substrat_synapses_array:np.ndarray = np.array(substrat_synapses, dtype=np.float64)
        substrat_synapses_array = np.array([sub_arr for sub_arr in substrat_synapses_array if len(np.unique(sub_arr)) > 1])

        idx = np.lexsort((substrat_synapses_array[:, 3], substrat_synapses_array[:, 2], substrat_synapses_array[:, 1], substrat_synapses_array[:, 0]))
        substrat_synapses_array = substrat_synapses_array[idx]

        # 2 - Build substrat neurons data
        for sub_substrat in substrats:
            substrat_neurons.extend(sub_substrat.tolist())
        substrat_neurons_array:np.ndarray = np.array(substrat_neurons, dtype=np.float64)
        
        return substrat_synapses_array, substrat_neurons_array

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def combine_arrays(a:np.ndarray, b:np.ndarray) -> np.ndarray:
        combined_array:np.ndarray = np.zeros((a.shape[0] * b.shape[0], a.shape[1] + b.shape[1]))
        for i in nb.prange(a.shape[0]):
            for j in nb.prange(b.shape[0]):
                if np.array_equal(a[i], b[j]) == False:
                    combined_array[i* b.shape[0] + j] = np.concatenate((a[i], b[j]))
        return combined_array
    
    def run_neurons_cppn(self, genomes_from_cppn:Dict[int, Genome]) -> None:
        # 1 - Run CPPN
        self.cppn_neurons_record:Dict[int, np.ndarray] = self.cppn_neurons_runner.run_anns(genomes_from_cppn, self.substrat_neurons_configuration)
        
        # 2 - Get output indexes (if not already done)
        if self.cppn_neurons_output_indexes is None:
            first_key:int = next(iter(self.population_cppn_neurons.population))
            genome:Genome_NN = self.population_cppn_neurons.population[first_key]
            self.cppn_neurons_output_indexes:np.ndarray = genome.nn.outputs["neurons_indexes"]
        # print("substrat_neurons_configuration", self.substrat_neurons_configuration.shape)
        # print("self.cppn_neurons_record", self.cppn_neurons_record, len(self.cppn_neurons_record))
        # print("cppn_neurons_output_indexes", self.cppn_neurons_output_indexes)
        # print("self.cppn_neurons_record[self.cppn_neurons_output_indexes]", self.cppn_neurons_record[first_key][0])
        # exit()

    def run_synpases_cppn(self, genomes_from_cppn:Dict[int, Genome]) -> None:
        
        # 1 - Run CPPN
        self.cppn_synapses_record:Dict[int, np.ndarray] = self.cppn_synpases_runner.run_anns(genomes_from_cppn, self.substrat_synpases_configuration)

        # 2 - Get output indexes (if not already done)
        if self.cppn_synapses_output_indexes is None:
            first_key:int = next(iter(self.population_cppn_synpases.population))
            genome:Genome_NN = self.population_cppn_synpases.population[first_key]
            self.cppn_synapses_output_indexes:np.ndarray = genome.nn.outputs["neurons_indexes"]
        # print("cppn_synapses_output_indexes", self.cppn_synapses_output_indexes)
        # # print("self.cppn_synapses_record", self.cppn_synapses_record)
        # print("self.cppn_synapses_record[self.cppn_neurons_output_indexes]", self.cppn_synapses_record[first_key][0])
        # exit()


    def run(self, global_population:Population, evaluation_function) -> Population:

        # self.population_manager.population = global_population.population

        # 0 - Run CPPNs
        self.first_generation(self.population_manager, evaluation_function)
        self.cppn_neurons_algorithm.run(self.population_cppn_neurons, self.run_neurons_cppn)
        self.cppn_synpases_algorithm.run(self.population_cppn_synpases, self.run_synpases_cppn)

        # 1 - Builds NNs from CPPNs
        self.__build_HyperNEAT_population()
        

        # 2 - Evaluate substrats population
        start_time = time.time()
        self.__evalutation(self.population_manager, evaluation_function)
        print(self.name+": Evaluation time:", time.time() - start_time, "s")

        # 3 - Syncronize fitness
        start_time = time.time()
        self.__syncronize_fitness(self.population_manager)
        print(self.name+": Syncronize fitness time:", time.time() - start_time, "s")

        # 4 - Print stats
        self.__print_stats()
        # exit()
        # 5 - Update population
        global_population.population = self.population_manager.population

        return global_population


    def first_generation(self, population_manager:Population, evaluation_function:Callable) -> None:
        if  population_manager.is_first_generation == True:
            start_time = time.time()
            self.is_first_generation = False
            population_manager.is_first_generation = False
            self.ajust_population(population_manager)
            print("HyperNEAT: First generation time:", time.time() - start_time, "s")

    def ajust_population(self, population_manager:Population) -> None:
        population:Dict[int, Genome_NN] = population_manager.population
        while len(population) < self.pop_size:
            new_genome:Genome_NN = Genome_NN(get_new_genome_id(), self.config_hyperneat["Genome_NN"], self.attributes_manager)
            new_genome.nn.set_arbitrary_parameters(is_random=False, weight_random=True)
            new_genome.nn.synapses_status[:] = self.genome_core.nn.synapses_status[:]
            new_genome.nn.update_indexes()
            population[new_genome.id] = new_genome

    def __build_HyperNEAT_population(self):
        hyperneat_population:List[Genome_NN] = list(self.population_manager.population.values())
        cppn_neurons_population:List[np.ndarray] = list(self.cppn_neurons_record.values())
        cppn_synapses_population:List[np.ndarray] = list(self.cppn_synapses_record.values())
        synapse_indexes:np.ndarray = self.genome_core.nn.synapses_actives_indexes

        # for index in range(self.pop_size):
        for (id_genome, genome), (id_cppn_neuron, record_cppn_neuron), (id_cppn_synapse, record_cppn_synpase) in zip(self.population_manager.population.items(), self.cppn_neurons_record.items(), self.cppn_synapses_record.items()):

            # genome:Genome_NN = hyperneat_population[index]
            # nn_parameters:Dict[str, np.ndarray] = hyperneat_population[index].nn.parameters

            genome.info["id_cppn_neurons"] = id_cppn_neuron
            genome.info["id_cppn_synapses"] = id_cppn_synapse
            nn_parameters:Dict[str, np.ndarray] = genome.nn.parameters

            if genome.nn.network_type != "SNN": raise Exception("HyperNEAT: NN network_type (" + str(genome.nn.network_type) + ") must be equal to SNN")
            self.__build_snn_from_cppn(
                # cppn_neurons_population[index][:, self.cppn_neurons_output_indexes],
                # cppn_synapses_population[index][:, self.cppn_synapses_output_indexes],
                # cppn_neurons_population[index],
                # cppn_synapses_population[index],
                record_cppn_neuron,
                record_cppn_synpase,
                nn_parameters["voltage"],
                nn_parameters["threshold"],
                nn_parameters["tau"],
                nn_parameters["input_current"],
                nn_parameters["refractory"],
                nn_parameters["weight"],
                nn_parameters["delay"],
                synapse_indexes
                )
            # hyperneat_population[index].nn.update_indexes()
            genome.nn.update_indexes()
            # print("genome[id_cppn_neurons]", genome.info["id_cppn_neurons"])
            # print("genome[id_cppn_synapses]", genome.info["id_cppn_synapses"])
            # exit()

            

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __build_snn_from_cppn(
                                cppn_neurons:np.ndarray, 
                                cppn_synapses:np.ndarray, 
                                voltage:np.ndarray,
                                threshold:np.ndarray,
                                tau:np.ndarray,
                                input_current:np.ndarray,
                                refractory:np.ndarray,
                                weights:np.ndarray,
                                delays:np.ndarray,
                                synapse_indexes:np.ndarray
                              ) -> None:

        # print("cppn_neurons", cppn_neurons)
        # print("cppn_synapses", cppn_synapses)
        # 1 - Decoding Spikes to numerical values
        cppn_neurons = cppn_neurons.astype(np.float32)
        cppn_synapses = cppn_synapses.astype(np.float32)

        # print("before voltage", voltage)
        # print("before threshold", threshold)
        # print("before tau", tau)
        # print("before input_current", input_current)
        # print("before refractory", refractory)
        # print("before weights", weights)
        # print("before delays", delays)


        # 2 - set values to NN parameters (voltage, threshold, tau, input_current, refractory) and synapses (weights, delays)
        # Order is very important (it depend of the global_parameters.nn_neuron_parameters and global_parameters.nn_synapse_parameters)
        # for i in nb.prange(voltage.shape[0]):
        #     voltage[i]       = cppn_neurons[i][0]
        #     threshold[i]     = cppn_neurons[i][1]
        #     tau[i]           = cppn_neurons[i][2]
        #     input_current[i] = cppn_neurons[i][3]
        #     refractory[i]    = cppn_neurons[i][4]
        
        for i in nb.prange(cppn_synapses.shape[0]):
            weights[synapse_indexes[0][i], synapse_indexes[1][i]] = cppn_synapses[i][0]
            delays[synapse_indexes[0][i], synapse_indexes[1][i]]  = cppn_synapses[i][1]
            # weights[synapse_indexes[0][i], synapse_indexes[1][i]] = transformer_valeur(cppn_synapses[i][0], -1, 1, -2, 2)
            # delays[synapse_indexes[0][i], synapse_indexes[1][i]]  = transformer_valeur(cppn_synapses[i][1], -1, 1, 0, 5)

        # print("before after voltage", voltage)
        # print("before after threshold", threshold)
        # print("before after tau", tau)
        # print("before after input_current", input_current)
        # print("before after refractory", refractory)
        # print("before after weights", weights)
        # print("before after delays", delays)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def __build_nn_from_cppn(
                                cppn_neurons:np.ndarray, 
                                cppn_synapses:np.ndarray, 
                                decodeur_neurons:np.ndarray,
                                decodeur_synapses:np.ndarray,
                                bias:np.ndarray,
                                weights:np.ndarray,
                                synapse_indexes:np.ndarray
                              ) -> None:

        # print("cppn_neurons", cppn_neurons)
        # print("cppn_synapses", cppn_synapses)
        # print("decodeur_neurons", decodeur_neurons)
        # print("decodeur_synapses", decodeur_synapses)
        # 1 - Decoding Spikes to numerical values
        cppn_neurons = cppn_neurons.astype(np.float32)
        cppn_synapses = cppn_synapses.astype(np.float32)
        # cppn_neurons += 1
        cppn_neurons *= decodeur_neurons
        # cppn_synapses += 1
        cppn_synapses *= decodeur_synapses


        # 2 - set values to NN parameters (bias) and synapses (weights)
        # Order is very important (it depend of the global_parameters.nn_neuron_parameters and global_parameters.nn_synapse_parameters)
        for i in nb.prange(bias.shape[0]):
            bias[i]       = cppn_neurons[i][0]
        
        for i in nb.prange(cppn_synapses.shape[0]):
            weights[synapse_indexes[0][i], synapse_indexes[1][i]] = cppn_synapses[i][0]

        # print("bias", bias)
        # print("threshold", threshold)
        # print("tau", tau)
        # print("input_current", input_current)
        # print("refractory", refractory)
        # print("weights", weights)
        # print("delays", delays)


    def __syncronize_fitness(self, population_manager:Population) -> None:
        hyperneat_pop:Dict[int, Genome_NN] = population_manager.population
        cppn_neurons_pop:Dict[int, Genome] = self.population_cppn_neurons.population
        cppn_synapses_pop:Dict[int, Genome] = self.population_cppn_synpases.population
        # for hyperneat, neuron, synapse in zip(hyperneat_pop.values(), cppn_neurons_pop.values(), cppn_synapses_pop.values()):
        #     neuron.fitness.score = hyperneat.fitness.score
        #     synapse.fitness.score = hyperneat.fitness.score
        for genome in hyperneat_pop.values():
            cppn_neurons_pop[genome.info["id_cppn_neurons"]].fitness.score = genome.fitness.score
            cppn_synapses_pop[genome.info["id_cppn_synapses"]].fitness.score = genome.fitness.score



    def __evalutation(self, population_manager:Population, evaluation_function:Callable) -> None:
        if evaluation_function is not None:
            evaluation_function(population_manager.population)
        else:
            self.evaluation_function(population_manager.population)

    def __get_info_stats_population(self):
        stats:List[List[int, float, float, int]] = []
        # self.population_manager.update_info()
        best_fitness:float = self.population_manager.fitness.score
        mean_fitness:float = self.population_manager.fitness.mean
        stagnation:float = self.population_manager.stagnation
        best_genome:Genome_NN = self.population_manager.best_genome
        nb_neurons:int = len(best_genome.nn.hiddens["neurons_indexes_active"])
        nb_synapses:int = best_genome.nn.synapses_actives_indexes[0].size
        stats.append([0, len(self.population_manager.population), (best_genome.id, round(best_fitness, 3), nb_neurons, nb_synapses), round(mean_fitness, 3), stagnation])
        return stats

    def __get_info_distance(self):
        elite_id:int = self.population_manager.best_genome.id
        population_ids:List[int] = self.population_manager.population.keys()
        pop_dict:Dict[int, Genome] = self.population_manager.population
        self.distance.distance_genomes_list([elite_id], population_ids, pop_dict, reset_cache=True)
        print("global distance:", self.distance.mean_distance["global"], ", local distance:", self.distance.mean_distance["local"])
        mean_distance:float = self.distance.mean_distance["global"]
        print("Mean_distance (compared with one elite only):", round(mean_distance, 3))
        print("--------------------------------------------------------------------------------------------------------------------->>> " +self.name)

    def __print_stats(self):
        if self.verbose == False: return
        self.population_manager.update_info()
        self.__get_info_distance()
        titles = [[self.name, "Size", "Best(id, fit, neur, syn)", "Avg", "Stagnation"]]
        titles.extend(self.__get_info_stats_population())
        col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for row in titles:
            print("".join(str(word).ljust(col_width) for word in row))
        print("\n")

@nb.njit(cache=True, fastmath=True, nogil=True)
def transformer_valeur(value, min_input=0, max_input=1, min_output=-1, max_output=1):
    """Transforme une valeur proportionnellement d'une plage de départ vers une plage de sortie."""
    return min_output + (value - min_input) / (max_input - min_input) * (max_output - min_output)
