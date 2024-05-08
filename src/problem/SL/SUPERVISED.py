import sys

sys.setrecursionlimit(100000)
from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Fitness import Fitness_Manager
from evo_simulator.GENERAL.Population import Population
from evo_simulator import TOOLS

from snn_simulator.runner_api_cython import Runner_Info
from snn_simulator.runner_api_cython import Runner as SNN_Runner
from ann_simulator.runner import ANN_Runner

from problem.Problem import Problem

import random
from typing import List, Dict, Tuple, Any
import numpy as np

class Supervised_Manager(Runner_Info, Problem):
    def __init__(self, config_path:str, features:np.ndarray, labels:np.ndarray, nb_generations:int=None):
        Runner_Info.__init__(self, config_path)
        Problem.__init__(self)
        # Public variables
        self.run_number:int = 0
        self.config_path:str = config_path
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Genome_NN", "NEURO_EVOLUTION"])
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.nb_generations:int =  nb_generations if nb_generations != None else int(self.config["NEURO_EVOLUTION"]["nb_generations"])

        if self.network_type == "SNN":
            self.runner = SNN_Runner(self.config_path) # Initialize the Runner
        elif self.network_type == "ANN":
            self.runner = ANN_Runner(self.config_path)
        self.is_bias:bool = TOOLS.is_config_section(config_path, "bias_neuron_parameter")

        # Private variables
        self.__batch_index:int = 0
        self.__build_batches(features, labels) # create batches of data set
    

    def run(self, population:Population, run_nb:int, generation:int, seed:int=None, indexes:set=None) -> Population:  
        if indexes is not None:
            indexes = set(indexes)
            genomes:Dict[int, Genome_NN] = {genome.id:genome for i , genome in enumerate(population.population.values()) if i in indexes}
        else:
            genomes:Dict[int, Genome_NN] = population.population
        self.run_number:int = run_nb
        self.generation:int = generation

        # 1 - Select Batch for the run
        features_batch, labels_batch = self.__select_random_batch()
        
        # 2 - RUN NNs
        if self.network_type == "SNN":
            record_dict: Dict[int, np.ndarray] = self.run_snns(genomes, features_batch)[self.record_type]
        elif self.network_type == "ANN":
            record_dict: Dict[int, np.ndarray] = self.run_anns(genomes, features_batch)
        
        # 3 - CHECK SNNs ACCURACY AND UPDATE GENOMES FITNESS
        nn_accuracy = self.__evaluate_accuracy_and_update_genomes_fitnesses(genomes, record_dict, labels_batch)
        return population

    def run_anns(self, genomes:Dict[int,  Genome_NN], features:np.ndarray) -> Dict[int, np.ndarray]:
        # print("----------------------------RUN ANNs----------------------------")
        # 0 - Check if the number of inputs of the genome is equal to the number of features of the data set
        if genomes[next(iter(genomes))].nn.nb_inputs != len(features[0]):
            raise Exception(str("The number of inputs of the genome (" + str(genomes[0].inputs) + ") is different from the number of features of the data set (" + str(len(features[0]))+ ")"))
        self.runner.set_net_torch_population(genomes, self.is_bias)
        records_dict:Dict[int, np.ndarray] = self.runner.run_SL(genomes, features)
        self.runner.unset_net_torch_population(genomes)
        return records_dict

    def run_snns(self, genomes:Dict[int,  Genome_NN], features:np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
        # print("----------------------------RUN SNNs----------------------------")
        # 0 - Check if the number of inputs of the genome is equal to the number of features of the data set
        if genomes[next(iter(genomes))].nn.nb_inputs != len(features[0]):
            raise Exception(str("The number of inputs of the genome (" + str(genomes[0].inputs) + ") is different from the number of features of the data set (" + str(len(features[0]))+ ")"))

        # 1 - Initialize NNs in the Runner
        self.runner.init_networks(genomes.values())

        # 2 - Initialize Inputs in the Runner
        self.runner.init_inputs_networks(features) # Set the inputs spikes

        # 3 - Run NNs in the Runner
        records = self.runner.run()

        return records

    def __evaluate_accuracy_and_update_genomes_fitnesses(self, genomes:Dict[int, Genome_NN], record_dict:Dict[int, np.ndarray], labels:np.ndarray) -> np.ndarray:
        nets_accuracy:np.ndarray = np.empty(len(genomes))
        if self.network_type == "SNN":
            first_genome:Genome_NN = genomes[next(iter(genomes))]
            if "input" not in self.record_layer:
                output_indexes:np.ndarray = first_genome.nn.outputs["neurons_indexes_formated"] - first_genome.inputs
            else:
                output_indexes:np.ndarray = first_genome.nn.outputs["neurons_indexes_formated"]
        for i, (id, genome) in enumerate(genomes.items()):
            if self.network_type == "SNN":                
                nets_accuracy[i] = Fitness_Manager.deterministic_neuron_output_accuracy_spikes(output_indexes, record_dict[id], labels)
                # nets_accuracy[i] = Fitness_Manager.probabilistic_neuron_output_accuracy(genome.nn.outputs["neurons_indexes_formated"], record_dict[id], labels)
            elif self.network_type == "ANN":
                nets_accuracy[i] = Fitness_Manager.deterministic_neuron_output_accuracy_numerical(record_dict[id], labels)
            genome.fitness.score = nets_accuracy[i]
        return nets_accuracy

    def __build_batches(self, features:List[List[float]], labels:List[float]):
        self.batch_features:int = min(self.batch_features, len(labels))
        self.batch_running:int = min(self.batch_running, self.batch_features)
        data_set:Tuple[(List[List[float]], List[float])] = list(zip(features, labels))
        random.shuffle(data_set)
        data_set = list(zip(*data_set))
        labels:List[float] = data_set[1]
        features:List[List[float]] = data_set[0]
        self.labels_batches = np.split(labels, np.arange(self.batch_features, len(labels), self.batch_features))
        self.features_batches = np.split(features, np.arange(self.batch_features, len(features), self.batch_features))
        self.__batch_index:np.ndarray = np.random.randint(len(self.labels_batches), size=self.nb_generations)

    def __select_random_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        features = self.features_batches[self.__batch_index[self.generation]]
        labels = self.labels_batches[self.__batch_index[self.generation]]
        return features, labels

    def record_energy(self, genomes:Dict[int, Genome_NN], records:Dict[str,Dict[str, np.ndarray]]):
            has_spike:bool = "spike" in self.record_layer
            has_augmented:bool = "augmented" in self.record_layer
            if has_spike == False and has_augmented == False: return

            for id, genome in genomes.items():
                if has_spike:
                    genome.info["energy_spikes_sum"] = records["spike"][id].sum()

                if has_augmented:
                    genome.info["energy_augmented_spikes_sum"] = records["augmented"][id].sum()